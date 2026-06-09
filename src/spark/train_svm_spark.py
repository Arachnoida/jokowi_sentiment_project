"""Latih & evaluasi SVM + TF-IDF memakai PySpark MLlib (jalur Big Data).

Padanan Spark dari ``src/modeling/train_svm_full14k.py`` (sklearn). Tujuannya
mendemonstrasikan pipeline ML terdistribusi, BUKAN menggantikan hasil sklearn —
keduanya dilaporkan berdampingan.

Kesetaraan yang dijaga (apple-to-apple dengan sklearn & IndoBERT):
  - Split train/val/test DETERMINISTIK dihitung sekali via ``split_version``
    (logika sama persis dgn jalur sklearn) lalu di-join ke Spark DataFrame.
  - Teks fitur ``svm`` sama (sudah diverifikasi 100% cocok di preprocess_spark).
  - Metrik test dihitung dgn ``evaluate`` yang sama (sklearn) -> definisi
    macro-F1/akurasi identik.

Beda yang melekat pada Spark MLlib (dicatat jujur untuk skripsi):
  - TF-IDF: CountVectorizer + IDF (Spark). Tidak ada opsi ``sublinear_tf``
    seperti TfidfVectorizer sklearn -> bobot term sedikit berbeda.
  - Multiclass: OneVsRest(LinearSVC) (Spark hanya punya SVM biner).
  - Regularisasi diparametri ``regParam`` (bukan ``C`` sklearn) -> grid beda.

Prasyarat: ``python -m src.spark.export_mongo`` lalu ``python -m src.spark.preprocess_spark``
(skrip ini membaca ``features_spark.parquet`` hasil preprocess Spark -> pipeline
preprocessing->training sepenuhnya via Spark).
Jalankan  : python -m src.spark.train_svm_spark
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.feature import CountVectorizer, IDF, NGram, SQLTransformer, Tokenizer
from pyspark.sql import functions as F
from sklearn.metrics import f1_score

from src.modeling.train_svm_full14k import (
    LABELS,
    VERSIONS,
    evaluate,
    split_version,
)
from src.spark.session import get_spark, hold_for_ui, parquet_dir, reports_dir

TEXT, LAB, IDC = "svm", "label_id", "comment_id"
# Grid native Spark (regParam menggantikan C; minDF menyamai min_df sklearn).
MIN_DF_GRID = [1.0, 2.0, 3.0]
REG_PARAM_GRID = [0.001, 0.01, 0.1]
VOCAB_SIZE = 50000
MAX_ITER = 50


def make_pipeline(min_df: float, reg_param: float) -> Pipeline:
    """Tokenizer -> unigram+bigram -> CountVectorizer -> IDF -> OneVsRest(LinearSVC)."""
    return Pipeline(
        stages=[
            Tokenizer(inputCol=TEXT, outputCol="tok"),
            NGram(n=2, inputCol="tok", outputCol="bg"),
            SQLTransformer(statement="SELECT *, concat(tok, bg) AS terms FROM __THIS__"),
            CountVectorizer(inputCol="terms", outputCol="tf", minDF=min_df, vocabSize=VOCAB_SIZE),
            IDF(inputCol="tf", outputCol="features"),
            OneVsRest(
                classifier=LinearSVC(maxIter=MAX_ITER, regParam=reg_param),
                labelCol=LAB,
                featuresCol="features",
                weightCol="w",
            ),
        ]
    )


def add_weights(df):
    """Bobot kelas 'balanced' = N / (k * n_kelas), dihitung pada set fit ini."""
    counts = {r[LAB]: r["n"] for r in df.groupBy(LAB).count().withColumnRenamed("count", "n").collect()}
    total = sum(counts.values())
    k = len(counts)
    w = None
    for lid, n in counts.items():
        val = total / (k * n)
        cond = F.col(LAB) == lid
        w = F.when(cond, F.lit(val)) if w is None else w.when(cond, F.lit(val))
    return df.withColumn("w", w.otherwise(F.lit(1.0)))


def macro_f1_on(model, dframe) -> float:
    pdf = model.transform(dframe).select(LAB, "prediction").toPandas()
    return f1_score(pdf[LAB], pdf["prediction"].astype(int), average="macro", zero_division=0)


def fold_dataframe(spark, sub_pdf: pd.DataFrame):
    """Bangun peta comment_id -> fold (train/val/test) dari split sklearn."""
    tr, va, te = split_version(sub_pdf)
    rows = (
        [(cid, "train") for cid in tr[IDC]]
        + [(cid, "val") for cid in va[IDC]]
        + [(cid, "test") for cid in te[IDC]]
    )
    return spark.createDataFrame(rows, [IDC, "fold"]), len(tr), len(va), len(te)


def run_version(spark, feats, members_pdf: pd.DataFrame, flag: str) -> dict:
    sub_pdf = members_pdf[members_pdf[flag]].copy()
    fold_df, _, _, _ = fold_dataframe(spark, sub_pdf)
    data = feats.join(F.broadcast(fold_df), on=IDC, how="inner")

    # Buang teks kosong dari train/val saja (test dibiarkan utuh -> identik sklearn).
    nonempty = F.length(F.trim(F.col(TEXT))) > 0
    train = data.filter((F.col("fold") == "train") & nonempty)
    val = data.filter((F.col("fold") == "val") & nonempty)
    test = data.filter(F.col("fold") == "test")
    trainval = data.filter((F.col("fold").isin("train", "val")) & nonempty)

    train_w = add_weights(train).cache()
    val_c = val.cache()

    best = None  # (macro_f1, min_df, reg_param)
    for min_df in MIN_DF_GRID:
        for reg in REG_PARAM_GRID:
            model = make_pipeline(min_df, reg).fit(train_w)
            score = macro_f1_on(model, val_c)
            if best is None or score > best[0]:
                best = (score, min_df, reg)

    # Refit pada train+val dgn param terbaik, lalu evaluasi test.
    final = make_pipeline(best[1], best[2]).fit(add_weights(trainval))
    pdf_test = final.transform(test).select(LAB, "prediction").toPandas()
    train_w.unpersist()
    val_c.unpersist()

    m = evaluate(pdf_test[LAB].tolist(), pdf_test["prediction"].astype(int).tolist())
    m["n_train"] = int(trainval.count())
    m["n_test"] = int(len(pdf_test))
    m["best_params"] = {"minDF": best[1], "regParam": best[2], "ngram": "(1,2)"}
    m["val_macro_f1"] = round(float(best[0]), 4)
    return m


def main() -> None:
    spark = get_spark("train-svm-spark")
    pq = parquet_dir()
    feat_path = pq / "features_spark.parquet"
    if not feat_path.exists():
        raise SystemExit("features_spark.parquet belum ada — jalankan dulu: "
                         "python -m src.spark.preprocess_spark")
    src = spark.read.parquet(str(feat_path))
    feats = src.select(IDC, F.col(TEXT), F.col(LAB).cast("int").alias(LAB)).cache()

    # Keanggotaan + flag versi (kecil) -> pandas untuk hitung split deterministik.
    flag_cols = [f for _, f in VERSIONS if f != "in_full14k"]
    members_pdf = src.select(IDC, LAB, *flag_cols).toPandas()
    for f in flag_cols:
        members_pdf[f] = members_pdf[f].fillna(False).astype(bool)
    members_pdf["in_full14k"] = True

    results = {}
    for name, flag in VERSIONS:
        m = run_version(spark, feats, members_pdf, flag)
        results[name] = m
        line = " ".join(f"{l[:3]}={m['per_class'][l]['f1']:.2f}" for l in LABELS)
        print(
            f"[{name:<18}] n_train={m['n_train']:<5} n_test={m['n_test']:<5} "
            f"macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} | {line} | {m['best_params']}"
        )

    rep = reports_dir()
    json.dump(
        {"model": "SVM+TF-IDF (PySpark MLlib)", "by_version": results},
        open(rep / "svm_spark_metrics.json", "w"),
        ensure_ascii=False,
        indent=2,
    )

    rows = []
    for name, _ in VERSIONS:
        m = results[name]
        rows.append(
            {
                "versi": name,
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "macro_F1": m["macro_f1"],
                "accuracy": m["accuracy"],
                "weighted_F1": m["weighted_f1"],
                **{f"F1_{l}": m["per_class"][l]["f1"] for l in LABELS},
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(rep / "svm_spark_table.csv", index=False)

    _plots(rep, comp, results)
    _cross_engine(rep, results)

    print("\nTersimpan ke", rep)
    print(comp.to_string(index=False))
    hold_for_ui(spark)
    spark.stop()


def _plots(rep, comp, results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in VERSIONS]
    metrics = ["macro_F1"] + [f"F1_{l}" for l in LABELS]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, met in enumerate(metrics):
        ax.bar(x + k * w, [comp.loc[comp.versi == n, met].values[0] for n in names], w, label=met)
    ax.set_xticks(x + w * 1.5, names, rotation=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.legend(ncol=4, fontsize=8)
    ax.set_title("SVM (PySpark MLlib) lintas versi dataset")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(rep / "svm_spark_compare.png", dpi=120)

    fig2, axes = plt.subplots(1, len(VERSIONS), figsize=(4 * len(VERSIONS), 3.6))
    for ax, (name, _) in zip(axes, VERSIONS):
        cm = np.array(results[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), [l[:3] for l in LABELS])
        ax.set_yticks(range(3), [l[:3] for l in LABELS])
        ax.set_title(f"{name}\nmacroF1={results[name]['macro_f1']:.3f}", fontsize=9)
        th = cm.max() / 2
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > th else "black")
    fig2.supxlabel("Prediksi")
    fig2.supylabel("Aktual")
    fig2.tight_layout()
    fig2.savefig(rep / "svm_spark_confusion.png", dpi=120)


def _cross_engine(rep, results) -> None:
    """Bandingkan macro-F1 SVM sklearn vs SVM Spark (jika hasil sklearn ada)."""
    sk_file = rep / "svm_full14k_metrics.json"
    if not sk_file.exists():
        print("\n(svm_full14k_metrics.json belum ada — lewati perbandingan sklearn vs Spark.)")
        return
    sk = json.load(open(sk_file))["by_version"]
    rows = []
    for name, _ in VERSIONS:
        s = sk.get(name, {}).get("macro_f1")
        p = results[name]["macro_f1"]
        rows.append(
            {
                "versi": name,
                "SVM_sklearn": s,
                "SVM_spark": p,
                "selisih": None if s is None else round(p - s, 4),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(rep / "svm_sklearn_vs_spark.csv", index=False)
    print("\nSVM sklearn vs Spark (macro-F1):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

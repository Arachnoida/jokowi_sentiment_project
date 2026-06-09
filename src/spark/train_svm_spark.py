"""Latih & evaluasi SVM + TF-IDF memakai PySpark MLlib (jalur Big Data).

Padanan Spark dari ``src/modeling/train_svm_full14k.py`` (sklearn). Tujuannya
mendemonstrasikan pipeline ML terdistribusi, BUKAN menggantikan hasil sklearn —
keduanya dilaporkan berdampingan. SATU dataset: full 14k komentar berlabel.

Kesetaraan yang dijaga (apple-to-apple dengan sklearn & IndoBERT):
  - Split train/val/test DETERMINISTIK dihitung sekali via ``split_version``
    (logika sama persis dgn jalur sklearn) lalu di-join ke Spark DataFrame.
  - Teks fitur ``svm`` sama (sudah diverifikasi 100% cocok di preprocess_spark).
  - Metrik test dihitung dgn ``evaluate`` yang sama (sklearn) -> definisi
    macro-F1/akurasi identik.

Untuk mendekati TfidfVectorizer(sublinear_tf=True, norm='l2') sklearn, pipeline ini
menambahkan stage kustom ``SublinearTF`` (tf -> 1+log(tf)) + ``Normalizer(p=2)`` (L2).

Beda STRUKTURAL yang tetap melekat (tak tersedia di MLlib, dicatat jujur utk skripsi):
  - Rumus IDF: Spark ``ln((n+1)/(df+1))`` vs sklearn ``ln((1+n)/(1+df))+1`` (beda +1).
  - Loss: Spark LinearSVC = hinge (OWL-QN); sklearn LinearSVC = squared-hinge.
  - Regularisasi ``regParam`` (bukan ``C`` sklearn) -> skala/grid beda.
  - Multiclass sama-sama OneVsRest (sklearn LinearSVC default ovr juga).

Prasyarat: ``python -m src.spark.export_mongo`` lalu ``python -m src.spark.preprocess_spark``
(skrip ini membaca ``features_spark.parquet`` hasil preprocess Spark -> pipeline
preprocessing->training sepenuhnya via Spark).
Jalankan  : python -m src.spark.train_svm_spark
"""
from __future__ import annotations

import json
import math

import pandas as pd
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.feature import (
    CountVectorizer,
    IDF,
    NGram,
    Normalizer,
    SQLTransformer,
    Tokenizer,
)
from pyspark.ml.linalg import SparseVector, Vectors, VectorUDT
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from sklearn.metrics import f1_score

from src.modeling.train_svm_full14k import LABELS, evaluate, split_version
from src.spark.session import get_spark, hold_for_ui, parquet_dir, reports_dir

TEXT, LAB, IDC = "svm", "label_id", "comment_id"
# Grid native Spark (regParam menggantikan C; minDF menyamai min_df sklearn).
MIN_DF_GRID = [1.0, 2.0, 3.0]
# L2-normalize membuat skala fitur kecil -> beri regParam lebih kecil juga.
REG_PARAM_GRID = [0.0001, 0.001, 0.01, 0.1]
VOCAB_SIZE = 50000
MAX_ITER = 50


def _sublinear(v):
    """tf -> 1 + log(tf) pada nilai non-nol (samakan sklearn sublinear_tf=True)."""
    if v is None:
        return v
    if isinstance(v, SparseVector):
        return Vectors.sparse(v.size, v.indices, [1.0 + math.log(x) for x in v.values])
    return Vectors.dense([0.0 if x == 0 else 1.0 + math.log(x) for x in v])


class SublinearTF(Transformer, HasInputCol, HasOutputCol,
                  DefaultParamsReadable, DefaultParamsWritable):
    """Stage Pipeline kustom: terapkan sublinear-TF (1+log) pada vektor TF."""

    def __init__(self, inputCol=None, outputCol=None):
        super().__init__()
        self._set(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, dataset):
        f = udf(_sublinear, VectorUDT())
        return dataset.withColumn(self.getOutputCol(), f(dataset[self.getInputCol()]))


def make_pipeline(min_df: float, reg_param: float) -> Pipeline:
    """Tokenizer -> unigram+bigram -> CountVectorizer -> sublinear-TF -> IDF ->
    L2-normalize -> OneVsRest(LinearSVC). sublinear-TF + L2-norm ditambahkan agar
    lebih dekat ke TfidfVectorizer(sublinear_tf=True, norm='l2') sklearn."""
    return Pipeline(
        stages=[
            Tokenizer(inputCol=TEXT, outputCol="tok"),
            NGram(n=2, inputCol="tok", outputCol="bg"),
            SQLTransformer(statement="SELECT *, concat(tok, bg) AS terms FROM __THIS__"),
            CountVectorizer(inputCol="terms", outputCol="tf", minDF=min_df, vocabSize=VOCAB_SIZE),
            SublinearTF(inputCol="tf", outputCol="tf_sub"),
            IDF(inputCol="tf_sub", outputCol="tfidf"),
            Normalizer(inputCol="tfidf", outputCol="features", p=2.0),
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


def fold_dataframe(spark, members_pdf: pd.DataFrame):
    """Bangun peta comment_id -> fold (train/val/test) dari split sklearn."""
    tr, va, te = split_version(members_pdf)
    rows = (
        [(cid, "train") for cid in tr[IDC]]
        + [(cid, "val") for cid in va[IDC]]
        + [(cid, "test") for cid in te[IDC]]
    )
    return spark.createDataFrame(rows, [IDC, "fold"]), len(tr), len(va), len(te)


def run(spark, feats, members_pdf: pd.DataFrame) -> dict:
    fold_df, _, _, _ = fold_dataframe(spark, members_pdf)
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

    # Keanggotaan (kecil) -> pandas untuk hitung split deterministik.
    members_pdf = src.select(IDC, LAB).toPandas()

    m = run(spark, feats, members_pdf)
    line = " ".join(f"{l[:3]}={m['per_class'][l]['f1']:.2f}" for l in LABELS)
    print(
        f"[full 14k] n_train={m['n_train']:<5} n_test={m['n_test']:<5} "
        f"macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} | {line} | {m['best_params']}"
    )

    rep = reports_dir()
    json.dump(
        {"model": "SVM+TF-IDF (PySpark MLlib)", "test": m},
        open(rep / "svm_spark_metrics.json", "w"),
        ensure_ascii=False,
        indent=2,
    )
    _confusion_plot(rep, m)
    _cross_engine(rep, m)

    print("\nTersimpan ke", rep)
    hold_for_ui(spark)
    spark.stop()


def _confusion_plot(rep, m) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(f"SVM (PySpark MLlib) — Test (macro-F1={m['macro_f1']:.3f})")
    th = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > th else "black")
    fig.tight_layout()
    fig.savefig(rep / "svm_spark_confusion.png", dpi=120)


def _cross_engine(rep, m) -> None:
    """Bandingkan macro-F1 SVM sklearn vs SVM Spark (jika hasil sklearn ada)."""
    sk_file = rep / "svm_full14k_metrics.json"
    if not sk_file.exists():
        print("\n(svm_full14k_metrics.json belum ada — lewati perbandingan sklearn vs Spark.)")
        return
    s = json.load(open(sk_file)).get("test", {}).get("macro_f1")
    if s is None:
        print("\n(svm_full14k_metrics.json belum format single-dataset — jalankan "
              "ulang train_svm_full14k.py utk perbandingan sklearn vs Spark.)")
        return
    p = m["macro_f1"]
    df = pd.DataFrame([{
        "SVM_sklearn": s,
        "SVM_spark": p,
        "selisih": round(p - s, 4),
    }])
    df.to_csv(rep / "svm_sklearn_vs_spark.csv", index=False)
    print("\nSVM sklearn vs Spark (macro-F1):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""EDA terdistribusi (Spark) untuk tahap preprocessing.

Memigrasikan bagian ANALISIS dari notebook preprocessing lama ke agregasi Spark:
  - distribusi kelas (groupBy count)
  - statistik panjang teks: mentah vs fitur SVM vs fitur BERT (avg + kuantil)
  - ukuran vocabulary SVM + token paling sering
  - % teks kosong setelah cleaning, % duplikat
  - cek 0-LEAKAGE antar split train/val/test (pakai split v5 deterministik)
  - top term DISKRIMINATIF per kelas (skor lift)

Semua hitungan via Spark (groupBy/explode/join) — inilah "kerja" Big Data yang
terlihat di Spark UI. Hasil -> outputs/reports/eda_spark.{json,png}.

Prasyarat: python -m src.spark.export_mongo
Jalankan : python -m src.spark.eda_spark
"""
from __future__ import annotations

import json

import pandas as pd
from pyspark.sql import functions as F

from src.modeling.train_svm_full14k import LABELS, split_version
from src.spark.session import get_spark, hold_for_ui, parquet_dir, reports_dir

TEXT, LAB, IDC = "svm", "label_id", "comment_id"
MIN_TOKEN_COUNT = 5   # ambang agar lift tidak didominasi token super-langka
TOP_N = 12


def _wc(col):
    """Jumlah kata yang aman: string kosong -> 0 (bukan 1)."""
    return F.when(F.length(F.trim(col)) == 0, F.lit(0)).otherwise(F.size(F.split(F.trim(col), r"\s+")))


def class_distribution(df) -> dict:
    rows = df.groupBy(LAB).count().orderBy(LAB).collect()
    return {LABELS[r[LAB]]: int(r["count"]) for r in rows}


def length_stats(df) -> dict:
    d = df.select(
        _wc(F.col("text")).alias("raw"),
        _wc(F.col("svm")).alias("svm"),
        _wc(F.col("bert")).alias("bert"),
    )
    agg = d.agg(
        F.avg("raw").alias("raw_avg"), F.avg("svm").alias("svm_avg"), F.avg("bert").alias("bert_avg"),
        F.max("raw").alias("raw_max"), F.max("svm").alias("svm_max"),
    ).collect()[0]
    med = {k: v for k, v in zip(["raw", "svm", "bert"], d.approxQuantile(["raw", "svm", "bert"], [0.5], 0.01))}
    return {
        "avg_words": {"raw": round(agg["raw_avg"], 2), "svm": round(agg["svm_avg"], 2), "bert": round(agg["bert_avg"], 2)},
        "median_words": {k: int(v[0]) for k, v in med.items()},
        "max_words": {"raw": int(agg["raw_max"]), "svm": int(agg["svm_max"])},
    }


def vocab_stats(tokens) -> dict:
    vocab = tokens.select("tok").distinct().count()
    top = (
        tokens.groupBy("tok").count().orderBy(F.desc("count")).limit(TOP_N).collect()
    )
    return {"vocab_size": int(vocab), "top_terms": [(r["tok"], int(r["count"])) for r in top]}


def quality_stats(df, total: int) -> dict:
    empty = df.filter(F.length(F.trim(F.col("svm"))) == 0).count()
    distinct_svm = df.filter(F.length(F.trim(F.col("svm"))) > 0).select("svm").distinct().count()
    nonempty = total - empty
    return {
        "n_total": total,
        "svm_empty": int(empty),
        "svm_empty_pct": round(100 * empty / total, 2),
        "svm_duplicate": int(nonempty - distinct_svm),
        "svm_duplicate_pct": round(100 * (nonempty - distinct_svm) / total, 2),
    }


def leakage_check(spark, df, members_pdf: pd.DataFrame) -> dict:
    """Cek teks svm identik yang bocor lintas split (train/val/test) pada v5."""
    tr, va, te = split_version(members_pdf)
    rows = (
        [(c, "train") for c in tr[IDC]] + [(c, "val") for c in va[IDC]] + [(c, "test") for c in te[IDC]]
    )
    fold = spark.createDataFrame(rows, [IDC, "fold"])
    joined = df.join(fold, on=IDC, how="inner").filter(F.length(F.trim(F.col("svm"))) > 0)
    # svm string yang muncul di >1 fold berbeda = potensi leakage.
    per_text = joined.groupBy("svm").agg(F.countDistinct("fold").alias("nfold"), F.count("*").alias("nrow"))
    leak = per_text.filter(F.col("nfold") > 1)
    n_leak_text = leak.count()
    n_leak_rows = leak.agg(F.coalesce(F.sum("nrow"), F.lit(0)).alias("s")).collect()[0]["s"]
    return {
        "id_overlap": 0,  # split by comment_id disjoint per konstruksi
        "duplicate_text_across_splits": int(n_leak_text),
        "rows_affected": int(n_leak_rows),
    }


def discriminative_terms(tokens) -> dict:
    """Top term per kelas berdasarkan lift = P(token|kelas) / P(token)."""
    total_tokens = tokens.count()
    overall = tokens.groupBy("tok").count().withColumnRenamed("count", "ct").filter(F.col("ct") >= MIN_TOKEN_COUNT)
    per_class = tokens.groupBy(LAB, "tok").count().withColumnRenamed("count", "c_in")
    class_tot = {r[LAB]: r["n"] for r in tokens.groupBy(LAB).count().withColumnRenamed("count", "n").collect()}

    joined = per_class.join(overall, on="tok", how="inner")
    out = {}
    for lid, ctot in sorted(class_tot.items()):
        rows = (
            joined.filter(F.col(LAB) == lid)
            .withColumn("lift", (F.col("c_in") / F.lit(ctot)) / (F.col("ct") / F.lit(total_tokens)))
            .orderBy(F.desc("lift"))
            .limit(TOP_N)
            .collect()
        )
        out[LABELS[lid]] = [(r["tok"], round(r["lift"], 2), int(r["c_in"])) for r in rows]
    return out


def _plot(rep, dist, length) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(list(dist.keys()), list(dist.values()), color=["#c0392b", "#7f8c8d", "#27ae60"])
    ax1.set_title("Distribusi kelas (14k)")
    ax1.set_ylabel("jumlah komentar")
    for i, v in enumerate(dist.values()):
        ax1.text(i, v + 40, str(v), ha="center", fontsize=9)

    stages = ["raw", "svm", "bert"]
    ax2.bar(stages, [length["avg_words"][s] for s in stages], color="#2980b9")
    ax2.set_title("Rata-rata jumlah kata per tahap")
    ax2.set_ylabel("kata")
    for i, s in enumerate(stages):
        ax2.text(i, length["avg_words"][s] + 0.1, length["avg_words"][s], ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(rep / "eda_spark.png", dpi=120)


def main() -> None:
    spark = get_spark("eda-spark")
    pq = parquet_dir()
    svm = spark.read.parquet(str(pq / "processed_svm.parquet")).select(
        IDC, F.col(LAB).cast("int").alias(LAB), "text", "svm"
    )
    bert = spark.read.parquet(str(pq / "processed_bert.parquet")).select(IDC, "bert")
    df = svm.join(bert, on=IDC, how="left").cache()
    total = df.count()

    tokens = df.select(LAB, F.explode(F.split(F.trim(F.col("svm")), r"\s+")).alias("tok")).filter(F.col("tok") != "")
    tokens = tokens.cache()

    members_pdf = df.select(IDC, LAB).toPandas()

    report = {
        "n_total": total,
        "class_distribution": class_distribution(df),
        "length": length_stats(df),
        "vocab": vocab_stats(tokens),
        "quality": quality_stats(df, total),
        "leakage_v5": leakage_check(spark, df, members_pdf),
        "discriminative_terms": discriminative_terms(tokens),
    }

    rep = reports_dir()
    json.dump(report, open(rep / "eda_spark.json", "w"), ensure_ascii=False, indent=2)
    _plot(rep, report["class_distribution"], report["length"])

    # Ringkasan ke konsol.
    print(f"\nTotal: {total} | kelas: {report['class_distribution']}")
    print(f"Rata-rata kata: {report['length']['avg_words']}")
    print(f"Vocab SVM: {report['vocab']['vocab_size']} | kosong: {report['quality']['svm_empty']} "
          f"({report['quality']['svm_empty_pct']}%) | duplikat: {report['quality']['svm_duplicate']} "
          f"({report['quality']['svm_duplicate_pct']}%)")
    lk = report["leakage_v5"]
    print(f"Leakage v5: id_overlap={lk['id_overlap']} | teks-sama-lintas-split={lk['duplicate_text_across_splits']} "
          f"({lk['rows_affected']} baris)")
    print("Top term diskriminatif (lift):")
    for cls, terms in report["discriminative_terms"].items():
        print(f"  {cls:<8}: " + ", ".join(f"{t}({l})" for t, l, _ in terms[:6]))
    print("\nTersimpan ke", rep / "eda_spark.json", "+ eda_spark.png")

    hold_for_ui(spark)
    spark.stop()


if __name__ == "__main__":
    main()

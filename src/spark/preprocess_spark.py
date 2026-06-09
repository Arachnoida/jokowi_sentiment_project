"""Preprocessing terdistribusi (Spark) dari teks mentah -> fitur SVM & BERT.

Tujuan: mendemonstrasikan tahap preprocessing sebagai pipeline Spark, sekaligus
memverifikasi bahwa hasil UDF Spark IDENTIK dengan fitur yang sudah tersimpan di
Mongo (kolom ``svm``/``bert``). Kalau match-rate ~100%, jalur Spark terbukti
setara dengan jalur sklearn/Mongo sebelumnya.

Input : data/spark_parquet/processed_{svm,bert}.parquet  (punya kolom `text`)
Output : data/spark_parquet/features_spark.parquet        (fitur hasil Spark)
Jalankan: python -m src.spark.preprocess_spark
"""
from __future__ import annotations

from pyspark.sql import functions as F

from src.spark.session import get_spark, parquet_dir
from src.spark.udf import make_bert_text_udf, make_svm_text_udf


def main() -> None:
    spark = get_spark("preprocess-spark")
    pq = parquet_dir()

    svm_src = spark.read.parquet(str(pq / "processed_svm.parquet"))
    bert_src = spark.read.parquet(str(pq / "processed_bert.parquet")).select(
        "comment_id", F.col("bert").alias("bert_stored")
    )

    # Hitung ulang fitur dari teks mentah via UDF Spark.
    feats = (
        svm_src.select(
            "comment_id",
            "label_id",
            F.col("text"),
            F.col("svm").alias("svm_stored"),
            make_svm_text_udf(F.col("text")).alias("svm_spark"),
            make_bert_text_udf(F.col("text")).alias("bert_spark"),
        )
        .join(bert_src, on="comment_id", how="left")
    )
    feats = feats.cache()

    total = feats.count()
    svm_match = feats.filter(F.col("svm_spark") == F.col("svm_stored")).count()
    bert_match = feats.filter(F.col("bert_spark") == F.col("bert_stored")).count()
    print(f"Total baris           : {total}")
    print(f"SVM  cocok vs Mongo   : {svm_match}/{total} ({100*svm_match/total:.2f}%)")
    print(f"BERT cocok vs Mongo   : {bert_match}/{total} ({100*bert_match/total:.2f}%)")

    out = pq / "features_spark.parquet"
    (
        feats.select("comment_id", "label_id", "svm_spark", "bert_spark")
        .write.mode("overwrite")
        .parquet(str(out))
    )
    print(f"Fitur Spark tersimpan -> {out}")
    spark.stop()


if __name__ == "__main__":
    main()

"""Regenerasi koleksi ``processed_svm`` & ``processed_bert`` di MongoDB Atlas via Spark.

Menghitung ULANG teks fitur SVM (preprocess + stem Sastrawi) dan BERT (cleaning
minimal) dari teks mentah ``raw_comments`` (semua yang berlabel = SATU dataset full
14k), terdistribusi pada worker Spark, lalu menulis dokumen BERSIH ke Mongo
(schema seragam, TANPA flag versi apa pun).

Kenapa pola ini cocok untuk cluster TANPA shared filesystem:
  - Data di-*parallelize* dari driver (``createDataFrame``), bukan baca file ->
    tak butuh replikasi parquet ke tiap node.
  - Hasil di-*collect* ke driver, lalu ditulis ke Mongo via pymongo ->
    tak ada commit file lintas-mesin (yang rusak tanpa HDFS).
  - UDF Python (Sastrawi + src.text_normalizer) jalan di executor -> tiap worker
    wajib punya venv + paket; lihat docs/spark_cluster_kelompok.md.

Schema dokumen hasil:
  processed_svm  : comment_id, video_id, text, svm,  label, label_id
  processed_bert : comment_id, video_id, text, bert, label, label_id

Jalankan (cluster):
  SPARK_MASTER=spark://<ip>:7077 SPARK_DRIVER_HOST=<ip> \
      .venv/bin/python -m src.spark.regenerate_processed_mongo
Jalankan (lokal): python -m src.spark.regenerate_processed_mongo
"""
from __future__ import annotations

import os
import time

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne
from pyspark.sql import functions as F

from src.spark.session import get_spark, hold_for_ui, project_root
from src.spark.udf import make_bert_text_udf, make_svm_text_udf

DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
LABELS = ["Negatif", "Netral", "Positif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
N_PARTITIONS = 8  # >1 agar task tersebar ke beberapa worker


def _connect(tries: int = 8) -> MongoClient:
    load_dotenv(project_root() / ".env")
    uri = os.environ["MONGO_URI"]
    last = None
    for attempt in range(1, tries + 1):
        try:
            c = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
            c.admin.command("ping")
            return c
        except Exception as exc:  # noqa: BLE001 - retry transient SSL/DNS
            last = exc
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Gagal koneksi Mongo: {last}")


def _write_clean(col, records: list[dict], feat: str) -> int:
    """Replace_one upsert -> dokumen ter-overwrite ke schema bersih (drop field lama)."""
    ops = []
    for r in records:
        doc = {
            "comment_id": r["comment_id"],
            "video_id": r.get("video_id"),
            "text": r["text"],
            feat: r[feat],
            "label": r["label"],
            "label_id": int(LABEL2ID[r["label"]]),
        }
        ops.append(ReplaceOne({"comment_id": r["comment_id"]}, doc, upsert=True))
    written = 0
    for i in range(0, len(ops), 1000):
        res = col.bulk_write(ops[i : i + 1000], ordered=False)
        written += res.upserted_count + res.modified_count
    return written


def main() -> None:
    client = _connect()
    db = client[DB]

    # 1) Sumber kebenaran teks: raw_comments yang berlabel (driver-side read).
    docs = list(
        db["raw_comments"].find(
            {"label": {"$exists": True}},
            {"_id": 0, "comment_id": 1, "video_id": 1, "text": 1, "label": 1},
        )
    )
    pdf = pd.DataFrame(docs)
    pdf["text"] = pdf["text"].fillna("").astype(str)
    pdf["video_id"] = pdf.get("video_id", pd.Series([None] * len(pdf))).astype(object)
    print(f"raw_comments berlabel: {len(pdf)}")

    # 2) Hitung fitur SVM & BERT terdistribusi di executor (UDF Python).
    spark = get_spark("regenerate-processed")
    sdf = spark.createDataFrame(pdf).repartition(N_PARTITIONS)
    out = sdf.select(
        "comment_id",
        "video_id",
        "text",
        "label",
        make_svm_text_udf(F.col("text")).alias("svm"),
        make_bert_text_udf(F.col("text")).alias("bert"),
    )
    rows = [r.asDict() for r in out.collect()]
    print(f"fitur dihitung (Spark): {len(rows)} baris")

    # 3) Tulis dokumen BERSIH ke Mongo (schema seragam, tanpa flag).
    n_svm = _write_clean(db["processed_svm"], rows, "svm")
    n_bert = _write_clean(db["processed_bert"], rows, "bert")
    print(f"processed_svm  ditulis: {n_svm} | total dok: {db['processed_svm'].count_documents({})}")
    print(f"processed_bert ditulis: {n_bert} | total dok: {db['processed_bert'].count_documents({})}")

    # 4) Ringkas field (verifikasi tak ada flag tersisa).
    for name in ("processed_svm", "processed_bert"):
        d = db[name].find_one({}, {"_id": 0})
        print(f"  {name} fields: {sorted(d.keys())}")

    hold_for_ui(spark)
    spark.stop()
    client.close()


if __name__ == "__main__":
    main()

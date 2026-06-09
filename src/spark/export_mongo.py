"""Ekspor sekali jalan: MongoDB Atlas -> Parquet (sumber baca untuk Spark).

Kenapa lewat Parquet, bukan konektor Mongo-Spark langsung?
  - Konektor ``mongo-spark`` butuh jar tambahan + koordinat versi yang cocok
    dengan Spark 4 (rapuh, lambat di-resolve). Snapshot Parquet sekali jalan:
    lebih reproducible, offline-friendly, dan cukup untuk 14k baris.
  - URI Mongo TIDAK pernah dicetak/di-commit (dibaca dari .env).

Koleksi yang diekspor (keduanya sudah self-contained: punya teks mentah, fitur,
label, label_id, dan flag versi):
  - processed_svm  -> data/spark_parquet/processed_svm.parquet
  - processed_bert -> data/spark_parquet/processed_bert.parquet

Jalankan: python -m src.spark.export_mongo
"""
from __future__ import annotations

import os
import time

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from src.spark.session import parquet_dir, project_root

DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
FLAGS = ["in_set6k", "in_balanced_set", "in_set10k", "in_balanced10k"]
COLLECTIONS = {
    "processed_svm": ["comment_id", "video_id", "text", "svm", "label", "label_id", *FLAGS],
    "processed_bert": ["comment_id", "video_id", "text", "bert", "label", "label_id", *FLAGS],
}


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
    raise RuntimeError(f"Gagal koneksi Mongo setelah {tries} percobaan: {last}")


def export_collection(db, name: str, fields: list[str], out_dir) -> int:
    proj = {"_id": 0, **{f: 1 for f in fields}}
    docs = list(db[name].find({"label": {"$exists": True}}, proj))
    df = pd.DataFrame(docs)
    # Normalisasi: flag boolean yang hilang -> False; teks/fitur kosong -> "".
    for f in FLAGS:
        if f not in df.columns:
            df[f] = False
        df[f] = df[f].fillna(False).astype(bool)
    for col in ("text", "svm", "bert"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)
    out = out_dir / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return len(df)


def main() -> None:
    client = _connect()
    db = client[DB]
    out_dir = parquet_dir()
    print(f"Ekspor ke {out_dir}")
    for name, fields in COLLECTIONS.items():
        n = export_collection(db, name, fields, out_dir)
        print(f"  {name:<16} -> {n} baris")
    client.close()
    print("Selesai. Spark dapat membaca snapshot Parquet ini secara offline.")


if __name__ == "__main__":
    main()

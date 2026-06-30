"""Push daftar comment_id subset (mis. balanced_3000.csv) ke koleksi Mongo khusus.

Agar notebook Colab IndoBERT TIDAK perlu upload CSV manual: subset disimpan sebagai
koleksi tersendiri (default `balanced3k_ids`, hanya {comment_id}) sehingga schema
koleksi utama (raw_comments/processed_bert) tetap bersih. Notebook cukup baca koleksi
ini lalu filter processed_bert.

Idempotent: drop + insert ulang.

  python -m src.modeling.push_subset_ids                                  # balanced_3000 -> balanced3k_ids
  python -m src.modeling.push_subset_ids --csv <path> --collection <name>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/labeling/balanced_3000.csv")
    ap.add_argument("--collection", default="balanced3k_ids")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    ids = pd.read_csv(args.csv, usecols=["comment_id"])["comment_id"].astype(str).tolist()

    uri = os.environ["MONGO_URI"]
    dbn = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
    db = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)[dbn]
    col = db[args.collection]
    col.drop()
    col.insert_many([{"comment_id": c} for c in ids])
    col.create_index("comment_id")
    print(f"{args.collection}: {col.count_documents({})} comment_id ditulis (dari {args.csv}).")


if __name__ == "__main__":
    main()

"""Tandai anggota subset (mis. balanced_3000.csv) dgn FLAG boolean di processed_bert.

Agar notebook Colab IndoBERT self-contained (gaya indobert_finetune_colab_variant.ipynb):
baca `processed_bert` lewat filter `{FLAG: True}` — tanpa upload CSV / clone repo, cukup
MONGO_URI. Idempotent: unset flag lama lalu set True pada comment_id subset.

  python -m src.modeling.push_subset_ids                                   # balanced_3000 -> flag in_balanced3k
  python -m src.modeling.push_subset_ids --csv <path> --flag <field> --collection <name>
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
    ap.add_argument("--flag", default="in_balanced3k", help="Nama field boolean penanda subset.")
    ap.add_argument("--collection", default="processed_bert",
                    help="Koleksi yang ditandai (default processed_bert; bisa 'raw_comments').")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    ids = pd.read_csv(args.csv, usecols=["comment_id"])["comment_id"].astype(str).tolist()

    uri = os.environ["MONGO_URI"]
    dbn = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
    col = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)[dbn][args.collection]

    col.update_many({args.flag: {"$exists": True}}, {"$unset": {args.flag: ""}})
    r = col.update_many({"comment_id": {"$in": ids}}, {"$set": {args.flag: True}})
    n = col.count_documents({args.flag: True})
    print(f"{args.collection}.{args.flag}: matched={r.matched_count} set True -> total {n} (subset {len(ids)} dari {args.csv})")


if __name__ == "__main__":
    main()

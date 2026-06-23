"""Backfill `processed_bert` untuk komentar berlabel yang belum punya teks fitur BERT.

- Teks fitur BERT memakai pipeline IDENTIK notebook preprocessing_indobert:
  make_text_bert = clean_for_bert (cleaning minimal; morfologi/negasi terjaga).
- Upsert dokumen yang HILANG saja (per comment_id). Tidak menghapus apa pun.
- Proyek memakai SATU dataset (semua komentar berlabel) — tanpa flag versi.
"""
import os
import time

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from src.text_normalizer import clean_for_bert

LABELS = ["Negatif", "Netral", "Positif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")


# buka koneksi MongoDB Atlas (retry transien SSL/DNS), verifikasi via ping
def _connect(tries: int = 6) -> MongoClient:
    load_dotenv()
    uri = os.environ["MONGO_URI"]
    last = None
    for attempt in range(1, tries + 1):
        try:
            c = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
            c.admin.command("ping")
            return c
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Gagal koneksi Mongo: {last}")


# [JALUR B - BERT] tambal di 1 mesin (pandas): hanya comment_id yang BELUM ada di processed_bert
def main() -> None:
    client = _connect()
    db = client[DB]
    rc, pb = db["raw_comments"], db["processed_bert"]

    have = {d["comment_id"] for d in pb.find({}, {"comment_id": 1, "_id": 0})}
    proj = {"_id": 0, "comment_id": 1, "video_id": 1, "text": 1, "label": 1}
    labeled = list(rc.find({"label": {"$exists": True}}, proj))
    missing = [d for d in labeled if d["comment_id"] not in have]
    print(f"berlabel={len(labeled)} | sudah di processed_bert={len(have)} | backfill={len(missing)}")

    if missing:
        df = pd.DataFrame(missing)
        df["bert"] = df["text"].astype(str).map(lambda t: clean_for_bert(t or ""))
        df["label_id"] = df["label"].map(LABEL2ID)
        empty = int((df["bert"].str.len() == 0).sum())
        print(f"bert kosong setelah cleaning: {empty} (tetap diupsert agar split kanonik konsisten)")

        ops = []
        for r in df.to_dict("records"):
            doc = {
                "comment_id": r["comment_id"],
                "video_id": r.get("video_id"),
                "text": r["text"],
                "bert": r["bert"],
                "label": r["label"],
                "label_id": int(r["label_id"]),
            }
            ops.append(UpdateOne({"comment_id": r["comment_id"]}, {"$set": doc}, upsert=True))
        pb.bulk_write(ops, ordered=False)

    print(f"processed_bert sekarang: {pb.count_documents({})} dok")


if __name__ == "__main__":
    main()

"""Backfill `processed_svm` untuk komentar berlabel yang belum punya teks fitur SVM.

Memakai pipeline IDENTIK dengan notebook preprocessing_svm:
  preprocess_svm_python (clean -> slang -> stopword)  ->  Sastrawi stem (string utuh).
Stemmer dibuat SEKALI dan men-stem seluruh string sekaligus (bukan per-token) agar
hasil persis sama dengan 9996 baris yang sudah ada.

Hanya menambah dokumen yang HILANG (upsert per comment_id). Tidak menghapus apa pun,
tidak menyentuh flag versi pada raw_comments. Flag versi pada dok baru disalin apa
adanya dari raw_comments (umumnya False untuk 4107 komentar tambahan).
"""
import os
import time

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from src.text_normalizer import preprocess_svm_python

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
        except Exception as exc:  # transient SSL/DNS
            last = exc
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Gagal koneksi Mongo: {last}")


# [JALUR B - SVM] tambal di 1 mesin (pandas): hanya comment_id yang BELUM ada di processed_svm
def main() -> None:
    client = _connect()
    db = client[DB]
    rc, ps = db["raw_comments"], db["processed_svm"]

    have = {d["comment_id"] for d in ps.find({}, {"comment_id": 1, "_id": 0})}
    proj = {"_id": 0, "comment_id": 1, "video_id": 1, "text": 1, "label": 1}
    labeled = list(rc.find({"label": {"$exists": True}}, proj))
    missing = [d for d in labeled if d["comment_id"] not in have]
    print(f"berlabel={len(labeled)} | sudah di processed_svm={len(have)} | backfill={len(missing)}")
    if not missing:
        print("Tidak ada yang perlu di-backfill.")
        return

    # Stemmer dibuat SEKALI; men-stem string utuh (samakan dgn notebook make_text_svm).
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    stemmer = StemmerFactory().create_stemmer()
    # Lindungi kata "se-" bermuatan sikap dari over-stemming (setuju->tuju, dst.).
    # Harus sebelum stem pertama (CachedStemmer meng-cache). Samakan dgn src/spark/udf.py.
    _inner = getattr(stemmer, "delegatedStemmer", stemmer)
    for _w in ("setuju", "sependapat", "sepaham", "sepakat",
               "seting", "setting", "setel", "mentri"):
        _inner.dictionary.add(_w)

    # sama persis dgn udf.make_svm_text: preprocess lalu stem string penuh
    def make_text_svm(text: str) -> str:
        pre = preprocess_svm_python(text or "")
        return stemmer.stem(pre) if pre else pre

    df = pd.DataFrame(missing)
    df["svm"] = df["text"].astype(str).map(make_text_svm)
    df["label_id"] = df["label"].map(LABEL2ID)

    empty = int((df["svm"].str.len() == 0).sum())
    print(f"svm kosong setelah preprocessing: {empty} (tetap diupsert agar split kanonik konsisten)")

    ops = []
    for r in df.to_dict("records"):
        doc = {
            "comment_id": r["comment_id"],
            "video_id": r.get("video_id"),
            "text": r["text"],
            "svm": r["svm"],
            "label": r["label"],
            "label_id": int(r["label_id"]),
        }
        ops.append(UpdateOne({"comment_id": r["comment_id"]}, {"$set": doc}, upsert=True))

    ps.bulk_write(ops, ordered=False)
    print(f"processed_svm sekarang: {ps.count_documents({})} dok")


if __name__ == "__main__":
    main()

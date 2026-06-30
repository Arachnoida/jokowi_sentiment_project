"""Push koreksi re-label LLM pass-2 (Opus) ke MongoDB + master CSV.

Label hidup di DUA tempat di Mongo:
  - ``raw_comments.label``    → dibaca trainer SVM (src.modeling.train_svm_full14k)
  - ``processed_bert.label``  → dibaca trainer IndoBERT (src.modeling.train_indobert, Colab)
Keduanya harus diperbarui agar SVM & IndoBERT memakai label baru.

Skrip ini:
  1. Gabungkan kandidat suspect + output JSONL agen Opus → koreksi (baris berubah saja).
  2. Tulis catatan durable: outputs/labeling/relabel_pass2_opus_<date>.csv
  3. Update master outputs/labeling/labeling_dataset.csv (label/confidence/annotator/notes).
  4. Push $set ke raw_comments & processed_bert (bulk, idempotent).

Jalankan:
  python -m src.modeling.push_relabel_to_mongo --candidates <csv> --out-dir <dir>            # dry-run
  python -m src.modeling.push_relabel_to_mongo --candidates <csv> --out-dir <dir> --commit
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from src.modeling.apply_relabel_rebuild import load_pass2

ANNOTATOR = "claude-opus-relabel-20260630"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--durable", default="relabel_pass2_opus_20260630.csv",
                    help="Nama file catatan durable di outputs/labeling/.")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    root = _repo_root()
    load_dotenv(root / ".env")

    pass2 = load_pass2(Path(args.candidates), Path(args.out_dir))
    changed = pass2[pass2["label_pass1"] != pass2["label_pass2"]].copy()
    print(f"Re-label: {len(pass2)} total | berubah: {len(changed)}")
    print(changed.groupby(["label_pass1", "label_pass2"]).size().to_string())

    # 1. catatan durable (semua hasil pass-2, bukan cuma yg berubah)
    cand = pd.read_csv(args.candidates)
    cand["comment_id"] = cand["comment_id"].astype(str)
    durable = cand.merge(pass2, on="comment_id", how="left")
    dpath = root / "outputs" / "labeling" / args.durable

    # 2. update master labeling_dataset.csv
    master_path = root / "outputs" / "labeling" / "labeling_dataset.csv"
    master = pd.read_csv(master_path)
    master["comment_id"] = master["comment_id"].astype(str)
    c = changed.set_index("comment_id")
    m = master["comment_id"].isin(c.index)
    master.loc[m, "label"] = master.loc[m, "comment_id"].map(c["label_pass2"])
    master.loc[m, "confidence"] = master.loc[m, "comment_id"].map(c["conf_pass2"])
    master.loc[m, "annotator"] = ANNOTATOR
    print(f"\nMaster CSV: {int(m.sum())} baris akan di-update (dari {len(master)}).")

    # 3. Mongo bulk ops
    uri = os.environ["MONGO_URI"]
    dbn = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
    ops_raw, ops_bert = [], []
    for cid, r in c.iterrows():
        ops_raw.append(UpdateOne(
            {"comment_id": cid},
            {"$set": {"label": r["label_pass2"], "confidence": float(r["conf_pass2"]),
                      "annotator": ANNOTATOR}},
        ))
        ops_bert.append(UpdateOne(
            {"comment_id": cid}, {"$set": {"label": r["label_pass2"]}},
        ))

    if not args.commit:
        print(f"\n[dry-run] akan tulis durable -> {dpath}")
        print(f"[dry-run] akan update master -> {master_path}")
        print(f"[dry-run] akan push {len(ops_raw)} ke raw_comments & {len(ops_bert)} ke processed_bert.")
        print("Tambah --commit untuk eksekusi.")
        return

    durable.to_csv(dpath, index=False)
    print(f"Durable ditulis -> {dpath}")
    master.to_csv(master_path, index=False)
    print(f"Master di-update -> {master_path}")

    client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
    db = client[dbn]
    r1 = db["raw_comments"].bulk_write(ops_raw, ordered=False)
    r2 = db["processed_bert"].bulk_write(ops_bert, ordered=False)
    print(f"raw_comments: matched={r1.matched_count} modified={r1.modified_count}")
    print(f"processed_bert: matched={r2.matched_count} modified={r2.modified_count}")
    client.close()


if __name__ == "__main__":
    main()

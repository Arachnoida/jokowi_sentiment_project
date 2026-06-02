"""
src/push_labels_to_mongo.py

Tulis label hasil pelabelan (claude-llm) ke koleksi `raw_comments` di MongoDB
Atlas, dicocokkan via `comment_id`. Menambahkan field: label, annotator,
confidence, notes — plus flag `in_balanced_set` (True untuk komentar yang masuk
dataset balanced 1.000/kelas).

Sumber label  : outputs/labeling/labeling_dataset.csv  (6.000 berlabel)
Sumber balanced: outputs/labeling/balanced_1000.csv     (3.000 comment_id)

Idempotent: aman dijalankan ulang (pakai update_one $set, bukan insert).

Contoh:
    python -m src.push_labels_to_mongo --dry-run     # cek, tak menulis
    python -m src.push_labels_to_mongo --no-dry-run  # eksekusi
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Set, Tuple

from pymongo import UpdateOne

from configs.config import Config
from src.mongo_utils import get_collection
from src.utils import setup_logger

logger = setup_logger("push_labels_mongo")

ROOT = Path(__file__).resolve().parent.parent
SRC_LABELS = ROOT / "outputs" / "labeling" / "labeling_dataset.csv"
SRC_BALANCED = ROOT / "outputs" / "labeling" / "balanced_1000.csv"
VALID = {"Positif", "Negatif", "Netral"}


def _read_labels(path: Path) -> Dict[str, dict]:
    """comment_id -> {label, annotator, confidence, notes} untuk baris berlabel valid."""
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = (row.get("comment_id") or "").strip()
            lab = (row.get("label") or "").strip()
            if not cid or lab not in VALID:
                continue
            conf_raw = (row.get("confidence") or "").strip()
            try:
                conf = float(conf_raw) if conf_raw else None
            except ValueError:
                conf = None
            out[cid] = {
                "label": lab,
                "annotator": (row.get("annotator") or "").strip() or None,
                "confidence": conf,
                "notes": (row.get("notes") or "").strip() or None,
            }
    return out


def _read_balanced_ids(path: Path) -> Set[str]:
    if not path.exists():
        logger.warning("File balanced tidak ada: %s (flag in_balanced_set dilewati).", path)
        return set()
    ids: Set[str] = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = (row.get("comment_id") or "").strip()
            if cid:
                ids.add(cid)
    return ids


def push(dry_run: bool) -> None:
    labels = _read_labels(SRC_LABELS)
    balanced = _read_balanced_ids(SRC_BALANCED)
    logger.info("Label sumber: %d | balanced ids: %d", len(labels), len(balanced))

    col = get_collection(Config.mongo.URI, Config.mongo.DB_NAME, Config.mongo.COLLECTION_RAW)

    # Hanya update dokumen yang comment_id-nya memang ada di koleksi.
    existing = set(
        d["comment_id"]
        for d in col.find({"comment_id": {"$in": list(labels)}}, {"comment_id": 1, "_id": 0})
    )
    missing = len(labels) - len(existing)
    logger.info("Cocok di raw_comments: %d | label tanpa dokumen: %d", len(existing), missing)

    ops = []
    for cid in existing:
        data = dict(labels[cid])
        data["in_balanced_set"] = cid in balanced
        ops.append(UpdateOne({"comment_id": cid}, {"$set": data}))

    n_bal = sum(1 for cid in existing if cid in balanced)
    logger.info("Akan update %d dok (%d ditandai in_balanced_set=True).", len(ops), n_bal)

    if dry_run:
        logger.info("DRY-RUN: tidak menulis. Jalankan --no-dry-run untuk eksekusi.")
        return

    res = col.bulk_write(ops, ordered=False)
    logger.info("SELESAI: matched=%d modified=%d", res.matched_count, res.modified_count)


def main() -> None:
    ap = argparse.ArgumentParser(description="Push label claude-llm ke raw_comments (Mongo).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    g.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()
    push(args.dry_run)


if __name__ == "__main__":
    main()

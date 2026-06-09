"""
src/push_labels_to_mongo.py

Tulis label hasil pelabelan (claude-llm) ke koleksi `raw_comments` di MongoDB
Atlas (dicocokkan via `comment_id`). Proyek memakai SATU dataset (semua komentar
berlabel) — tidak ada lagi flag versi subset.

Field label yang ditulis: label, annotator, confidence, notes.

Sumber:
  - outputs/labeling/labeling_dataset.csv  (komentar + label)

Idempotent (update_one $set). Contoh:
    python -m src.push_labels_to_mongo --dry-run
    python -m src.push_labels_to_mongo --no-dry-run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

from pymongo import UpdateOne

from configs.config import Config
from src.mongo_utils import get_collection
from src.utils import setup_logger

logger = setup_logger("push_labels_mongo")

ROOT = Path(__file__).resolve().parent.parent
LBL = ROOT / "outputs" / "labeling"
VALID = {"Positif", "Negatif", "Netral"}


def _read_master(path: Path):
    """Kembalikan labels: cid -> {label, annotator, confidence, notes}."""
    labels: Dict[str, dict] = {}
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
            labels[cid] = {
                "label": lab,
                "annotator": (row.get("annotator") or "").strip() or None,
                "confidence": conf,
                "notes": (row.get("notes") or "").strip() or None,
            }
    return labels


def push(dry_run: bool) -> None:
    labels = _read_master(LBL / "labeling_dataset.csv")
    logger.info("Label terbaca: %d", len(labels))

    col = get_collection(Config.mongo.URI, Config.mongo.DB_NAME, Config.mongo.COLLECTION_RAW)
    existing = set(d["comment_id"] for d in
                   col.find({"comment_id": {"$in": list(labels)}}, {"comment_id": 1, "_id": 0}))
    logger.info("Cocok di raw_comments: %d | label tanpa dokumen: %d",
                len(existing), len(labels) - len(existing))

    ops = [UpdateOne({"comment_id": cid}, {"$set": dict(labels[cid])}) for cid in existing]

    logger.info("Akan update %d dokumen dgn label (satu dataset, tanpa flag versi).", len(ops))
    if dry_run:
        logger.info("DRY-RUN: tidak menulis. Jalankan --no-dry-run untuk eksekusi.")
        return
    res = col.bulk_write(ops, ordered=False)
    logger.info("SELESAI: matched=%d modified=%d", res.matched_count, res.modified_count)


def main() -> None:
    ap = argparse.ArgumentParser(description="Push label (claude-llm) ke raw_comments (Mongo).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    g.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()
    push(args.dry_run)


if __name__ == "__main__":
    main()

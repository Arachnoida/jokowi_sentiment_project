"""
src/push_labels_to_mongo.py

Tulis label hasil pelabelan (claude-llm) ke koleksi `raw_comments` di MongoDB
Atlas (dicocokkan via `comment_id`), beserta **flag 4 versi dataset**:

  - `in_set6k`        : v1 — 6.000 berlabel pertama (distribusi alami / imbalanced)
  - `in_balanced_set` : v2 — balanced 1.000/kelas (3.000) dari pool 6k  [nama lama dipertahankan]
  - `in_set10k`       : v3 — seluruh 10.000 berlabel (imbalanced)
  - `in_balanced10k`  : v4 — balanced ke kelas terlangka (~1.936/kelas) dari pool 10k

Field label yang ditulis: label, annotator, confidence, notes.

Sumber:
  - outputs/labeling/labeling_dataset.csv  (10.000 baris; urutan baris = set6k/set10k)
  - outputs/labeling/balanced_1000.csv     (comment_id v2)
  - outputs/labeling/balanced_10k.csv      (comment_id v4)

Idempotent (update_one $set). Contoh:
    python -m src.push_labels_to_mongo --dry-run
    python -m src.push_labels_to_mongo --no-dry-run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Set

from pymongo import UpdateOne

from configs.config import Config
from src.mongo_utils import get_collection
from src.utils import setup_logger

logger = setup_logger("push_labels_mongo")

ROOT = Path(__file__).resolve().parent.parent
LBL = ROOT / "outputs" / "labeling"
VALID = {"Positif", "Negatif", "Netral"}


def _read_master(path: Path):
    """Kembalikan (labels: cid->fields, set6k_ids, set10k_ids) berurut sesuai baris CSV."""
    labels: Dict[str, dict] = {}
    set6k: Set[str] = set()
    set10k: Set[str] = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row_i, row in enumerate(csv.DictReader(f)):
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
            set10k.add(cid)
            if row_i < 6000:
                set6k.add(cid)
    return labels, set6k, set10k


def _read_ids(path: Path) -> Set[str]:
    if not path.exists():
        logger.warning("File tidak ada: %s (flag dilewati).", path)
        return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {(r.get("comment_id") or "").strip() for r in csv.DictReader(f) if r.get("comment_id")}


def push(dry_run: bool) -> None:
    labels, set6k, set10k = _read_master(LBL / "labeling_dataset.csv")
    bal3k = _read_ids(LBL / "balanced_1000.csv")
    bal10k = _read_ids(LBL / "balanced_10k.csv")
    logger.info("Label: %d | v1(6k): %d | v2(3k): %d | v3(10k): %d | v4(bal10k): %d",
                len(labels), len(set6k), len(bal3k), len(set10k), len(bal10k))

    col = get_collection(Config.mongo.URI, Config.mongo.DB_NAME, Config.mongo.COLLECTION_RAW)
    existing = set(d["comment_id"] for d in
                   col.find({"comment_id": {"$in": list(labels)}}, {"comment_id": 1, "_id": 0}))
    logger.info("Cocok di raw_comments: %d | label tanpa dokumen: %d",
                len(existing), len(labels) - len(existing))

    ops = []
    for cid in existing:
        data = dict(labels[cid])
        data["in_set6k"] = cid in set6k
        data["in_balanced_set"] = cid in bal3k       # v2 (nama lama dipertahankan)
        data["in_set10k"] = cid in set10k
        data["in_balanced10k"] = cid in bal10k
        ops.append(UpdateOne({"comment_id": cid}, {"$set": data}))

    logger.info("Akan update %d dokumen dgn label + 4 flag versi.", len(ops))
    if dry_run:
        logger.info("DRY-RUN: tidak menulis. Jalankan --no-dry-run untuk eksekusi.")
        return
    res = col.bulk_write(ops, ordered=False)
    logger.info("SELESAI: matched=%d modified=%d", res.matched_count, res.modified_count)


def main() -> None:
    ap = argparse.ArgumentParser(description="Push label + flag 4 versi ke raw_comments (Mongo).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    g.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()
    push(args.dry_run)


if __name__ == "__main__":
    main()

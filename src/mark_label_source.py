"""Tandai PROVENANCE label: ``llm`` (default) vs ``manual`` (anotasi manusia).

Menambahkan field ``label_source`` ke koleksi ``raw_comments``, ``processed_svm``,
``processed_bert`` di MongoDB Atlas:
  - ``llm``    : label dari ``claude-llm`` (kondisi awal SEMUA dok).
  - ``manual`` : komentar yang BENAR-BENAR dianotasi manusia di Label Studio
    (project id=4 "Verifikasi Manual"); ``manual_label`` menyimpan label manusianya.

Pakai:
  python -m src.mark_label_source --init           # set label_source='llm' di semua dok (sekali)
  python -m src.mark_label_source                  # tarik anotasi manusia LS id=4 -> tandai 'manual'
  python -m src.mark_label_source --project 4 --dry-run

JUJUR: hanya menandai 'manual' untuk comment_id yang benar-benar punya anotasi
manusia di Label Studio. Tidak pernah mengarang label manual.
"""
from __future__ import annotations

import argparse
import os
import time

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
COLLECTIONS = ["raw_comments", "processed_svm", "processed_bert"]


def _connect(tries: int = 6) -> MongoClient:
    load_dotenv(os.path.join(ROOT, ".env"))
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


def init_llm(db) -> None:
    """Set label_source='llm' pada dok yang belum punya field (kondisi awal)."""
    for name in COLLECTIONS:
        flt = {"label_source": {"$exists": False}}
        if name == "raw_comments":
            flt["label"] = {"$exists": True}  # hanya yang berlabel
        res = db[name].update_many(flt, {"$set": {"label_source": "llm"}})
        total = db[name].count_documents({"label_source": {"$exists": True}})
        print(f"  {name:<16} +llm={res.modified_count:<6} | total ber-label_source={total}")


def _fetch_manual(project: int) -> dict[str, str]:
    """Ambil {comment_id -> label_manusia} dari anotasi LS (hanya yg sudah dianotasi)."""
    from configs.config import Config
    from src.modeling.labels import parse_label_studio_export
    from src.push_labels_to_label_studio import LSClient, _read_token

    c = LSClient(Config.label_studio.URL, _read_token(None))
    r = c.request("GET", f"/api/projects/{project}/export?exportType=JSON")
    if r.status_code != 200:
        raise SystemExit(f"ERROR export project {project}: HTTP {r.status_code} {r.text[:200]}")
    snap = os.path.join(ROOT, "outputs", "labeling", f"_ls_export_p{project}.json")
    with open(snap, "wb") as f:
        f.write(r.content)
    df = parse_label_studio_export(snap)
    if df.empty or "label" not in df.columns:
        return {}
    df = df.dropna(subset=["label"])
    return dict(zip(df["comment_id"], df["label"]))


def mark_manual(db, project: int, dry_run: bool) -> None:
    manual = _fetch_manual(project)
    print(f"anotasi MANUSIA di LS project {project}: {len(manual)} komentar")
    if not manual:
        print("Belum ada anotasi manusia -> tak ada yang ditandai 'manual'. "
              "(Wajar bila labeling belum dikerjakan.)")
        return
    if dry_run:
        print("DRY-RUN: tidak menulis. Jalankan tanpa --dry-run untuk eksekusi.")
        return
    for name in COLLECTIONS:
        ops = [
            UpdateOne({"comment_id": cid},
                      {"$set": {"label_source": "manual", "manual_label": lab}})
            for cid, lab in manual.items()
        ]
        res = db[name].bulk_write(ops, ordered=False)
        print(f"  {name:<16} -> manual={res.modified_count}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tandai provenance label (llm vs manual).")
    ap.add_argument("--init", action="store_true",
                    help="Set label_source='llm' pada semua dok yang belum punya.")
    ap.add_argument("--project", type=int, default=4, help="project id LS sumber anotasi manusia")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = _connect()
    db = client[DB]
    if args.init:
        print("== INIT: label_source='llm' (kondisi awal) ==")
        init_llm(db)
    else:
        print("== MARK MANUAL dari Label Studio ==")
        mark_manual(db, args.project, args.dry_run)
    client.close()


if __name__ == "__main__":
    main()

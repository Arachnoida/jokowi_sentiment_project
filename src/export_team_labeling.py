"""Buat project Label Studio 'Pelabelan Manual Tim (N)' berisi N task BUTA.

Untuk pelabelan manual oleh TIM: ambil sampel ACAK dari ``raw_comments`` (distribusi
natural), KECUALIKAN test set v5 (1.411) agar tak tumpang tindih dengan split
evaluasi, lalu impor sebagai task BUTA (label LLM TIDAK ditampilkan -> hindari
anchoring bias). Label LLM disimpan terpisah ke CSV untuk hitung agreement
(Cohen's kappa) SETELAH pelabelan — JANGAN diimpor.

IDEMPOTEN: kalau project berjudul sama sudah ada, TIDAK membuat/import ulang.
Sampel deterministik (seed dari env RANDOM_SEED) -> bisa direproduksi.

  python -m src.export_team_labeling                 # dry-run: bangun pool + cek
  python -m src.export_team_labeling --commit        # buat project + import N task
  python -m src.export_team_labeling --n 3000 --commit
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

from configs.config import Config
from src.import_testset_project import _find_project
from src.push_labels_to_label_studio import LSClient, _read_token

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "outputs" / "labeling"
CONFIG_XML = ROOT / "configs" / "label_studio_sentiment.xml"
TESTSET_BLIND = LBL / "testset_v5_blind.json"   # sumber 1411 comment_id test set utk dikecualikan
BATCH = 500

# Field yang ikut ke task (hanya text/source_title/like_count yang DIRENDER; sisanya
# tersimpan utk join balik ke label LLM saat kappa). Lihat configs/label_studio_sentiment.xml.
DATA_FIELDS = [
    "comment_id", "video_id", "source_title", "source_url",
    "published_at", "like_count", "text",
]


def _connect(tries: int = 8) -> MongoClient:
    load_dotenv(ROOT / ".env")
    uri = os.environ["MONGO_URI"]
    last = None
    for attempt in range(1, tries + 1):
        try:
            c = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
            c.admin.command("ping")  # paksa koneksi nyata; pulihkan pool yang ter-pause
            return c
        except Exception as exc:  # noqa: BLE001 - retry transien SSL/DNS/pool
            last = exc
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Gagal koneksi Mongo: {last}")


def _exclude_ids() -> set[str]:
    """comment_id test set v5 (1411) yang harus dikecualikan dari pool tim."""
    if not TESTSET_BLIND.exists():
        sys.exit(f"File test set tak ada: {TESTSET_BLIND}\n"
                 "Jalankan dulu: python -m src.export_testset_for_labeling")
    tasks = json.loads(TESTSET_BLIND.read_text(encoding="utf-8"))
    return {str(t["data"]["comment_id"]) for t in tasks}


def _clean(v):
    return None if v is None else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="jumlah task untuk tim")
    ap.add_argument("--title", default=None, help="judul project (default: 'Pelabelan Manual Tim (N)')")
    ap.add_argument("--commit", action="store_true", help="benar-benar buat project + import")
    args = ap.parse_args()

    seed = int(os.environ.get("RANDOM_SEED", "42"))
    title = args.title or f"Pelabelan Manual Tim ({args.n})"

    # 1) Pool: semua komentar berlabel KECUALI test set v5.
    db = _connect()[os.environ.get("MONGO_DB_NAME", "youtube_sentiment")]
    proj = {"_id": 0, "label": 1, "confidence": 1, **{f: 1 for f in DATA_FIELDS}}
    pool = list(db["raw_comments"].find({"label": {"$exists": True}}, proj))
    exclude = _exclude_ids()
    pool = [d for d in pool if str(d["comment_id"]) not in exclude]
    print(f"pool (berlabel - test set): {len(pool)} | dikecualikan: {len(exclude)}")

    if args.n > len(pool):
        sys.exit(f"--n {args.n} > pool tersedia ({len(pool)})")

    # 2) Sampel ACAK deterministik (distribusi natural).
    rng = random.Random(seed)
    sample = rng.sample(pool, args.n)
    from collections import Counter
    dist = Counter(d["label"] for d in sample)
    print(f"sampel acak: {len(sample)} (seed={seed}) | dist label LLM: {dict(dist)}")

    # 3) Task BUTA (tanpa label) + simpan referensi LLM terpisah.
    tasks = [{"data": {k: _clean(d.get(k)) for k in DATA_FIELDS}} for d in sample]
    LBL.mkdir(parents=True, exist_ok=True)
    blind_path = LBL / f"team_labeling_{args.n}_blind.json"
    blind_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    ref_path = LBL / f"team_labeling_{args.n}_llm_reference.csv"
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write("comment_id,llm_label,confidence\n")
        for d in sample:
            f.write(f"{d['comment_id']},{d['label']},{d.get('confidence','')}\n")
    print(f"task buta  -> {blind_path.relative_to(ROOT)}")
    print(f"ref LLM    -> {ref_path.relative_to(ROOT)}  (JANGAN diimpor; untuk kappa)")

    # 4) Buat project + import.
    c = LSClient(Config.label_studio.URL, _read_token(None))
    existing = _find_project(c, title)
    if existing:
        pid = existing[0]["id"]
        pr = c.request("GET", f"/api/projects/{pid}/").json()
        print(f"\nProject {title!r} SUDAH ADA (id={pid}, task={pr.get('task_number')}). "
              "Tidak import ulang (idempotent).")
        return

    if not args.commit:
        print(f"\nDRY-RUN: project {title!r} belum ada. "
              f"Jalankan dengan --commit untuk buat + import {len(tasks)} task buta.")
        return

    label_config = CONFIG_XML.read_text(encoding="utf-8")
    r = c.request("POST", "/api/projects/", json={"title": title, "label_config": label_config})
    if r.status_code not in (200, 201):
        sys.exit(f"ERROR buat project: HTTP {r.status_code} {r.text[:400]}")
    pid = r.json()["id"]
    print(f"\nproject dibuat: id={pid}")

    total = 0
    for i in range(0, len(tasks), BATCH):
        batch = tasks[i:i + BATCH]
        r = c.request("POST", f"/api/projects/{pid}/import", json=batch, timeout=180)
        if r.status_code not in (200, 201):
            sys.exit(f"ERROR import batch @{i}: HTTP {r.status_code} {r.text[:400]}")
        n = r.json().get("task_count", len(batch))
        total += n
        print(f"  import batch {i // BATCH + 1}: +{n} (total {total})")

    print(f"\nSELESAI: {total} task buta diimpor ke project id={pid} {title!r}.")
    print(f"Buka {Config.label_studio.URL} -> {title!r} -> bagikan ke tim (Label All Tasks).")


if __name__ == "__main__":
    main()

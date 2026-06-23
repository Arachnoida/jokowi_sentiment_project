"""Buat project Label Studio 'Boost Minoritas' berisi task BUTA kelas Neg/Pos.

Masalah: di project utama "Pelabelan Manual Tim (3000)" (id=6), label MANUSIA
sangat timpang ke Netral (~82%) walau LLM-ref-nya seimbang — efek rubrik yang
konservatif (banyak LLM-Neg/Pos → manusia Netral). Untuk MENAIKKAN jumlah label
manusia Negatif/Positif, project ini menambah task yang **LLM prediksi Neg/Pos**,
diprioritaskan **confidence tinggi** (paling mungkin benar-benar Neg/Pos → yield
manusia Neg/Pos lebih besar), seimbang 50/50 antar kedua kelas.

Sumber: ``raw_comments`` berlabel, KECUALIKAN (a) test set v5 (1.411) dan
(b) 3.000 task project utama — supaya tidak tumpang tindih. Task BUTA (label LLM
TIDAK ditampilkan → hindari anchoring); label LLM disimpan ke CSV terpisah untuk
hitung agreement SETELAH pelabelan. JANGAN diimpor.

Project TERPISAH dari id=6 agar distribusi natural 3.000 (basis kappa
representatif) tetap bersih. IDEMPOTEN + sampel deterministik (RANDOM_SEED).

  python -m src.export_team_labeling_boost                 # dry-run: bangun pool + cek
  python -m src.export_team_labeling_boost --commit        # buat project + import
  python -m src.export_team_labeling_boost --n 1000 --commit
  python -m src.export_team_labeling_boost --order random  # acak (default: confidence desc)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from collections import Counter

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

from configs.config import Config
from src.import_testset_project import _find_project
from src.push_labels_to_label_studio import LSClient, _read_token

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "outputs" / "labeling"
CONFIG_XML = ROOT / "configs" / "label_studio_sentiment.xml"
# comment_id yang harus dikecualikan: test set v5 + 3.000 task project utama.
EXCLUDE_BLINDS = [
    LBL / "testset_v5_blind.json",
    LBL / "team_labeling_3000_blind.json",
]
MINORITY = ("Negatif", "Positif")
BATCH = 500

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
    """Gabungan comment_id yang harus dikecualikan dari pool boost."""
    ids: set[str] = set()
    for path in EXCLUDE_BLINDS:
        if not path.exists():
            sys.exit(f"File exclude tak ada: {path}\n"
                     "Pastikan test set v5 + project 3.000 sudah dibuat.")
        tasks = json.loads(path.read_text(encoding="utf-8"))
        ids |= {str(t["data"]["comment_id"]) for t in tasks}
    return ids


def _conf(d) -> float:
    """Confidence sebagai float; non-numerik -> -1 (turun ke bawah saat sort desc)."""
    v = d.get("confidence")
    return float(v) if isinstance(v, (int, float)) else -1.0


def _clean(v):
    return None if v is None else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                    help="total task boost (dibagi rata Negatif/Positif)")
    ap.add_argument("--order", choices=("confidence", "random"), default="confidence",
                    help="urutan pengambilan per kelas (default: confidence desc)")
    ap.add_argument("--title", default=None, help="judul project (default: 'Boost Minoritas (N)')")
    ap.add_argument("--commit", action="store_true", help="benar-benar buat project + import")
    args = ap.parse_args()

    seed = int(os.environ.get("RANDOM_SEED", "42"))
    title = args.title or f"Boost Minoritas ({args.n})"
    per_class = args.n // 2

    # 1) Pool: komentar berlabel Neg/Pos, KECUALI test set v5 + project 3.000.
    db = _connect()[os.environ.get("MONGO_DB_NAME", "youtube_sentiment")]
    proj = {"_id": 0, "label": 1, "confidence": 1, **{f: 1 for f in DATA_FIELDS}}
    raw = list(db["raw_comments"].find({"label": {"$in": list(MINORITY)}}, proj))
    exclude = _exclude_ids()
    pool = [d for d in raw if str(d["comment_id"]) not in exclude]
    by_class = {lab: [d for d in pool if d["label"] == lab] for lab in MINORITY}
    print(f"pool Neg/Pos (berlabel - excluded): {len(pool)} | dikecualikan: {len(exclude)}")
    for lab in MINORITY:
        print(f"  {lab}: {len(by_class[lab])} tersedia")

    short = [lab for lab in MINORITY if len(by_class[lab]) < per_class]
    if short:
        sys.exit(f"--n {args.n} butuh {per_class}/kelas, tapi kurang untuk: {short}. "
                 "Turunkan --n.")

    # 2) Ambil per kelas: confidence desc (yield Neg/Pos genuin maksimal) atau acak.
    rng = random.Random(seed)
    sample = []
    for lab in MINORITY:
        items = by_class[lab]
        if args.order == "confidence":
            # tie-break deterministik via comment_id agar reproducible
            items = sorted(items, key=lambda d: (-_conf(d), str(d["comment_id"])))
            picked = items[:per_class]
        else:
            picked = rng.sample(items, per_class)
        sample.extend(picked)
        confs = [_conf(d) for d in picked if _conf(d) >= 0]
        rng_txt = f"conf {min(confs):.3f}–{max(confs):.3f}" if confs else "tanpa confidence"
        print(f"  ambil {lab}: {len(picked)} ({args.order}; {rng_txt})")
    rng.shuffle(sample)  # acak urutan tampil supaya Neg/Pos berselang-seling
    print(f"sampel boost: {len(sample)} (seed={seed}) | dist LLM: {dict(Counter(d['label'] for d in sample))}")

    # 3) Task BUTA + referensi LLM terpisah.
    tasks = [{"data": {k: _clean(d.get(k)) for k in DATA_FIELDS}} for d in sample]
    LBL.mkdir(parents=True, exist_ok=True)
    blind_path = LBL / f"team_boost_{args.n}_blind.json"
    blind_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    ref_path = LBL / f"team_boost_{args.n}_llm_reference.csv"
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write("comment_id,llm_label,confidence\n")
        for d in sample:
            f.write(f"{d['comment_id']},{d['label']},{d.get('confidence','')}\n")
    print(f"task buta  -> {blind_path.relative_to(ROOT)}")
    print(f"ref LLM    -> {ref_path.relative_to(ROOT)}  (JANGAN diimpor; untuk kappa)")

    # 4) Buat project + import (idempotent).
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

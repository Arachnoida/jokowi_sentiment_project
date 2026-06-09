"""Buat project Label Studio 'Test Set v5 — Verifikasi Manual' + import task buta.

Membuat project baru khusus untuk pelabelan ULANG MANUAL test set v5 (1.411 task),
lalu mengimpor ``outputs/labeling/testset_v5_blind.json`` (tanpa label LLM).

IDEMPOTEN: kalau project berjudul sama sudah ada, TIDAK membuat/import ulang
(cegah duplikat). Auth via PAT (refresh-token JWT) di ``.env`` proyek ini.

  python -m src.import_testset_project            # dry-run: cek auth + status
  python -m src.import_testset_project --commit   # benar-benar buat project + import
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from configs.config import Config
from src.push_labels_to_label_studio import LSClient, _read_token

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_XML = ROOT / "configs" / "label_studio_sentiment.xml"
TASKS_JSON = ROOT / "outputs" / "labeling" / "testset_v5_blind.json"
TITLE = "Test Set v5 — Verifikasi Manual"
BATCH = 500


def _find_project(c: LSClient, title: str):
    r = c.request("GET", "/api/projects/?page_size=1000")
    r.raise_for_status()
    data = r.json()
    results = data.get("results", data) if isinstance(data, dict) else data
    return [p for p in results if p.get("title") == title]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="benar-benar buat project + import")
    args = ap.parse_args()

    tasks = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    label_config = CONFIG_XML.read_text(encoding="utf-8")
    print(f"task siap : {len(tasks)}")
    print(f"base URL  : {Config.label_studio.URL}")

    c = LSClient(Config.label_studio.URL, _read_token(None))
    print("auth OK   : PAT -> JWT access token")

    existing = _find_project(c, TITLE)
    if existing:
        pid = existing[0]["id"]
        pr = c.request("GET", f"/api/projects/{pid}/").json()
        print(f"\nProject '{TITLE}' SUDAH ADA (id={pid}, task={pr.get('task_number')}).")
        print("Tidak membuat/import ulang (idempotent). Hapus project itu dulu bila ingin re-import.")
        return

    if not args.commit:
        print(f"\nDRY-RUN: project '{TITLE}' belum ada.")
        print("Jalankan dengan --commit untuk membuat project + import 1411 task.")
        return

    r = c.request("POST", "/api/projects/", json={"title": TITLE, "label_config": label_config})
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
        print(f"  batch {i // BATCH + 1}: +{n} (total {total})")

    print(f"\nSELESAI: {total} task diimpor ke project id={pid}.")
    print(f"Buka {Config.label_studio.URL} -> '{TITLE}' -> Label All Tasks (hotkey 1/2/3).")


if __name__ == "__main__":
    main()

"""Buat project Label Studio DEMO duplikat berisi N task + anotasi label LLM.

Berbeda dari project verifikasi manual (kappa), project ini DIPERUNTUKKAN untuk
TAMPILAN/DEMO saja: tiap komentar langsung diisi anotasi label ``claude-llm``
supaya bisa diperlihatkan sebagai contoh hasil pelabelan. Karena BUKAN untuk
analisis kappa, mengisi anotasi otomatis di sini tidak mencemari metodologi.

Sumber task : ``outputs/labeling/testset_v5_blind.json`` (urutan round-robin
              stratified — N pertama tetap seimbang antar kelas).
Sumber label: ``outputs/labeling/testset_v5_llm_reference.csv`` (comment_id->llm_label).

IDEMPOTEN: kalau project berjudul sama sudah ada, TIDAK membuat/import ulang.
Anotasi yang dibuat via API tercatat atas nama pemilik PAT (user_id=2).

  python -m src.duplicate_labeling_demo                 # dry-run: cek rencana
  python -m src.duplicate_labeling_demo --commit        # buat project + import + anotasi
  python -m src.duplicate_labeling_demo --n 512 --commit
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

from configs.config import Config
from src.import_testset_project import _find_project
from src.push_labels_to_label_studio import LSClient, _read_token, _fetch_taskmap

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_XML = ROOT / "configs" / "label_studio_sentiment.xml"
BLIND_JSON = ROOT / "outputs" / "labeling" / "testset_v5_blind.json"
REF_CSV = ROOT / "outputs" / "labeling" / "testset_v5_llm_reference.csv"
BATCH = 500
VALID = {"Positif", "Negatif", "Netral"}


def _load_llm_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    with open(REF_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = (row.get("comment_id") or "").strip()
            lab = (row.get("llm_label") or "").strip()
            if cid and lab in VALID:
                labels[cid] = lab
    return labels


def _annotation_result(label: str) -> list[dict]:
    return [{
        "from_name": "sentiment",
        "to_name": "text",
        "type": "choices",
        "value": {"choices": [label]},
    }]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="jumlah komentar di project demo")
    ap.add_argument("--title", default=None, help="judul project (default: 'Labeling — Demo (N)')")
    ap.add_argument("--commit", action="store_true", help="benar-benar buat project + import + anotasi")
    args = ap.parse_args()

    title = args.title or f"Labeling — Demo ({args.n})"
    tasks_all = json.loads(BLIND_JSON.read_text(encoding="utf-8"))
    if args.n > len(tasks_all):
        sys.exit(f"--n {args.n} > total task tersedia ({len(tasks_all)})")
    tasks = tasks_all[:args.n]
    llm = _load_llm_labels()

    have = sum(1 for t in tasks if str(t["data"]["comment_id"]) in llm)
    print(f"judul    : {title!r}")
    print(f"task     : {len(tasks)} (dari {len(tasks_all)} blind)")
    print(f"label LLM: {have}/{len(tasks)} task punya label referensi")
    print(f"base URL : {Config.label_studio.URL}")

    c = LSClient(Config.label_studio.URL, _read_token(None))
    print("auth OK  : PAT -> JWT access token")

    existing = _find_project(c, title)
    if existing:
        # RESUME: project sudah ada (mis. run sebelumnya timeout di tahap anotasi).
        # Tidak membuat/import ulang; langsung lanjutkan anotasi yang belum ada.
        pid = existing[0]["id"]
        pr = c.request("GET", f"/api/projects/{pid}/").json()
        print(f"\nProject {title!r} SUDAH ADA (id={pid}, task={pr.get('task_number')}).")
        if not args.commit:
            print("DRY-RUN: tidak menulis. Jalankan dengan --commit untuk lanjut anotasi yang kurang.")
            return
        print("RESUME: lanjut buat anotasi untuk task yang belum punya anotasi.")
    else:
        if not args.commit:
            print(f"\nDRY-RUN: project {title!r} belum ada.")
            print(f"Jalankan dengan --commit untuk membuat + import {len(tasks)} task + anotasi.")
            return

        # 1) Buat project dengan label config yang sama.
        label_config = CONFIG_XML.read_text(encoding="utf-8")
        r = c.request("POST", "/api/projects/", json={"title": title, "label_config": label_config})
        if r.status_code not in (200, 201):
            sys.exit(f"ERROR buat project: HTTP {r.status_code} {r.text[:400]}")
        pid = r.json()["id"]
        print(f"\nproject dibuat: id={pid}")

        # 2) Import task per-batch.
        total = 0
        for i in range(0, len(tasks), BATCH):
            batch = tasks[i:i + BATCH]
            r = c.request("POST", f"/api/projects/{pid}/import", json=batch, timeout=180)
            if r.status_code not in (200, 201):
                sys.exit(f"ERROR import batch @{i}: HTTP {r.status_code} {r.text[:400]}")
            n = r.json().get("task_count", len(batch))
            total += n
            print(f"  import batch {i // BATCH + 1}: +{n} (total {total})")

    # 3) Petakan comment_id -> task_id, lalu buat anotasi label LLM.
    taskmap, _has_pred, has_ann = _fetch_taskmap(c, pid)
    targets = [(cid, llm[cid], tid) for cid, tid in taskmap.items()
               if cid in llm and tid not in has_ann]
    print(f"\nanotasi  : akan buat {len(targets)} anotasi label LLM")

    ok = err = 0
    for i, (cid, lab, tid) in enumerate(targets, 1):
        last = None
        for attempt in range(1, 6):
            try:
                rr = c.request("POST", f"/api/tasks/{tid}/annotations/",
                               json={"result": _annotation_result(lab), "ground_truth": False},
                               timeout=30)
                rr.raise_for_status()
                ok += 1
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(min(2 ** attempt, 15))
        if last is not None:
            err += 1
            if err <= 10:
                print(f"  gagal anotasi task {tid} ({cid}): {last}")
        time.sleep(0.15)
        if i % 200 == 0:
            print(f"  …{i}/{len(targets)} (ok={ok} err={err})")

    pr = c.request("GET", f"/api/projects/{pid}/").json()
    print(f"\nSELESAI: project id={pid} | task={pr.get('task_number')} | "
          f"anotasi ok={ok} gagal={err}")
    print(f"Buka {Config.label_studio.URL} -> {title!r}.")


if __name__ == "__main__":
    main()

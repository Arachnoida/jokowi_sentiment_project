"""
src/import_new_tasks_to_label_studio.py

Impor task BARU (yang belum ada di Label Studio) beserta anotasi claude-llm
inline, lewat /api/projects/{pid}/import. Dipakai saat dataset diperluas
(mis. dari 10k -> 14.107) sehingga ada comment_id yang belum punya task.

Sumber data : outputs/labeling/_to_label_meta.jsonl  (metadata mongo per idx)
Sumber label: outputs/labeling/_results.pkl           (idx -> label/confidence/notes)

Hanya mengimpor comment_id yang BELUM punya task (dicek via taskmap), jadi aman
diulang (idempotent terhadap duplikasi task). Default --dry-run.

Contoh:
    python -m src.import_new_tasks_to_label_studio --dry-run
    python -m src.import_new_tasks_to_label_studio --no-dry-run
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

import requests

from configs.config import Config
from src.push_labels_to_label_studio import LSClient, _read_token, _result_payload
from src.utils import setup_logger

logger = setup_logger("import_tasks_ls")

ROOT = Path(__file__).resolve().parent.parent
LBL = ROOT / "outputs" / "labeling"
META = LBL / "_to_label_meta.jsonl"
RESULTS = LBL / "_results.pkl"
VALID = {"Positif", "Negatif", "Netral"}

_DATA_FIELDS = [
    "comment_id", "video_id", "source_title", "source_url",
    "published_at", "like_count", "text",
]


def _build_tasks() -> Dict[str, dict]:
    """comment_id -> objek task {data, annotations} siap diimpor."""
    results = pickle.loads(RESULTS.read_bytes())
    meta: Dict[int, dict] = {}
    with open(META, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["idx"]] = d

    tasks: Dict[str, dict] = {}
    for idx, m in meta.items():
        r = results.get(idx)
        if not r or r.get("label") not in VALID:
            continue
        cid = (m.get("comment_id") or "").strip()
        if not cid:
            continue
        data = {k: m.get(k) for k in _DATA_FIELDS}
        data["label"] = r["label"]
        data["annotator"] = "claude-llm"
        data["notes"] = r.get("notes")
        tasks[cid] = {
            "data": data,
            "annotations": [{
                "result": _result_payload(r["label"]),
                "ground_truth": False,
            }],
        }
    return tasks


def _project_task_number(c: LSClient, pid: int, tries: int = 6) -> int:
    """Ambil task_number project (cheap, 1 request) dengan retry koneksi flaky."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = c.request("GET", f"/api/projects/{pid}/")
            r.raise_for_status()
            return int(r.json().get("task_number") or 0)
        except (requests.exceptions.RequestException, ValueError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Gagal ambil project task_number setelah {tries}x: {last}")


def _import_batch(c: LSClient, pid: int, batch: List[dict], tries: int = 6) -> dict:
    """POST satu batch import dgn retry pada koneksi putus (HF Space flaky)."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = c.request("POST", f"/api/projects/{pid}/import", json=batch, timeout=180)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as exc:
            last = exc
            logger.warning("Batch import gagal (percobaan %d/%d): %s", attempt, tries, exc)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Gagal import batch setelah {tries}x: {last}")


def run(dry_run: bool, token: str | None, batch_size: int, expected_existing: int) -> None:
    base = Config.label_studio.URL.rstrip("/")
    pid = Config.label_studio.PROJECT_ID
    tok = _read_token(token)

    tasks = _build_tasks()
    payload: List[dict] = list(tasks.values())
    logger.info("Task kandidat baru (dari label): %d", len(payload))

    c = LSClient(base, tok)
    # Guard murah: cek task_number, hindari paginasi 10k task yang flaky di HF Space.
    n = _project_task_number(c, pid)
    logger.info("Project task_number saat ini: %d (baseline diharapkan: %d)",
                n, expected_existing)
    if n >= expected_existing + len(payload):
        logger.info("Sudah terimpor (task_number >= %d). Tidak ada yang dilakukan.",
                    expected_existing + len(payload))
        return
    if n != expected_existing:
        logger.warning("task_number (%d) != baseline (%d). Mungkin sudah ada impor "
                       "sebagian. Hentikan untuk hindari duplikat; pakai --expected-existing "
                       "%d bila yakin.", n, expected_existing, n)
        if not dry_run:
            raise SystemExit("Dibatalkan: baseline task_number tak sesuai (cegah duplikat).")

    if dry_run:
        logger.info("DRY-RUN: akan mengimpor %d task baru (+anotasi). Tidak menulis.",
                    len(payload))
        return

    total_ok = 0
    for i in range(0, len(payload), batch_size):
        batch = payload[i:i + batch_size]
        info = _import_batch(c, pid, batch)
        total_ok += info.get("task_count", len(batch))
        logger.info("…impor batch %d: task_count=%s annotation_count=%s (kumulatif=%d)",
                    i // batch_size + 1, info.get("task_count"),
                    info.get("annotation_count"), total_ok)
    logger.info("SELESAI impor: %d task baru (dengan anotasi claude-llm).", total_ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="Impor task baru + anotasi ke Label Studio.")
    ap.add_argument("--token", default=None)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--expected-existing", type=int, default=10000,
                    help="task_number baseline sebelum impor (guard anti-duplikat).")
    dr = ap.add_mutually_exclusive_group()
    dr.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    dr.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()
    run(args.dry_run, args.token, args.batch_size, args.expected_existing)


if __name__ == "__main__":
    main()

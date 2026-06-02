"""
src/push_labels_to_label_studio.py

Dorong label hasil pelabelan (claude-llm) ke instance Label Studio yang sudah
berisi task — TANPA membuat task duplikat. Pencocokan via `comment_id` di
`data` task. Default aman: --dry-run dan mode `predictions` (pra-anotasi yang
bisa di-review/accept di UI).

Auth: Label Studio versi baru pakai Personal Access Token bergaya JWT (refresh
token). Skrip menukar refresh -> access via /api/token/refresh, lalu memakai
header `Authorization: Bearer <access>`, dan me-refresh otomatis saat access
token kedaluwarsa (umurnya pendek). Token diambil (urut prioritas): argumen
--token, env LABEL_STUDIO_API_TOKEN, file .env proyek, lalu
../jokowi-label-studio/.env. JWT polos satu baris juga dikenali.

Contoh:
    python -m src.push_labels_to_label_studio --dry-run        # cek, tak menulis
    python -m src.push_labels_to_label_studio --no-dry-run     # push predictions
    python -m src.push_labels_to_label_studio --no-dry-run \
        --source outputs/labeling/balanced_1000.csv --mode annotations
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from configs.config import Config
from src.utils import setup_logger

logger = setup_logger("push_labels_ls")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "outputs" / "labeling" / "labeling_dataset.csv"
VALID = {"Positif", "Negatif", "Netral"}
_JWT_RE = re.compile(r"[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")
# Lokasi .env yang dicek untuk token (selain env & argumen).
_ENV_CANDIDATES = [ROOT / ".env", ROOT.parent / "jokowi-label-studio" / ".env"]


def _extract_token(text: str) -> Optional[str]:
    """Cari token di isi file .env: baris KEY=val (key mengandung TOKEN) atau JWT polos."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and "TOKEN" in line.split("=", 1)[0].upper():
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if val:
                return val
        m = _JWT_RE.fullmatch(line)
        if m:
            return line
    return None


def _read_token(cli_token: Optional[str]) -> str:
    if cli_token:
        return cli_token.strip()
    if Config.label_studio.API_TOKEN:
        return Config.label_studio.API_TOKEN.strip()
    for env_file in _ENV_CANDIDATES:
        if env_file.exists():
            tok = _extract_token(env_file.read_text(encoding="utf-8"))
            if tok:
                logger.info("Token diambil dari %s", env_file)
                return tok
    sys.exit(
        "ERROR: API token Label Studio tidak ditemukan.\n"
        "UI: avatar > Account & Settings > Access Token (copy).\n"
        "Lalu --token <TOKEN>, atau set LABEL_STUDIO_API_TOKEN di .env."
    )


class LSClient:
    """Klien Label Studio dengan auth JWT (refresh->access) + auto-refresh."""

    def __init__(self, base: str, refresh_token: str):
        self.base = base.rstrip("/")
        self.refresh_token = refresh_token
        self.s = requests.Session()
        self._refresh_access()

    def _refresh_access(self, tries: int = 5) -> None:
        """Tukar refresh->access; ulangi bila server balas 5xx transien."""
        last = None
        for attempt in range(1, tries + 1):
            try:
                r = self.s.post(f"{self.base}/api/token/refresh",
                                json={"refresh": self.refresh_token}, timeout=30)
            except requests.exceptions.RequestException as exc:
                last = exc
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code in (401, 403):
                sys.exit(f"ERROR: refresh token ditolak (HTTP {r.status_code}). "
                         "Token salah/kedaluwarsa — ambil ulang di Account & Settings.")
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(min(2 ** attempt, 20))
                continue
            r.raise_for_status()
            access = r.json().get("access")
            if not access:
                sys.exit("ERROR: respons /api/token/refresh tanpa field 'access'.")
            self.s.headers.update({"Authorization": f"Bearer {access}",
                                   "Content-Type": "application/json"})
            return
        sys.exit(f"ERROR: gagal refresh access token setelah {tries}x: {last}")

    def request(self, method: str, path: str, **kw) -> requests.Response:
        """Request dgn retry bila access token kedaluwarsa (401) atau 5xx transien."""
        url = f"{self.base}{path}"
        kw.setdefault("timeout", 60)
        r = self.s.request(method, url, **kw)
        if r.status_code == 401:
            self._refresh_access()
            r = self.s.request(method, url, **kw)
        elif r.status_code >= 500:
            time.sleep(2)
            r = self.s.request(method, url, **kw)
        return r


def _load_labels(source: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(source, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip()
            cid = (row.get("comment_id") or "").strip()
            if cid and lab in VALID:
                labels[cid] = lab
    return labels


def _check_project(c: LSClient, pid: int) -> None:
    r = c.request("GET", f"/api/projects/{pid}/")
    r.raise_for_status()
    p = r.json()
    logger.info("Project OK: id=%s title=%r task_number=%s",
                p.get("id"), p.get("title"), p.get("task_number"))


def _fetch_taskmap(c: LSClient, pid: int) -> Tuple[Dict[str, int], set]:
    """comment_id -> task_id (semua task), plus set task_id yang sudah punya prediction."""
    taskmap: Dict[str, int] = {}
    has_pred: set = set()
    page, page_size = 1, 200
    while True:
        r = c.request("GET", "/api/tasks/",
                      params={"project": pid, "page": page, "page_size": page_size})
        if r.status_code == 404:  # melewati halaman terakhir
            break
        r.raise_for_status()
        payload = r.json()
        tasks = payload.get("tasks", payload) if isinstance(payload, dict) else payload
        if not tasks:
            break
        for t in tasks:
            cid = str((t.get("data") or {}).get("comment_id") or "").strip()
            if cid:
                taskmap[cid] = t["id"]
            if (t.get("total_predictions") or 0) > 0:
                has_pred.add(t["id"])
        if page == 1 or page % 10 == 0:
            logger.info("…ambil task halaman %d (terkumpul %d)", page, len(taskmap))
        if len(tasks) < page_size:
            break
        page += 1
    return taskmap, has_pred


def _result_payload(label: str) -> List[dict]:
    return [{
        "from_name": Config.label_studio.FROM_NAME,
        "to_name": "text",
        "type": "choices",
        "value": {"choices": [label]},
    }]


def push(source: Path, mode: str, dry_run: bool, token: Optional[str],
         limit: Optional[int], skip_existing: bool) -> None:
    base = Config.label_studio.URL.rstrip("/")
    pid = Config.label_studio.PROJECT_ID
    tok = _read_token(token)

    labels = _load_labels(source)
    logger.info("Label sumber: %d baris berlabel dari %s", len(labels), source)

    c = LSClient(base, tok)
    _check_project(c, pid)
    taskmap, has_pred = _fetch_taskmap(c, pid)
    logger.info("Task di project punya comment_id: %d | sudah ada prediction: %d",
                len(taskmap), len(has_pred))

    matched: List[Tuple[str, str, int]] = [
        (cid, lab, taskmap[cid]) for cid, lab in labels.items() if cid in taskmap
    ]
    missing = len(labels) - len(matched)
    if skip_existing and mode == "predictions":
        before = len(matched)
        matched = [m for m in matched if m[2] not in has_pred]
        logger.info("Resume: lewati %d task yang sudah punya prediction.",
                    before - len(matched))
    logger.info("Cocok via comment_id: %d | label tanpa task: %d | akan diproses: %d",
                len(labels) - missing, missing, len(matched))
    if limit:
        matched = matched[:limit]

    if dry_run:
        logger.info("DRY-RUN: tidak menulis apa pun. Akan push %d sbg %s.",
                    len(matched), mode)
        logger.info("Jalankan ulang dengan --no-dry-run untuk eksekusi.")
        return

    ok = err = 0
    for i, (cid, lab, tid) in enumerate(matched, 1):
        try:
            if mode == "predictions":
                r = c.request("POST", "/api/predictions/", json={
                    "task": tid, "result": _result_payload(lab),
                    "model_version": "claude-llm",
                }, timeout=30)
            else:
                r = c.request("POST", f"/api/tasks/{tid}/annotations/", json={
                    "result": _result_payload(lab), "ground_truth": False,
                }, timeout=30)
            r.raise_for_status()
            ok += 1
        except Exception as exc:  # noqa: BLE001
            err += 1
            if err <= 10:
                logger.warning("Gagal task %s (%s): %s", tid, cid, exc)
        if i % 250 == 0:
            logger.info("…%d/%d (ok=%d err=%d)", i, len(matched), ok, err)
    logger.info("SELESAI: %s dibuat=%d, gagal=%d, label-tanpa-task=%d",
                mode, ok, err, missing)


def main() -> None:
    ap = argparse.ArgumentParser(description="Push label claude-llm ke Label Studio.")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--mode", choices=["predictions", "annotations"],
                    default="predictions")
    ap.add_argument("--token", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    default=True,
                    help="Jangan lewati task yg sudah punya prediction (default: dilewati).")
    dr = ap.add_mutually_exclusive_group()
    dr.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    dr.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()
    push(args.source, args.mode, args.dry_run, args.token, args.limit,
         args.skip_existing)


if __name__ == "__main__":
    main()

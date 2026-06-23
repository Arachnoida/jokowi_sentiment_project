"""Setel ulang project Label Studio 'Test Set v5 — Verifikasi Manual' ke subset N.

Project verifikasi manual semula berisi 1.411 task (seluruh test set v5). Skrip ini
menyusutkannya ke N task pertama (default 512) menurut urutan round-robin stratified
yang sama dengan ekspor (``testset_v5_blind.json``) — sehingga subset tetap seimbang
antar kelas — lalu mengganti judul project agar mencerminkan jumlah baru.

AMAN secara metodologi & data:
  * Tidak membuat/menulis anotasi apa pun. Hanya menghapus task surplus + rename.
  * GUARD: bila ada task yang akan dihapus ternyata sudah punya anotasi manusia,
    skrip BATAL (menolak membuang kerja anotasi yang sudah ada).
  * Urutan diambil dari ``testset_v5_blind.json`` (file yang dulu diimpor), jadi
    N task yang DIPERTAHANKAN identik dengan N pertama yang sudah ada di LS.

Auth via PAT (refresh-token JWT) di ``.env`` proyek.

  python -m src.resize_testset_project                 # dry-run: tampilkan rencana
  python -m src.resize_testset_project --commit        # rename + hapus surplus
  python -m src.resize_testset_project --n 512 --commit
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from configs.config import Config
from src.import_testset_project import _find_project
from src.push_labels_to_label_studio import LSClient, _read_token, _fetch_taskmap

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLIND_JSON = ROOT / "outputs" / "labeling" / "testset_v5_blind.json"
OLD_TITLE = "Test Set v5 — Verifikasi Manual"


def _keep_ids(n: int) -> list[str]:
    """N comment_id pertama menurut urutan round-robin di file blind."""
    tasks = json.loads(BLIND_JSON.read_text(encoding="utf-8"))
    if n > len(tasks):
        sys.exit(f"--n {n} > total task ({len(tasks)}) di {BLIND_JSON.name}")
    return [str(t["data"]["comment_id"]) for t in tasks[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="jumlah task yang dipertahankan")
    ap.add_argument("--commit", action="store_true", help="benar-benar rename + hapus")
    args = ap.parse_args()

    keep = set(_keep_ids(args.n))
    new_title = f"{OLD_TITLE} ({args.n})"
    print(f"target  : pertahankan {len(keep)} task, judul -> {new_title!r}")

    c = LSClient(Config.label_studio.URL, _read_token(None))
    found = _find_project(c, OLD_TITLE) or _find_project(c, new_title)
    if not found:
        sys.exit(f"Project '{OLD_TITLE}' tidak ditemukan di {Config.label_studio.URL}")
    pid = found[0]["id"]
    print(f"project : id={pid} '{found[0].get('title')}' (task_number={found[0].get('task_number')})")

    taskmap, _has_pred, has_ann = _fetch_taskmap(c, pid)
    print(f"diambil : {len(taskmap)} task punya comment_id | sudah ada anotasi: {len(has_ann)}")

    keep_present = [cid for cid in keep if cid in taskmap]
    missing = len(keep) - len(keep_present)
    if missing:
        print(f"PERINGATAN: {missing} comment_id target tak ada di project (urutan beda?).")

    to_delete = [(cid, tid) for cid, tid in taskmap.items() if cid not in keep]
    del_with_ann = [(cid, tid) for cid, tid in to_delete if tid in has_ann]
    if del_with_ann:
        sys.exit(
            f"BATAL: {len(del_with_ann)} task yang akan dihapus sudah punya anotasi.\n"
            "Menolak membuang kerja anotasi. Periksa manual sebelum lanjut."
        )

    print(f"rencana : hapus {len(to_delete)} task surplus, pertahankan {len(taskmap) - len(to_delete)}")

    if not args.commit:
        print("\nDRY-RUN: tidak ada perubahan. Jalankan ulang dengan --commit.")
        return

    # 1) Rename judul lebih dulu (operasi murah, mudah diverifikasi).
    r = c.request("PATCH", f"/api/projects/{pid}/", json={"title": new_title})
    if r.status_code not in (200, 201):
        sys.exit(f"ERROR rename: HTTP {r.status_code} {r.text[:300]}")
    print(f"judul   : -> {new_title!r}")

    # 2) Hapus task surplus satu per satu (HF free-tier kerap memutus koneksi).
    ok = err = 0
    for i, (cid, tid) in enumerate(to_delete, 1):
        last = None
        for attempt in range(1, 6):
            try:
                rr = c.request("DELETE", f"/api/tasks/{tid}/", timeout=30)
                if rr.status_code in (200, 204):
                    ok += 1
                    last = None
                    break
                last = f"HTTP {rr.status_code}"
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(min(2 ** attempt, 15))
        if last is not None:
            err += 1
            if err <= 10:
                print(f"  gagal hapus task {tid} ({cid}): {last}")
        time.sleep(0.1)
        if i % 200 == 0:
            print(f"  …{i}/{len(to_delete)} (ok={ok} err={err})")

    pr = c.request("GET", f"/api/projects/{pid}/").json()
    print(f"\nSELESAI: hapus ok={ok} gagal={err} | task_number sekarang={pr.get('task_number')}")
    print(f"Buka {Config.label_studio.URL} -> '{new_title}' -> Label All Tasks.")


if __name__ == "__main__":
    main()

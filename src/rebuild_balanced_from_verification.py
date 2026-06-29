"""Terapkan hasil verifikasi manual (LS project id=9) → rebuild dataset balanced.

Alur:
  1. Tarik anotasi dari project verifikasi Label Studio (default id=9).
  2. Klasifikasikan tiap anotasi:
       - "Tidak terbaca"        → comment_id DIBUANG dari pool (gibberish).
       - Pos/Neg/Net ≠ label LLM → KOREKSI label (manusia = gold).
       - Pos/Neg/Net = label LLM → konfirmasi (tandai gold, label tetap).
  3. Terapkan ke pool penuh (`labeling_dataset.csv`): buang excluded, update koreksi.
  4. Rebuild balanced 1000/kelas: prioritas baris **terverifikasi-manusia**, lalu
     LLM confidence tertinggi (top-up otomatis menggantikan yg dibuang) → tetap 1000/kelas.
  5. Tulis `balanced_3000.csv` baru (backup yg lama) + ringkasan perubahan.

Mode tambahan ``--patch-config``: update label_config project (tambah "Tidak terbaca")
pada project LS yang sudah live (sekali jalan, tak menyentuh dataset).

  python -m src.rebuild_balanced_from_verification --patch-config       # update project id=9
  python -m src.rebuild_balanced_from_verification                      # dry-run: ringkasan
  python -m src.rebuild_balanced_from_verification --commit             # tulis balanced baru
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import pandas as pd

from configs.config import Config
from src.push_labels_to_label_studio import LSClient, _read_token

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "outputs" / "labeling"
POOL_CSV = LBL / "labeling_dataset.csv"
OUT_CSV = LBL / "balanced_3000.csv"
VERIFY_XML = ROOT / "configs" / "label_studio_verify.xml"
LABELS = ["Negatif", "Netral", "Positif"]
VALID = set(LABELS)
PER_CLASS = 1000
UNREADABLE = "Tidak terbaca"


def _fetch_export(c: LSClient, pid: int) -> list:
    r = c.request("GET", f"/api/projects/{pid}/export?exportType=JSON")
    if r.status_code != 200:
        raise SystemExit(f"ERROR export project {pid}: HTTP {r.status_code} {r.text[:300]}")
    return json.loads(r.content)


def _parse_annotations(tasks: list) -> dict[str, str]:
    """comment_id -> pilihan manusia (anotasi terbaru). Hanya task yg sudah dianotasi."""
    out: dict[str, str] = {}
    for t in tasks:
        cid = str((t.get("data") or {}).get("comment_id") or "").strip()
        anns = t.get("annotations") or []
        if not cid or not anns:
            continue
        choice = None
        for res in (anns[-1].get("result") or []):
            val = (res.get("value") or {}).get("choices") or []
            if val:
                choice = val[0]
        if choice:
            out[cid] = choice
    return out


def _patch_config(pid: int, token: str | None) -> None:
    c = LSClient(Config.label_studio.URL, _read_token(token))
    cfg = VERIFY_XML.read_text(encoding="utf-8")
    r = c.request("PATCH", f"/api/projects/{pid}/", json={"label_config": cfg})
    if r.status_code not in (200, 201):
        raise SystemExit(f"ERROR patch config: HTTP {r.status_code} {r.text[:300]}")
    print(f"OK: label_config project {pid} di-update (opsi '🗑️ {UNREADABLE}' ditambahkan).")


def rebuild(pid: int, token: str | None, commit: bool) -> None:
    c = LSClient(Config.label_studio.URL, _read_token(token))
    tasks = _fetch_export(c, pid)
    ann = _parse_annotations(tasks)
    print(f"Anotasi manusia ditarik: {len(ann)} dari project {pid}.")
    if not ann:
        print("Belum ada anotasi. Jalankan lagi setelah verifikasi.")
        return

    pool = pd.read_csv(POOL_CSV)
    pool["comment_id"] = pool["comment_id"].astype(str)
    cur = dict(zip(pool["comment_id"], pool["label"]))

    excluded, corrections, confirmed = set(), {}, set()
    for cid, choice in ann.items():
        if choice == UNREADABLE:
            excluded.add(cid)
        elif choice in VALID:
            if cur.get(cid) == choice:
                confirmed.add(cid)
            else:
                corrections[cid] = choice
    print(f"  buang (tidak terbaca): {len(excluded)}")
    print(f"  koreksi label        : {len(corrections)}")
    print(f"  konfirmasi (gold)    : {len(confirmed)}")

    # Terapkan ke pool: buang excluded, update koreksi, tandai gold (terverifikasi).
    pool = pool[~pool["comment_id"].isin(excluded)].copy()
    pool["label"] = pool.apply(
        lambda r: corrections.get(r["comment_id"], r["label"]), axis=1)
    gold = set(corrections) | confirmed
    pool["is_gold"] = pool["comment_id"].isin(gold)

    # Rebuild balanced: prioritas gold (manusia) lalu confidence LLM tertinggi.
    parts = []
    for lab in LABELS:
        sub = pool[pool["label"] == lab].sort_values(
            ["is_gold", "confidence", "comment_id"], ascending=[False, False, True])
        n = min(PER_CLASS, len(sub))
        if n < PER_CLASS:
            print(f"  ! {lab}: hanya {n} tersedia (< {PER_CLASS}).")
        parts.append(sub.head(n))
    new = pd.concat(parts, ignore_index=True)

    old_ids = set(pd.read_csv(OUT_CSV, usecols=["comment_id"])["comment_id"].astype(str)) \
        if OUT_CSV.exists() else set()
    new_ids = set(new["comment_id"])
    print(f"\nBalanced baru: {len(new)} baris | gold di-set: {int(new['is_gold'].sum())}")
    print(f"  vs balanced lama: keluar {len(old_ids - new_ids)}, masuk {len(new_ids - old_ids)}")
    print(new["label"].value_counts().to_string())

    if not commit:
        print("\nDRY-RUN: tidak menulis. Jalankan --commit untuk tulis balanced_3000.csv baru.")
        return

    if OUT_CSV.exists():
        bak = OUT_CSV.with_suffix(".csv.bak")
        shutil.copy(OUT_CSV, bak)
        print(f"Backup lama → {bak.name}")
    cols = [c for c in ["comment_id", "label", "confidence", "is_gold", "text"] if c in new.columns]
    new[cols].to_csv(OUT_CSV, index=False)
    print(f"Tersimpan: {OUT_CSV}")
    print("Lanjut re-train: train_svm_full14k / train_svm_spark / IndoBERT --subset balanced_3000.csv --tag balanced3k")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild balanced dari verifikasi manual LS.")
    ap.add_argument("--project", type=int, default=9)
    ap.add_argument("--token", default=None)
    ap.add_argument("--patch-config", action="store_true",
                    help="Update label_config project (tambah 'Tidak terbaca') lalu keluar.")
    ap.add_argument("--commit", action="store_true", help="Tulis balanced_3000.csv baru.")
    args = ap.parse_args()
    if args.patch_config:
        _patch_config(args.project, args.token)
        return
    rebuild(args.project, args.token, args.commit)


if __name__ == "__main__":
    main()

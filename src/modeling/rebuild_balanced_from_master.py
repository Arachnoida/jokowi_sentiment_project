"""Rebuild balanced subset dari master labeling_dataset.csv (pasca re-label Opus).

Dipakai setelah koreksi re-label di-push ke master (push_relabel_to_mongo). Master
sudah memuat label final (pass-1 koreksi + pass-2b domain-aware). Skrip ini:
  1. readable-filter + dedup teks identik.
  2. Tandai `verified` = comment_id yang pernah di-relabel Opus (union file durable
     `relabel_pass2_opus_*.csv` + `relabel_pass2b_domainaware_*.csv`).
  3. Sample per_class: prioritas verified, lalu confidence tertinggi.
  4. Tulis balanced_<3*per_class>.csv (backup yang lama).

  python -m src.modeling.rebuild_balanced_from_master --per-class 1000 --commit
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

LABELS = ["Negatif", "Netral", "Positif"]
_WORD3 = re.compile(r"[A-Za-z]{3,}")


def is_readable(t) -> bool:
    t = str(t).strip()
    if len(t) < 3 or not _WORD3.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    nonspace = sum(not c.isspace() for c in t)
    return bool(nonspace) and letters / nonspace >= 0.5


def _norm(t) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    root = _root()
    lbl = root / "outputs" / "labeling"

    df = pd.read_csv(lbl / "labeling_dataset.csv")
    df["comment_id"] = df["comment_id"].astype(str)
    df = df[df["text"].apply(is_readable)].copy()

    verified: set[str] = set()
    for pat in ("relabel_pass2_opus_*.csv", "relabel_pass2b_domainaware_*.csv"):
        for f in lbl.glob(pat):
            verified |= set(pd.read_csv(f)["comment_id"].astype(str))
    df["verified"] = df["comment_id"].isin(verified)

    df["_n"] = df["text"].map(_norm)
    b = len(df)
    df = (df.sort_values(["verified", "confidence"], ascending=[False, False])
            .drop_duplicates("_n", keep="first").drop(columns="_n"))
    print(f"readable+dedup: {b} -> {len(df)} | verified pool: {int(df['verified'].sum())}")

    parts = []
    for lab in LABELS:
        sub = df[df["label"] == lab].sort_values(
            ["verified", "confidence", "comment_id"], ascending=[False, False, True]
        )
        n = min(args.per_class, len(sub))
        take = sub.head(n)
        print(f"  {lab:<8} pool={len(sub):<6} ambil={n:<5} verified={int(take['verified'].sum())} "
              f"conf {take['confidence'].min():.2f}-{take['confidence'].max():.2f}")
        if n < args.per_class:
            print(f"    ! {lab}: hanya {n} < {args.per_class}")
        parts.append(take)
    bal = pd.concat(parts, ignore_index=True)
    print(f"\nBalanced: {len(bal)}\n{bal['label'].value_counts().to_string()}")

    out = lbl / f"balanced_{3*args.per_class}.csv"
    if args.commit:
        if out.exists():
            shutil.copy(out, out.with_suffix(".prerebuild.bak"))
        cols = ["comment_id", "label", "confidence", "text", "verified"]
        bal[[c for c in cols if c in bal.columns]].to_csv(out, index=False)
        print(f"Disimpan: {out}")
    else:
        print("[dry-run] tambah --commit untuk menyimpan.")


if __name__ == "__main__":
    main()

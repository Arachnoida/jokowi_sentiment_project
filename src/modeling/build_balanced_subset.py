"""Bangun subset BALANCED dari labeling_dataset.csv (label LLM).

Strategi (keputusan user 2026-06-29): ambil N/kelas dengan **confidence LLM
tertinggi** (tie-break comment_id) → label paling bersih, dataset seimbang.
Default N=1000/kelas → 3000 total (Negatif plafon 1213, jadi 1000 aman).

Output: CSV allowlist (comment_id, label, confidence, text) yang dipakai trainer
via --subset. TIDAK menyentuh Mongo/parquet — fitur untuk baris ini sudah ada di
processed_svm/processed_bert/features_spark (semua 14107 baris sudah ter-preprocess).

Jalankan: python -m src.modeling.build_balanced_subset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABELS = ["Negatif", "Netral", "Positif"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build(src_csv: Path, per_class: int) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    parts = []
    for lab in LABELS:
        sub = df[df["label"] == lab].sort_values(
            ["confidence", "comment_id"], ascending=[False, True]
        )
        n = min(per_class, len(sub))
        if n < per_class:
            print(f"  ! {lab}: hanya {n} tersedia (< {per_class}) — diambil semua.")
        parts.append(sub.head(n))
    out = pd.concat(parts, ignore_index=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangun subset balanced (top-confidence/kelas).")
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument(
        "--src", default=None,
        help="CSV sumber. Default outputs/labeling/labeling_dataset.csv.",
    )
    ap.add_argument(
        "--out", default=None,
        help="CSV keluaran. Default outputs/labeling/balanced_<3*per_class>.csv.",
    )
    args = ap.parse_args()

    root = _repo_root()
    src = Path(args.src) if args.src else root / "outputs" / "labeling" / "labeling_dataset.csv"
    total = 3 * args.per_class
    out = Path(args.out) if args.out else root / "outputs" / "labeling" / f"balanced_{total}.csv"

    df = build(src, args.per_class)
    cols = [c for c in ["comment_id", "label", "confidence", "text"] if c in df.columns]
    df[cols].to_csv(out, index=False)

    print(f"Subset balanced: {len(df)} baris ({args.per_class}/kelas target)")
    print(df["label"].value_counts().to_string())
    for lab in LABELS:
        s = df[df["label"] == lab]["confidence"]
        print(f"  {lab:<8} confidence {s.min():.2f}–{s.max():.2f} (median {s.median():.2f})")
    print(f"Disimpan: {out}")


if __name__ == "__main__":
    main()

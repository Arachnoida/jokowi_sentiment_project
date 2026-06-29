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
import re
from pathlib import Path

import pandas as pd

LABELS = ["Negatif", "Netral", "Positif"]
_WORD3 = re.compile(r"[A-Za-z]{3,}")


def is_readable(text) -> bool:
    """Kriteria 'readable' (tingkat SEDANG, keputusan user 2026-06-30):
    buang komentar emoji/simbol murni & yang didominasi emoji.

    Lolos jika: panjang ≥ 3, ada ≥ 1 kata beralfabet ≥ 3 huruf, dan mayoritas
    karakter non-spasi adalah huruf (rasio ≥ 0.5). Kata tunggal bermakna
    ('Mantap', 'HOAX') TETAP disimpan; '😂😂😂', '❤', 'Suwardi ❤😂🎉' dibuang.
    """
    t = str(text).strip()
    if len(t) < 3 or not _WORD3.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    nonspace = sum(not c.isspace() for c in t)
    return bool(nonspace) and letters / nonspace >= 0.5


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build(src_csv: Path, per_class: int, readable: bool = True) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    if readable:
        before = len(df)
        df = df[df["text"].apply(is_readable)].copy()
        print(f"  filter readable: {before} -> {len(df)} (buang {before - len(df)})")
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
    ap.add_argument(
        "--no-readable", dest="readable", action="store_false", default=True,
        help="Jangan saring komentar tak-readable (reproduksi versi lama tanpa filter).",
    )
    args = ap.parse_args()

    root = _repo_root()
    src = Path(args.src) if args.src else root / "outputs" / "labeling" / "labeling_dataset.csv"
    total = 3 * args.per_class
    out = Path(args.out) if args.out else root / "outputs" / "labeling" / f"balanced_{total}.csv"

    df = build(src, args.per_class, readable=args.readable)
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

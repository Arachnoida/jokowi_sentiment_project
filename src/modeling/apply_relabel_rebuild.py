"""Terapkan hasil re-label LLM pass-2 (Opus) → rebuild balanced subset.

Konteks (2026-06-30): kelas Negatif = leher botol (hanya 1.213 readable), sehingga
balanced 1.000/kelas memaksa Negatif turun ke confidence 0.75 (kotor) → error #1
adalah Negatif→Positif (stance flip). Solusi yang dipilih user: pertahankan
1.000/kelas tetapi PERBAIKI label suspect (disagree ∪ low-conf) dengan re-label
Opus (gold, menimpa Sonnet pass-1).

Alur:
  1. Baca kandidat suspect (`relabel_candidates.csv`, index = idx batch) + semua
     output JSONL agen Opus di `--out-dir`.
  2. Bangun mapping comment_id → label_pass2 (+ confidence Opus).
  3. Terapkan koreksi ke pool penuh (`labeling_dataset.csv`, sudah readable-filter):
     label di-override, baris ditandai `verified=True` (prioritas saat sampling).
  4. Dedup teks identik (normalisasi), simpan baris verified/confidence tertinggi.
  5. Rebuild balanced per_class/kelas: urut verified-first lalu confidence → top-up
     otomatis bila satu kelas menyusut akibat koreksi.
  6. Tulis `balanced_3000.csv` baru (backup yang lama) + ringkasan perubahan.

Jalankan (setelah agen Opus selesai):
  python -m src.modeling.apply_relabel_rebuild \
      --candidates <scratch>/relabel_candidates.csv \
      --out-dir <scratch>/relabel_out \
      --dry-run            # ringkasan saja
  python -m src.modeling.apply_relabel_rebuild ... --commit
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

LABELS = ["Negatif", "Netral", "Positif"]
_WORD3 = re.compile(r"[A-Za-z]{3,}")


def is_readable(text) -> bool:
    t = str(text).strip()
    if len(t) < 3 or not _WORD3.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    nonspace = sum(not c.isspace() for c in t)
    return bool(nonspace) and letters / nonspace >= 0.5


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_pass2(candidates: Path, out_dir: Path) -> pd.DataFrame:
    """Gabungkan output JSONL agen → DataFrame[comment_id, label_pass2, conf_pass2]."""
    cand = pd.read_csv(candidates)
    cand["comment_id"] = cand["comment_id"].astype(str)
    # kolom label pass-1 bisa bernama "llm_label_pass1" (pass-1) atau "label_pass1" (pass-2)
    if "llm_label_pass1" not in cand.columns and "label_pass1" in cand.columns:
        cand = cand.rename(columns={"label_pass1": "llm_label_pass1"})
    recs: dict[int, dict] = {}
    for jf in sorted(out_dir.glob("batch_*.jsonl")):
        for line in jf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            recs[int(o["idx"])] = o
    rows = []
    missing = []
    for idx, r in cand.iterrows():
        o = recs.get(int(idx))
        if o is None:
            missing.append(int(idx))
            continue
        lab = str(o["label"]).strip().capitalize()
        if lab not in LABELS:
            raise ValueError(f"label tak dikenal idx={idx}: {o['label']!r}")
        rows.append({
            "comment_id": r["comment_id"],
            "label_pass1": r["llm_label_pass1"],
            "label_pass2": lab,
            "conf_pass2": float(o.get("confidence", 0.9)),
        })
    if missing:
        print(f"  ! PERINGATAN: {len(missing)} idx tak ada di output JSONL: {missing[:10]}...")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--src", default=None, help="pool sumber (default labeling_dataset.csv)")
    ap.add_argument("--out", default=None, help="balanced keluaran (default balanced_3000.csv)")
    ap.add_argument("--no-dedup", dest="dedup", default=True, action="store_false",
                    help="Jangan dedup teks identik.")
    ap.add_argument("--commit", action="store_true", help="tulis file (default dry-run)")
    args = ap.parse_args()

    root = _repo_root()
    src = Path(args.src) if args.src else root / "outputs" / "labeling" / "labeling_dataset.csv"
    total = 3 * args.per_class
    out = Path(args.out) if args.out else root / "outputs" / "labeling" / f"balanced_{total}.csv"

    pass2 = load_pass2(Path(args.candidates), Path(args.out_dir))
    print(f"Re-label pass-2: {len(pass2)} baris.")
    changed = pass2[pass2["label_pass1"] != pass2["label_pass2"]]
    print(f"  Berubah: {len(changed)} ({len(changed)/max(len(pass2),1):.1%})")
    if len(changed):
        print("  Transisi (pass1 -> pass2):")
        print(changed.groupby(["label_pass1", "label_pass2"]).size().to_string())

    # pool penuh, readable
    df = pd.read_csv(src)
    df["comment_id"] = df["comment_id"].astype(str)
    before = len(df)
    df = df[df["text"].apply(is_readable)].copy()
    print(f"\nPool readable: {before} -> {len(df)}")

    # terapkan koreksi
    df["verified"] = False
    p2 = pass2.set_index("comment_id")
    mask = df["comment_id"].isin(p2.index)
    df.loc[mask, "label"] = df.loc[mask, "comment_id"].map(p2["label_pass2"])
    df.loc[mask, "confidence"] = df.loc[mask, "comment_id"].map(p2["conf_pass2"])
    df.loc[mask, "verified"] = True
    print(f"Koreksi diterapkan ke {int(mask.sum())} baris pool.")

    # dedup teks identik (simpan verified dulu, lalu confidence tertinggi)
    if args.dedup:
        df["_norm"] = df["text"].map(_norm)
        b = len(df)
        df = (df.sort_values(["verified", "confidence"], ascending=[False, False])
                .drop_duplicates("_norm", keep="first")
                .drop(columns="_norm"))
        print(f"Dedup teks: {b} -> {len(df)} (buang {b - len(df)})")

    # rebuild balanced: verified-first lalu confidence
    parts = []
    print("\nKetersediaan & komposisi per kelas:")
    for lab in LABELS:
        sub = df[df["label"] == lab].sort_values(
            ["verified", "confidence", "comment_id"], ascending=[False, False, True]
        )
        n = min(args.per_class, len(sub))
        take = sub.head(n)
        nver = int(take["verified"].sum())
        print(f"  {lab:<8} pool={len(sub):<5} ambil={n:<5} verified={nver} "
              f"conf {take['confidence'].min():.2f}-{take['confidence'].max():.2f}")
        if n < args.per_class:
            print(f"    ! {lab}: hanya {n} < {args.per_class} — diambil semua.")
        parts.append(take)
    balanced = pd.concat(parts, ignore_index=True)

    print(f"\nBalanced baru: {len(balanced)} baris")
    print(balanced["label"].value_counts().to_string())

    if args.commit:
        if out.exists():
            bak = out.with_suffix(".prerelabel.bak")
            shutil.copy(out, bak)
            print(f"Backup lama -> {bak}")
        cols = [c for c in ["comment_id", "label", "confidence", "text", "verified"] if c in balanced.columns]
        balanced[cols].to_csv(out, index=False)
        print(f"Disimpan: {out}")
    else:
        print("\n[dry-run] tak menulis. Tambah --commit untuk menyimpan.")


if __name__ == "__main__":
    main()

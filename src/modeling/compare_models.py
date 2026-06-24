"""Perbandingan 3 model (full 14k) — SVM sklearn vs SVM Spark MLlib vs IndoBERT.

Baca metrik test masing-masing dari outputs/reports/*.json (test set kanonik identik:
urut comment_id, seed 42, 70/20/10 → 1.411 sampel) lalu hasilkan:

  - model_comparison_full14k.csv : tabel akurasi + macro-F1 + F1 per-kelas
  - model_comparison_accuracy.png : chart batang AKURASI (metrik utama, keputusan
    user 2026-06-24 — data timpang 70% Netral → akurasi yang dipakai)

Tidak melatih apa pun; murni agregasi artefak yang sudah ada. Jalankan setelah
ketiga JSON tersedia:  python -m src.modeling.compare_models
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REP = Path("outputs/reports")
LABELS = ("Negatif", "Netral", "Positif")

# (nama tampil, file metrik). Urut dari baseline klasik -> deep learning.
SOURCES = (
    ("SVM sklearn", "svm_full14k_metrics.json"),
    ("SVM Spark MLlib", "svm_spark_metrics.json"),
    ("IndoBERT", "indobert_metrics.json"),
)


def _f1(per_class: dict, label: str) -> float:
    """Ambil F1 satu kelas; toleran key 'f1' (SVM) maupun 'f1-score' (IndoBERT)."""
    cell = per_class[label]
    return float(cell.get("f1", cell.get("f1-score")))


def _load(fname: str) -> dict:
    path = REP / fname
    if not path.exists():
        raise SystemExit(
            f"Metrik tak ada: {path}. Pastikan ketiga model sudah dievaluasi "
            f"(IndoBERT: taruh indobert_metrics.json dari Colab di outputs/reports/)."
        )
    return json.load(open(path))["test"]


def main() -> None:
    rows = []
    for name, fname in SOURCES:
        t = _load(fname)
        rows.append(
            {
                "model": name,
                "accuracy": round(float(t["accuracy"]), 4),
                "macro_f1": round(float(t["macro_f1"]), 4),
                **{f"f1_{lab.lower()}": round(_f1(t["per_class"], lab), 4) for lab in LABELS},
            }
        )

    df = pd.DataFrame(rows)
    out_csv = REP / "model_comparison_full14k.csv"
    df.to_csv(out_csv, index=False)

    # --- chart AKURASI (metrik utama) ---
    best_i = int(df["accuracy"].idxmax())
    colors = ["#4C72B0", "#55A868", "#DD8452"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(df["model"], df["accuracy"], color=colors, width=0.55)
    ax.set_ylim(0, min(1.0, df["accuracy"].max() + 0.12))
    ax.set_ylabel("Akurasi (test, n=1.411)")
    ax.set_title("Perbandingan Model (full 14k) — metrik utama: Akurasi")
    for i, (bar, v) in enumerate(zip(bars, df["accuracy"])):
        label = f"{v:.4f}" + ("  ★" if i == best_i else "")
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, label, ha="center", va="bottom")
    fig.tight_layout()
    out_png = REP / "model_comparison_accuracy.png"
    fig.savefig(out_png, dpi=120)

    print(df.to_string(index=False))
    print(f"\n-> Akurasi tertinggi: {df.loc[best_i, 'model']} ({df.loc[best_i, 'accuracy']})")
    print(f"Disimpan: {out_csv}\n          {out_png}")


if __name__ == "__main__":
    main()

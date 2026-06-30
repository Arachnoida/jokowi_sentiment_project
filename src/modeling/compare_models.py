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

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REP = Path("outputs/reports")
LABELS = ("Negatif", "Netral", "Positif")


def sources_for(tag: str):
    """(nama tampil, file metrik) per model untuk sebuah tag dataset."""
    suffix = "" if tag == "full14k" else f"_{tag}"
    return (
        ("SVM sklearn", f"svm_{tag}_metrics.json"),
        ("SVM Spark MLlib", f"svm_spark{suffix}_metrics.json"),
        ("IndoBERT", f"indobert{suffix}_metrics.json"),
        ("IndoBERTweet", f"indobertweet{suffix}_metrics.json"),
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
    ap = argparse.ArgumentParser(description="Bandingkan 3 model untuk satu tag dataset.")
    ap.add_argument("--tag", default="full14k",
                    help="Tag dataset: full14k (default) atau mis. balanced3k.")
    args = ap.parse_args()
    tag = args.tag

    rows = []
    for name, fname in sources_for(tag):
        # Spark di-DROP dari fokus (2026-06-30): lewati bila file metriknya tak ada.
        if not (REP / fname).exists():
            print(f"  (lewati {name}: {fname} tak ada)")
            continue
        t = json.load(open(REP / fname))["test"]
        rows.append(
            {
                "model": name,
                "accuracy": round(float(t["accuracy"]), 4),
                "macro_f1": round(float(t["macro_f1"]), 4),
                **{f"f1_{lab.lower()}": round(_f1(t["per_class"], lab), 4) for lab in LABELS},
            }
        )
    if not rows:
        raise SystemExit("Tak ada metrik model yang bisa dibaca utk tag ini.")

    df = pd.DataFrame(rows)
    # full14k mempertahankan nama lama (model_comparison_full14k.csv / _accuracy.png).
    csv_name = "model_comparison_full14k.csv" if tag == "full14k" else f"model_comparison_{tag}.csv"
    png_name = "model_comparison_accuracy.png" if tag == "full14k" else f"model_comparison_{tag}_accuracy.png"
    out_csv = REP / csv_name
    df.to_csv(out_csv, index=False)

    # Ukuran test set (jumlah support per-kelas dari model pertama).
    first = _load(sources_for(tag)[0][1])
    n_test = sum(int(first["per_class"][lab]["support"]) for lab in LABELS)

    # --- chart AKURASI (metrik utama) ---
    best_i = int(df["accuracy"].idxmax())
    palette = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3"]
    colors = [palette[i % len(palette)] for i in range(len(df))]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(df["model"], df["accuracy"], color=colors, width=0.55)
    ax.set_ylim(0, min(1.0, df["accuracy"].max() + 0.12))
    ax.set_ylabel(f"Akurasi (test, n={n_test:,})".replace(",", "."))
    ax.set_title(f"Perbandingan Model ({tag}) — metrik utama: Akurasi")
    for i, (bar, v) in enumerate(zip(bars, df["accuracy"])):
        label = f"{v:.4f}" + ("  ★" if i == best_i else "")
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, label, ha="center", va="bottom")
    fig.tight_layout()
    out_png = REP / png_name
    fig.savefig(out_png, dpi=120)

    print(df.to_string(index=False))
    print(f"\n-> Akurasi tertinggi: {df.loc[best_i, 'model']} ({df.loc[best_i, 'accuracy']})")
    print(f"Disimpan: {out_csv}\n          {out_png}")


if __name__ == "__main__":
    main()

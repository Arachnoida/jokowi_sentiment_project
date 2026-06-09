"""
src/modeling/evaluate.py
Evaluasi & pelaporan metrik bersama untuk SVM dan IndoBERT.

Metrik utama = macro-F1 (semua kelas diperlakukan setara). Karena dataset
seimbang (1.000/kelas), accuracy juga bermakna dan dilaporkan berdampingan.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.utils import setup_logger
from src.modeling.labels import LABELS

logger = setup_logger("modeling.evaluate")


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    Hitung metrik klasifikasi 3-kelas.

    Returns dict: accuracy, macro_f1, weighted_f1, per_class (P/R/F1/support),
    confusion_matrix (list-of-list, baris=true, kolom=pred).
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix,
    )

    labels = labels or LABELS
    label_ids = list(range(len(labels)))

    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=labels,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids).tolist()

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "per_class": {
            lab: {
                "precision": round(report[lab]["precision"], 4),
                "recall": round(report[lab]["recall"], 4),
                "f1": round(report[lab]["f1-score"], 4),
                "support": int(report[lab]["support"]),
            }
            for lab in labels
        },
        "confusion_matrix": cm,
        "labels": labels,
    }


def print_report(metrics: Dict, title: str = "EVALUASI") -> None:
    """Cetak metrik dalam format tabel ringkas ke konsol."""
    labels = metrics.get("labels", LABELS)
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1    : {metrics['macro_f1']:.4f}   <-- metrik utama")
    print(f"  Weighted-F1 : {metrics['weighted_f1']:.4f}")
    print("\n  Per kelas:")
    print(f"    {'Kelas':<10} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Sup':>6}")
    for lab in labels:
        pc = metrics["per_class"][lab]
        print(f"    {lab:<10} {pc['precision']:>7.3f} {pc['recall']:>7.3f} "
              f"{pc['f1']:>7.3f} {pc['support']:>6d}")
    print("\n  Confusion matrix (baris=aktual, kolom=prediksi):")
    header = "        " + "".join(f"{l[:6]:>8}" for l in labels)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"    {labels[i][:6]:<6}" + "".join(f"{v:>8d}" for v in row))
    print("=" * 64 + "\n")


def save_metrics(metrics: Dict, path: Path, model_name: str = "") -> Path:
    """Simpan metrik ke JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, **metrics} if model_name else metrics
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    logger.info("Metrik disimpan ke %s", path)
    return path


def plot_confusion_matrix(
    metrics: Dict, out_path: Optional[Path] = None, title: str = "Confusion Matrix",
):
    """Render confusion matrix sebagai heatmap (butuh matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib tidak tersedia — lewati plot confusion matrix.")
        return None

    labels = metrics.get("labels", LABELS)
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        logger.info("Confusion matrix disimpan ke %s", out_path)
    return fig


def compare_models(metrics_by_model: Dict[str, Dict]) -> str:
    """Bangun tabel teks perbandingan beberapa model (mis. SVM vs IndoBERT)."""
    lines = ["", "PERBANDINGAN MODEL",
             f"  {'Model':<16} {'Accuracy':>9} {'Macro-F1':>9} {'Weighted-F1':>12}"]
    for name, m in metrics_by_model.items():
        lines.append(f"  {name:<16} {m['accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
                     f"{m['weighted_f1']:>12.4f}")
    table = "\n".join(lines)
    print(table)
    return table

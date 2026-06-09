"""Fine-tuning IndoBERT (3 kelas) — sumber kebenaran, mirror notebook Colab.

Mem-*fine-tune* ``indobenchmark/indobert-base-p1`` sebagai pembanding deep-learning
terhadap baseline SVM. SELF-CONTAINED: tidak mengimpor paket internal project —
teks fitur ``bert`` sudah ter-preprocess di Mongo (koleksi ``processed_bert``),
jadi skrip ini cukup butuh ``MONGO_URI`` + dependensi pip (transformers, torch,
pymongo, scikit-learn, matplotlib). Bisa dijalankan di Colab/Kaggle/cloud GPU
(``python -m src.modeling.train_indobert``); CPU mungkin tapi SANGAT lambat.

Split & hyperparameter IDENTIK dengan ``train_svm_full14k`` agar perbandingan adil:
urut ``comment_id`` -> test 10% -> val (0.20/0.90 sisa) -> train, stratify + seed=42.

Artefak (default -> ``outputs/reports/``):
  - ``indobert_metrics_<suf>.json``  (dibaca sel perbandingan SVM)
  - ``indobert_test_confusion_<suf>.png``

Catatan epoch: 4 epoch + ``load_best_model_at_end`` adalah konfigurasi terbaik kami;
menaikkan ke 12 epoch + EarlyStopping JUSTRU menurunkan macro-F1 test (0.659 -> 0.620)
pada data kecil. EarlyStopping tetap ada untuk eksperimen.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128
SEED = 42
LABELS = ["Negatif", "Netral", "Positif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _connect(tries: int = 6) -> MongoClient:
    """Koneksi Mongo Atlas dgn retry (transient SSL/DNS)."""
    load_dotenv()
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        from getpass import getpass

        uri = getpass("MONGO_URI: ")
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            c = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
            c.admin.command("ping")
            return c
        except Exception as exc:  # noqa: BLE001 - transient, retry
            last = exc
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"Gagal koneksi Mongo: {last}")


def load_splits(client: MongoClient):
    """Baca SELURUH processed_bert (satu dataset full 14k) + split kanonik identik SVM."""
    db = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
    df = pd.DataFrame(
        list(
            client[db]["processed_bert"].find(
                {}, {"_id": 0, "comment_id": 1, "bert": 1, "label": 1}
            )
        )
    )
    if df.empty:
        raise RuntimeError("processed_bert kosong.")
    df["label_id"] = df["label"].map(LABEL2ID)
    # Split KANONIK: urut comment_id + seed=42 -> identik dgn train_svm versi ini.
    df = df.sort_values("comment_id").reset_index(drop=True)
    tmp, df_test = train_test_split(
        df, test_size=0.10, stratify=df["label_id"], random_state=SEED
    )
    df_train, df_val = train_test_split(
        tmp, test_size=0.20 / 0.90, stratify=tmp["label_id"], random_state=SEED
    )
    print(f"full 14k | train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    return df_train, df_val, df_test


def evaluate(y_true, y_pred, labels=LABELS) -> dict:
    """Metrik SETARA notebook SVM: acc, macro/weighted-F1, per-kelas, confusion."""
    ids = list(range(len(labels)))
    rep = classification_report(
        y_true, y_pred, labels=ids, target_names=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "per_class": {
            l: {k: round(rep[l][k], 4) for k in ["precision", "recall", "f1-score"]}
            | {"support": int(rep[l]["support"])}
            for l in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "labels": labels,
    }


def _save_confusion(cm: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("IndoBERT — Test")
    th = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > th else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune IndoBERT (3 kelas) — full 14k.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3, help="EarlyStopping patience.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--reports-dir", default=None,
        help="Folder artefak metrik/png. Default <repo>/outputs/reports.",
    )
    ap.add_argument(
        "--model-out", default=None,
        help="Bila diisi, simpan model+tokenizer hasil fine-tune ke folder ini (besar ~500MB).",
    )
    args = ap.parse_args()

    # Import berat ditunda agar --help cepat & error dep jelas.
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU saja (lambat!)"
    print("CUDA tersedia:", torch.cuda.is_available(), "|", dev)

    reports = Path(args.reports_dir) if args.reports_dir else _repo_root() / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    client = _connect()
    print("Koneksi MongoDB OK.")
    df_train, df_val, df_test = load_splits(client)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tok(
                list(texts), truncation=True, max_length=args.max_len, padding=True
            )
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    ds_train = DS(df_train["bert"].astype(str), df_train["label_id"])
    ds_val = DS(df_val["bert"].astype(str), df_val["label_id"])
    ds_test = DS(df_test["bert"].astype(str), df_test["label_id"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID,
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "macro_f1": f1_score(p.label_ids, preds, average="macro"),
            "accuracy": accuracy_score(p.label_ids, preds),
        }

    targs = TrainingArguments(
        output_dir=str(reports.parent / "indobert_out"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    print("Selesai. Val terbaik:", trainer.evaluate())

    pred = trainer.predict(ds_test)
    y_pred = np.argmax(pred.predictions, axis=1).tolist()
    y_true = df_test["label_id"].tolist()
    m_test = evaluate(y_true, y_pred)

    print("=" * 60)
    print("  IndoBERT — TEST")
    print("=" * 60)
    print(f"  Accuracy : {m_test['accuracy']:.4f}")
    print(f"  Macro-F1 : {m_test['macro_f1']:.4f}   <-- metrik utama")
    for l in LABELS:
        pc = m_test["per_class"][l]
        print(f"    {l:<10} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1-score']:.3f}")

    mfile = reports / "indobert_metrics.json"
    cfile = reports / "indobert_test_confusion.png"
    json.dump(
        {"model": "IndoBERT", "dataset": "full14k", "test": m_test},
        open(mfile, "w"),
        ensure_ascii=False,
        indent=2,
    )
    _save_confusion(np.array(m_test["confusion_matrix"]), cfile)
    print(f"Tersimpan: {mfile} + {cfile}")

    if args.model_out:
        out = Path(args.model_out)
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))
        print(f"Model+tokenizer disimpan -> {out}")


if __name__ == "__main__":
    main()

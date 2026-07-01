"""Latih & evaluasi SVM + TF-IDF pada SATU dataset: 14.107 komentar berlabel (full 14k).

Pipeline identik dengan IndoBERT agar perbandingan adil:
  - keanggotaan + label dari raw_comments (semua yang `label` ada); teks fitur `svm`
    dari processed_svm
  - split deterministik 70/20/10 (urut comment_id, stratified, seed=42)
  - GridSearchCV 24 kombinasi (ngram x min_df x C) dgn PredefinedSplit di val (f1_macro)
  - refit train+val pakai param terbaik -> evaluasi test (test dibiarkan utuh)
  - metrik utama macro-F1; simpan JSON + confusion matrix PNG ke outputs/reports/

Proyek memakai SATU dataset saja (full 14k, imbalanced) — versi-versi subset lama
(6k/balanced/10k) sudah ditinggalkan.
"""
import argparse
import json
import os
import pathlib
import time

import certifi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.modeling.subset import load_subset_ids

SEED = 42
TEXT, LAB = "svm", "label_id"
LABELS = ["Negatif", "Netral", "Positif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.1, 0.5, 1.0, 2.0],
}


def _connect(tries: int = 6) -> MongoClient:
    load_dotenv()
    uri = os.environ["MONGO_URI"]
    last = None
    for attempt in range(1, tries + 1):
        try:
            c = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
            c.admin.command("ping")
            return c
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Gagal koneksi Mongo: {last}")


def load_df(client: MongoClient) -> pd.DataFrame:
    db = client[DB]
    mem = pd.DataFrame(
        list(db["raw_comments"].find({"label": {"$exists": True}}, {"_id": 0, "comment_id": 1, "label": 1}))
    )
    mem["label_id"] = mem["label"].map(LABEL2ID)
    sv = pd.DataFrame(list(
        db["processed_svm"].find({}, {"_id": 0, "comment_id": 1, "svm": 1, "text": 1})
    ))
    df = mem.merge(sv, on="comment_id", how="left")
    df["svm"] = df["svm"].fillna("")
    df["text"] = df["text"].fillna("")
    return df


def build() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=SEED)),
        ]
    )


def split_version(sub: pd.DataFrame):
    sub = sub.sort_values("comment_id").reset_index(drop=True)
    tmp, te = train_test_split(sub, test_size=0.10, stratify=sub[LAB], random_state=SEED)
    tr, va = train_test_split(tmp, test_size=0.20 / 0.90, stratify=tmp[LAB], random_state=SEED)
    return tr, va, te


def _nonempty(d: pd.DataFrame) -> pd.DataFrame:
    return d[d[TEXT].str.len() > 0]


def evaluate(yt, yp) -> dict:
    ids = list(range(3))
    rep = classification_report(
        yt, yp, labels=ids, target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(accuracy_score(yt, yp), 4),
        "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
        "per_class": {
            l: {"f1": round(rep[l]["f1-score"], 4), "support": int(rep[l]["support"])}
            for l in LABELS
        },
        "confusion_matrix": confusion_matrix(yt, yp, labels=ids).tolist(),
    }


def run(df: pd.DataFrame) -> dict:
    """Latih GridSearch di train+val (PredefinedSplit), evaluasi pada test utuh."""
    tr, va, te = split_version(df)
    tr, va = _nonempty(tr), _nonempty(va)  # buang teks kosong dari train/val saja
    X = pd.concat([tr[TEXT], va[TEXT]], ignore_index=True)
    y = pd.concat([tr[LAB], va[LAB]], ignore_index=True)
    ps = PredefinedSplit(np.r_[np.full(len(tr), -1), np.zeros(len(va))])
    gs = GridSearchCV(build(), GRID, scoring="f1_macro", cv=ps, n_jobs=-1)
    gs.fit(X, y)
    best = gs.best_estimator_  # sudah refit pada train+val oleh GridSearchCV (refit=True)
    yp = best.predict(te[TEXT])
    m = evaluate(te[LAB].tolist(), yp)
    m["n_train"] = int(len(tr) + len(va))
    m["n_test"] = int(len(te))
    m["best_params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()}
    return m, te, yp, best, gs


def _save_detail(rep, tag, te, yp, best, gs) -> None:
    """Simpan artefak proses untuk notebook: prediksi/komentar, grid, top fitur."""
    # 1) Prediksi per-komentar (test) — teks + label asli + prediksi + benar/salah.
    preds = pd.DataFrame({
        "comment_id": te["comment_id"].to_numpy(),
        "text": te["text"].to_numpy(),
        "label_asli": [LABELS[i] for i in te[LAB].to_numpy()],
        "prediksi": [LABELS[i] for i in yp],
    })
    preds["benar"] = preds["label_asli"] == preds["prediksi"]
    # Keyakinan = margin keputusan OvR (skor kelas-tertinggi − kelas-kedua).
    sc = best.decision_function(te[TEXT])
    top2 = np.sort(sc, axis=1)[:, -2:]
    preds["keyakinan"] = np.round(top2[:, 1] - top2[:, 0], 4)
    preds.to_csv(rep / f"svm_{tag}_predictions.csv", index=False)

    # 2) Hasil grid search (24 kombinasi) diurut peringkat.
    gr = pd.DataFrame(gs.cv_results_)
    keep = [c for c in gr.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score"]
    gr[keep].sort_values("rank_test_score").to_csv(rep / f"svm_{tag}_grid.csv", index=False)

    # 3) Top fitur diskriminatif per kelas (koefisien LinearSVC OvR).
    tfidf, clf = best.named_steps["tfidf"], best.named_steps["clf"]
    feats = np.array(tfidf.get_feature_names_out())
    rows = []
    for ci, lab in enumerate(LABELS):
        coef = clf.coef_[ci]
        for rank, idx in enumerate(coef.argsort()[::-1][:15], 1):
            rows.append({"kelas": lab, "rank": rank, "fitur": feats[idx],
                         "koef": round(float(coef[idx]), 3)})
    pd.DataFrame(rows).to_csv(rep / f"svm_{tag}_top_features.csv", index=False)


def _repo_root() -> pathlib.Path:
    root = pathlib.Path.cwd()
    for p in [root, *root.parents]:
        if (p / "configs").exists() or (p / ".git").exists():
            return p
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description="Latih SVM+TF-IDF (sklearn).")
    ap.add_argument("--subset", default=None,
                    help="CSV allowlist comment_id (mis. balanced_3000.csv). "
                         "Default: semua baris berlabel (full 14k).")
    ap.add_argument("--tag", default="full14k",
                    help="Suffix artefak: svm_<tag>_metrics.json. Default full14k.")
    args = ap.parse_args()
    tag = args.tag

    client = _connect()
    df = load_df(client)
    if args.subset:
        sub = pd.read_csv(args.subset)
        sub["comment_id"] = sub["comment_id"].astype(str)
        df = df[df["comment_id"].astype(str).isin(set(sub["comment_id"]))].reset_index(drop=True)
        if "label" in sub.columns:
            # Override label dari subset CSV (mendukung v1/v1audited tanpa ubah Mongo).
            # Fitur `svm` tetap dari processed_svm (label-independent).
            lab = dict(zip(sub["comment_id"], sub["label"]))
            df["label"] = df["comment_id"].astype(str).map(lab)
            df["label_id"] = df["label"].map(LABEL2ID)
            print(f"[{tag}] label dari subset CSV (override raw_comments)")
    print(f"{len(df)} komentar berlabel [{tag}] | svm kosong (drop dari train/val): "
          f"{int((df['svm'].str.len() == 0).sum())}")

    m, te, yp, best, gs = run(df)
    line = " ".join(f"{l[:3]}={m['per_class'][l]['f1']:.2f}" for l in LABELS)
    print(
        f"[{tag}] n_train={m['n_train']:<5} n_test={m['n_test']:<5} "
        f"macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} | {line} | {m['best_params']}"
    )

    rep = _repo_root() / "outputs" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"model": "SVM+TF-IDF", "test": m},
        open(rep / f"svm_{tag}_metrics.json", "w"),
        ensure_ascii=False,
        indent=2,
    )
    _save_detail(rep, tag, te, yp, best, gs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(f"SVM+TF-IDF — Test (macro-F1={m['macro_f1']:.3f})")
    th = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > th else "black")
    fig.tight_layout()
    fig.savefig(rep / f"svm_{tag}_confusion.png", dpi=120)

    # --- Perbandingan SVM vs IndoBERT (legacy 2-model, hanya untuk full14k).
    #     Perbandingan 3-model lengkap pakai src.modeling.compare_models --tag <tag>.
    bfile = rep / "indobert_metrics.json"
    if tag == "full14k" and bfile.exists():
        b = json.load(open(bfile))["test"]
        winner = "SVM" if m["macro_f1"] > b["macro_f1"] else "IndoBERT"
        cmp = pd.DataFrame([
            {"model": "SVM+TF-IDF", "macro_F1": m["macro_f1"], "accuracy": m["accuracy"]},
            {"model": "IndoBERT", "macro_F1": round(b["macro_f1"], 4), "accuracy": round(b["accuracy"], 4)},
        ])
        cmp.to_csv(rep / "model_comparison_full14k.csv", index=False)
        fig2, ax2 = plt.subplots(figsize=(5, 4.2))
        ax2.bar(cmp["model"], cmp["macro_F1"], color=["#4C72B0", "#DD8452"], width=0.5)
        ax2.set_ylim(0, max(0.8, cmp["macro_F1"].max() + 0.1))
        ax2.set_ylabel("macro-F1 (test)")
        ax2.set_title(f"SVM vs IndoBERT (full 14k) — menang: {winner}")
        for i, v in enumerate(cmp["macro_F1"]):
            ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(rep / "svm_vs_indobert_full14k.png", dpi=120)
        print("\nSVM vs IndoBERT (macro-F1):")
        print(cmp.to_string(index=False))
        print(f"-> menang: {winner}")
    else:
        print("\n(indobert_metrics.json belum ada — jalankan IndoBERT, taruh hasilnya di "
              "outputs/reports/, lalu jalankan ulang skrip ini untuk tabel perbandingan.)")

    print("\nTersimpan ke", rep)


if __name__ == "__main__":
    main()

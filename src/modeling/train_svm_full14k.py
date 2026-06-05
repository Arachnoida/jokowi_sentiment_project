"""Latih & evaluasi SVM + TF-IDF pada dataset PENUH 14.107 komentar berlabel.

Pipeline identik dengan notebook train_svm.ipynb (agar setara & adil):
  - keanggotaan kanonik + label dari raw_comments; teks fitur `svm` dari processed_svm
  - split deterministik 70/20/10 (urut comment_id, stratified, seed=42)
  - GridSearchCV 24 kombinasi (ngram x min_df x C) dgn PredefinedSplit di val (f1_macro)
  - refit train+val pakai param terbaik -> evaluasi test (test dibiarkan utuh)
  - metrik utama macro-F1; simpan JSON + confusion matrix PNG ke outputs/reports/

Versi 'v5 full 14k' = SELURUH komentar berlabel (imbalanced). Untuk konteks, skrip juga
melatih ulang keempat versi kanonik memakai processed_svm yang kini sudah lengkap,
sehingga satu tabel memuat v1..v5 yang konsisten.
"""
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

SEED = 42
TEXT, LAB = "svm", "label_id"
LABELS = ["Negatif", "Netral", "Positif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
VERSIONS = [
    ("v1 imbalanced 6k", "in_set6k"),
    ("v2 balanced 3k", "in_balanced_set"),
    ("v3 imbalanced 10k", "in_set10k"),
    ("v4 balanced 10k", "in_balanced10k"),
    ("v5 full 14k", "in_full14k"),  # sintetis: semua komentar berlabel
]
FLAGS = [f for _, f in VERSIONS if f != "in_full14k"]
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
    proj = {"_id": 0, "comment_id": 1, "label": 1}
    proj.update({f: 1 for f in FLAGS})
    mem = pd.DataFrame(list(db["raw_comments"].find({"label": {"$exists": True}}, proj)))
    for f in FLAGS:
        mem[f] = mem[f].fillna(False)
    mem["in_full14k"] = True  # versi sintetis: seluruh anggota berlabel
    mem["label_id"] = mem["label"].map(LABEL2ID)
    sv = pd.DataFrame(list(db["processed_svm"].find({}, {"_id": 0, "comment_id": 1, "svm": 1})))
    df = mem.merge(sv, on="comment_id", how="left")
    df["svm"] = df["svm"].fillna("")
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


def run_version(df: pd.DataFrame, flag: str) -> dict:
    tr, va, te = split_version(df[df[flag]])
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
    return m


def main() -> None:
    client = _connect()
    df = load_df(client)
    print(f"{len(df)} member berlabel | per versi:", {f: int(df[f].sum()) for f, in [(f,) for _, f in VERSIONS]})
    print("svm kosong (drop dari train/val):", int((df["svm"].str.len() == 0).sum()))

    results = {}
    for name, flag in VERSIONS:
        m = run_version(df, flag)
        results[name] = m
        line = " ".join(f"{l[:3]}={m['per_class'][l]['f1']:.2f}" for l in LABELS)
        print(
            f"[{name:<18}] n_train={m['n_train']:<5} n_test={m['n_test']:<5} "
            f"macro-F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f} | {line} | {m['best_params']}"
        )

    root = pathlib.Path.cwd()
    for p in [root, *root.parents]:
        if (p / "configs").exists() or (p / ".git").exists():
            root = p
            break
    rep = root / "outputs" / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    json.dump(
        {"model": "SVM+TF-IDF", "by_version": results},
        open(rep / "svm_full14k_metrics.json", "w"),
        ensure_ascii=False,
        indent=2,
    )

    rows = []
    for name, _ in VERSIONS:
        m = results[name]
        rows.append(
            {
                "versi": name,
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "macro_F1": m["macro_f1"],
                "accuracy": m["accuracy"],
                "weighted_F1": m["weighted_f1"],
                **{f"F1_{l}": m["per_class"][l]["f1"] for l in LABELS},
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(rep / "svm_full14k_table.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in VERSIONS]
    metrics = ["macro_F1"] + [f"F1_{l}" for l in LABELS]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, met in enumerate(metrics):
        ax.bar(x + k * w, [comp.loc[comp.versi == n, met].values[0] for n in names], w, label=met)
    ax.set_xticks(x + w * 1.5, names, rotation=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.legend(ncol=4, fontsize=8)
    ax.set_title("SVM lintas versi dataset (termasuk v5 full 14k)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(rep / "svm_full14k_compare.png", dpi=120)

    fig2, axes = plt.subplots(1, len(VERSIONS), figsize=(4 * len(VERSIONS), 3.6))
    for ax, (name, _) in zip(axes, VERSIONS):
        cm = np.array(results[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), [l[:3] for l in LABELS])
        ax.set_yticks(range(3), [l[:3] for l in LABELS])
        ax.set_title(f"{name}\nmacroF1={results[name]['macro_f1']:.3f}", fontsize=9)
        th = cm.max() / 2
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > th else "black",
                )
    fig2.supxlabel("Prediksi")
    fig2.supylabel("Aktual")
    fig2.tight_layout()
    fig2.savefig(rep / "svm_full14k_confusion.png", dpi=120)

    # --- Perbandingan SVM vs IndoBERT (jika metrik IndoBERT per versi tersedia) ---
    SUF = {
        "v1 imbalanced 6k": "is6",
        "v2 balanced 3k": "ibs",
        "v3 imbalanced 10k": "is10",
        "v4 balanced 10k": "ib10",
        "v5 full 14k": "if14",
    }
    cmp_rows = []
    for name, _ in VERSIONS:
        s = results[name]
        f = rep / f"indobert_metrics_{SUF[name]}.json"
        b = json.load(open(f))["test"] if f.exists() else None
        cmp_rows.append(
            {
                "versi": name,
                "SVM": s["macro_f1"],
                "IndoBERT": round(b["macro_f1"], 4) if b else None,
                "winner": "-" if not b else ("SVM" if s["macro_f1"] > b["macro_f1"] else "IndoBERT"),
            }
        )
    cmp = pd.DataFrame(cmp_rows)
    have = [r for r in cmp_rows if r["IndoBERT"] is not None]
    if have:
        names_h = [r["versi"] for r in have]
        xh = np.arange(len(names_h))
        wb = 0.38
        fig3, ax3 = plt.subplots(figsize=(11, 4.5))
        ax3.bar(xh - wb / 2, [r["SVM"] for r in have], wb, label="SVM+TF-IDF")
        ax3.bar(xh + wb / 2, [r["IndoBERT"] for r in have], wb, label="IndoBERT")
        ax3.set_xticks(xh, names_h, rotation=12)
        ax3.set_ylim(0, 0.8)
        ax3.set_ylabel("macro-F1 (test)")
        ax3.legend()
        ax3.set_title("SVM vs IndoBERT lintas versi (termasuk v5 full 14k)")
        ax3.grid(axis="y", alpha=0.3)
        for i, r in enumerate(have):
            ax3.text(i - wb / 2, r["SVM"] + 0.01, f"{r['SVM']:.3f}", ha="center", fontsize=8)
            ax3.text(i + wb / 2, r["IndoBERT"] + 0.01, f"{r['IndoBERT']:.3f}", ha="center", fontsize=8)
        fig3.tight_layout()
        fig3.savefig(rep / "svm_vs_indobert_full14k.png", dpi=120)
        cmp.to_csv(rep / "model_comparison_full14k.csv", index=False)
        print("\nSVM vs IndoBERT:")
        print(cmp.to_string(index=False))
    else:
        print("\n(Metrik IndoBERT belum ada — jalankan notebook Colab utk tiap versi, "
              "taruh indobert_metrics_{suf}.json di outputs/reports/, lalu jalankan ulang skrip ini.)")

    print("\nTersimpan ke", rep)
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()

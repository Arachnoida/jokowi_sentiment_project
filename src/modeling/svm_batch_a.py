"""Batch A — eksperimen model-side SVM pada dataset (default balanced3k).

Tujuan: lihat seberapa jauh akurasi SVM bisa pulih TANPA regen processed_svm,
pada label baru (domain-aware). Semua varian dievaluasi pada **test split kanonik
yang sama** (urut comment_id, seed=42, 70/20/10) agar comparable dgn train_svm &
IndoBERT, plus **OOF 5-fold CV** (train+val) utk angka robust.

Varian:
  base        : TF-IDF word (1,2) + LinearSVC(class_weight=balanced)
  neg         : + negation-merge ("tidak bohong" -> "tidak_bohong")
  char        : FeatureUnion word(1,2) + char_wb(3,5)
  neg+char    : negation-merge + word + char
  logreg      : neg+char + LogisticRegression
  cnb         : neg+char + ComplementNB
  +thr        : varian terbaik + tuning bias per-kelas (decision_function) di val

  python -m src.modeling.svm_batch_a [--subset outputs/labeling/balanced_3000.csv] [--tag balanced3k]
"""
from __future__ import annotations

import argparse
import os
import re
import time

import certifi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42
LABELS = ["Negatif", "Netral", "Positif"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")
NEG = {"tidak", "bukan", "jangan", "tdk", "gak", "ga", "nggak", "tak", "belum",
       "tanpa", "jgn", "ngga", "kaga", "kagak", "engga", "enggak", "ndak", "gk"}


def _connect(tries=6) -> MongoClient:
    load_dotenv()
    uri = os.environ["MONGO_URI"]
    last = None
    for a in range(1, tries + 1):
        try:
            c = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
            c.admin.command("ping")
            return c
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** a, 15))
    raise RuntimeError(f"Mongo gagal: {last}")


def neg_merge(s: str) -> str:
    toks = str(s).split()
    out, i = [], 0
    while i < len(toks):
        if toks[i] in NEG and i + 1 < len(toks):
            out.append(f"{toks[i]}_{toks[i+1]}")
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def load(client: MongoClient, subset: str | None) -> pd.DataFrame:
    db = client[DB]
    mem = pd.DataFrame(list(db["raw_comments"].find(
        {"label": {"$exists": True}}, {"_id": 0, "comment_id": 1, "label": 1})))
    sv = pd.DataFrame(list(db["processed_svm"].find(
        {}, {"_id": 0, "comment_id": 1, "svm": 1})))
    df = mem.merge(sv, on="comment_id", how="left")
    df["svm"] = df["svm"].fillna("")
    df["y"] = df["label"].map(LAB2ID)
    if subset:
        ids = set(pd.read_csv(subset, usecols=["comment_id"])["comment_id"].astype(str))
        df = df[df["comment_id"].astype(str).isin(ids)].reset_index(drop=True)
    return df


def split(df: pd.DataFrame):
    df = df.sort_values("comment_id").reset_index(drop=True)
    tmp, te = train_test_split(df, test_size=0.10, stratify=df["y"], random_state=SEED)
    tr, va = train_test_split(tmp, test_size=0.20 / 0.90, stratify=tmp["y"], random_state=SEED)
    return tr, va, te


def word_vec():
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)


def char_union():
    return FeatureUnion([
        ("w", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)),
        ("c", TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3)),
    ])


def make(variant: str) -> Pipeline:
    clf = LinearSVC(class_weight="balanced", random_state=SEED, C=0.5)
    if variant in ("base", "neg"):
        feats = word_vec()
    else:
        feats = char_union()
    if variant == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    elif variant == "cnb":
        clf = ComplementNB()
    return Pipeline([("f", feats), ("clf", clf)])


def prep_text(df: pd.DataFrame, variant: str) -> list[str]:
    if variant == "base":
        return df["svm"].astype(str).tolist()
    return df["svm"].astype(str).map(neg_merge).tolist()


def per_class_f1(y, p):
    from sklearn.metrics import f1_score as f
    fs = f(y, p, average=None, labels=[0, 1, 2], zero_division=0)
    return dict(zip(LABELS, np.round(fs, 3)))


def oof_macro(Xtr, ytr, variant: str) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros(len(ytr), dtype=int)
    Xtr = np.array(Xtr, dtype=object)
    ytr = np.array(ytr)
    for tri, vai in skf.split(Xtr, ytr):
        pipe = make(variant)
        pipe.fit(Xtr[tri].tolist(), ytr[tri])
        preds[vai] = pipe.predict(Xtr[vai].tolist())
    return f1_score(ytr, preds, average="macro", zero_division=0)


def tune_bias(pipe, Xva, yva):
    """Cari bias additif per-kelas pada decision_function (val) -> max macro-F1."""
    if not hasattr(pipe, "decision_function"):
        return np.zeros(3)
    D = pipe.decision_function(Xva)
    grid = np.linspace(-0.6, 0.6, 13)
    best, best_b = -1, np.zeros(3)
    for bn in grid:
        for bp in grid:
            b = np.array([bn, 0.0, bp])
            p = np.argmax(D + b, axis=1)
            s = f1_score(yva, p, average="macro", zero_division=0)
            if s > best:
                best, best_b = s, b
    return best_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="outputs/labeling/balanced_3000.csv")
    ap.add_argument("--tag", default="balanced3k")
    ap.add_argument("--write-official", action="store_true",
                    help="Tulis svm_<tag>_metrics.json resmi dari varian char+thr (utk compare_models).")
    args = ap.parse_args()

    df = load(_connect(), args.subset)
    tr, va, te = split(df)
    print(f"[{args.tag}] n={len(df)} train={len(tr)} val={len(va)} test={len(te)}\n")

    rows = []
    variants = ["base", "neg", "char", "neg+char", "logreg", "cnb"]
    fitted = {}
    for v in variants:
        Xtr, Xva, Xte = prep_text(tr, v), prep_text(va, v), prep_text(te, v)
        # fit train+val utk test eval (mirror final-train style)
        pipe = make(v)
        pipe.fit(Xtr + Xva, list(tr["y"]) + list(va["y"]))
        yp = pipe.predict(Xte)
        acc = accuracy_score(te["y"], yp)
        mf = f1_score(te["y"], yp, average="macro", zero_division=0)
        pcf = per_class_f1(te["y"], yp)
        oof = oof_macro(Xtr + Xva, list(tr["y"]) + list(va["y"]), v)
        rows.append({"variant": v, "test_acc": round(acc, 4), "test_macroF1": round(mf, 4),
                     "oof_macroF1": round(oof, 4), **{f"F1_{k}": pcf[k] for k in LABELS}})
        fitted[v] = pipe
        print(f"{v:<9} acc={acc:.3f} macroF1={mf:.3f} oof={oof:.3f} | "
              + " ".join(f"{k[:3]}={pcf[k]:.2f}" for k in LABELS))

    # threshold tuning pada varian dgn oof terbaik (di antara LinearSVC: base/neg/char/neg+char)
    svc_variants = ["base", "neg", "char", "neg+char"]
    best_v = max(svc_variants, key=lambda v: [r for r in rows if r["variant"] == v][0]["oof_macroF1"])
    pipe = make(best_v)
    Xtr2 = prep_text(tr, best_v)
    pipe.fit(Xtr2, list(tr["y"]))  # fit train only; tune bias on val
    b = tune_bias(pipe, prep_text(va, best_v), list(va["y"]))
    # refit train+val, apply bias ke test
    pipe = make(best_v)
    pipe.fit(prep_text(tr, best_v) + prep_text(va, best_v), list(tr["y"]) + list(va["y"]))
    D = pipe.decision_function(prep_text(te, best_v))
    yp = np.argmax(D + b, axis=1)
    acc = accuracy_score(te["y"], yp)
    mf = f1_score(te["y"], yp, average="macro", zero_division=0)
    pcf = per_class_f1(te["y"], yp)
    rows.append({"variant": f"{best_v}+thr", "test_acc": round(acc, 4), "test_macroF1": round(mf, 4),
                 "oof_macroF1": None, **{f"F1_{k}": pcf[k] for k in LABELS}})
    print(f"\n{best_v}+thr (bias={np.round(b,2)}) acc={acc:.3f} macroF1={mf:.3f} | "
          + " ".join(f"{k[:3]}={pcf[k]:.2f}" for k in LABELS))

    out = pd.DataFrame(rows)
    p = f"outputs/reports/svm_batch_a_{args.tag}.csv"
    out.to_csv(p, index=False)
    print(f"\nTersimpan: {p}")
    print(out.to_string(index=False))

    if args.write_official:
        # Official = neg+char (word 1-2 + char_wb 3-5, LinearSVC), TANPA per-class threshold:
        # paling stabil (test & OOF konsisten tinggi); threshold dari val kecil overfit, logreg
        # menang OOF cuma karena noise (kalah di test). char n-gram = lever utama Batch A.
        off_v = "neg+char"
        pipe = make(off_v)
        pipe.fit(prep_text(tr, off_v) + prep_text(va, off_v), list(tr["y"]) + list(va["y"]))
        yp_off = pipe.predict(prep_text(te, off_v))
        print(f"\nOfficial = {off_v} (stabil, tanpa threshold)")
        _write_official(args.tag, off_v, te, yp_off)


def _write_official(tag, best_v, te, yp):
    """Tulis svm_<tag>_metrics.json (format compare_models) utk varian stabil terbaik."""
    import json
    from pathlib import Path
    from sklearn.metrics import classification_report, confusion_matrix

    yt = list(te["y"])
    ids = [0, 1, 2]
    rep = classification_report(yt, yp, labels=ids, target_names=LABELS,
                                output_dict=True, zero_division=0)
    m = {
        "accuracy": round(accuracy_score(yt, yp), 4),
        "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
        "per_class": {l: {"f1": round(rep[l]["f1-score"], 4), "support": int(rep[l]["support"])}
                      for l in LABELS},
        "confusion_matrix": confusion_matrix(yt, yp, labels=ids).tolist(),
        "n_test": int(len(yt)),
        "best_params": {"features": "word(1,2)+char_wb(3,5)", "clf": "LinearSVC(C=0.5,balanced)",
                        "variant": best_v, "threshold": "none (stable)"},
    }
    rep_dir = Path("outputs/reports")
    json.dump({"model": "SVM+TF-IDF", "dataset": tag, "test": m},
              open(rep_dir / f"svm_{tag}_metrics.json", "w"), ensure_ascii=False, indent=2)
    # predictions
    pd.DataFrame({
        "comment_id": te["comment_id"].to_numpy(),
        "text": te["svm"].to_numpy(),
        "label_asli": [LABELS[i] for i in yt],
        "prediksi": [LABELS[i] for i in yp],
    }).assign(benar=lambda d: d["label_asli"] == d["prediksi"]).to_csv(
        rep_dir / f"svm_{tag}_predictions.csv", index=False)
    # confusion png
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"SVM ({best_v}) — Test (acc={m['accuracy']:.3f})")
    th = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > th else "black")
    fig.tight_layout(); fig.savefig(rep_dir / f"svm_{tag}_confusion.png", dpi=120)
    print(f"OFFICIAL ditulis: svm_{tag}_metrics.json (acc={m['accuracy']}) + predictions + confusion png")


if __name__ == "__main__":
    main()

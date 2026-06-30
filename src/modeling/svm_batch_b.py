"""Batch B — SVM + fitur rekayasa DOMAIN (di atas TF-IDF word+char).

Ide: label v2 domain-aware bergantung pada SIAPA diserang (penuduh vs Jokowi) + verdict
klaim — sinyal yang sulit ditangkap bag-of-words. Tambahkan fitur eksplisit:
  - mention kubu penuduh / Jokowi-pembela
  - kata serangan + kata tuntutan-hukuman
  - verdict palsu / asli / fitnah
  - negasi
  - INTERAKSI: penuduh×(serang|hukum) -> sinyal Negatif; Jokowi×(serang|hukum) -> Positif
digabung (ColumnTransformer) dgn TF-IDF (word 1-2 + char_wb 3-5) lalu LinearSVC/LogReg.

Eval: test split kanonik (seed=42) + OOF 5-fold. Bandingkan dgn baseline neg+char (0,7533).
  python -m src.modeling.svm_batch_b [--write-official]
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
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.svm import LinearSVC

SEED = 42
LABELS = ["Negatif", "Netral", "Positif"]
L2I = {l: i for i, l in enumerate(LABELS)}
DB = os.environ.get("MONGO_DB_NAME", "youtube_sentiment")

ACCUSER = r"\broy\b|suryo|\broi\b|panci|\brrt\b|rismon|\btifa\b|rizal|fadhilah|fadilah|penuduh|menuduh"
JOKOWI = r"jokowi|mulyono|pria solo|\bjkw\b|\bugm\b|rektor|dekan"
PUNISH = r"penjara|adili|tangkap|\btahan\b|ditahan|\bbui\b|laporin|dipenjara|dihukum|\bhukum\b|tersangka|jeblos|borgol"
ATTACK = r"stres|stress|gila|bodoh|goblok|tolol|dungu|odgj|sakit hati|dendam|najis|laknat|setan|bohong|pembohong|sinting|edan|sarap|bego|kurang ajar"
PALSU = r"palsu|bodong|edit|rekayasa|tipu|aspal|abal"
ASLI = r"asli|terbukti|\bsah\b|valid"
FITNAH = r"fitnah|hoax|hoaks|cari sensasi|cari panggung|pencemaran"
NEG = r"\b(tidak|bukan|jangan|tdk|gak|nggak|tak|belum|tanpa|jgn|ngga)\b"


def _feat(texts) -> np.ndarray:
    rows = []
    for t in texts:
        s = str(t).lower()
        acc = 1 if re.search(ACCUSER, s) else 0
        jkw = 1 if re.search(JOKOWI, s) else 0
        pun = 1 if re.search(PUNISH, s) else 0
        atk = 1 if re.search(ATTACK, s) else 0
        palsu = 1 if re.search(PALSU, s) else 0
        asli = 1 if re.search(ASLI, s) else 0
        fit = 1 if re.search(FITNAH, s) else 0
        neg = 1 if re.search(NEG, s) else 0
        rows.append([
            acc, jkw, pun, atk, palsu, asli, fit, neg,
            acc * (pun or atk),      # serang/hukum penuduh -> Negatif
            jkw * (pun or atk),      # serang/hukum Jokowi -> Positif
            acc * fit,               # penuduh + fitnah -> Negatif
            palsu, asli,             # verdict langsung
        ])
    return np.array(rows, dtype=float)


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


def load(client, subset):
    db = client[DB]
    mem = pd.DataFrame(list(db["raw_comments"].find({"label": {"$exists": True}},
                                                    {"_id": 0, "comment_id": 1, "label": 1})))
    sv = pd.DataFrame(list(db["processed_svm"].find({}, {"_id": 0, "comment_id": 1, "svm": 1, "text": 1})))
    df = mem.merge(sv, on="comment_id", how="left")
    df["svm"] = df["svm"].fillna("")
    df["text"] = df["text"].fillna("")
    df["y"] = df["label"].map(L2I)
    ids = set(pd.read_csv(subset, usecols=["comment_id"])["comment_id"].astype(str))
    return df[df["comment_id"].astype(str).isin(ids)].reset_index(drop=True)


def split(df):
    df = df.sort_values("comment_id").reset_index(drop=True)
    tmp, te = train_test_split(df, test_size=0.10, stratify=df["y"], random_state=SEED)
    tr, va = train_test_split(tmp, test_size=0.20 / 0.90, stratify=tmp["y"], random_state=SEED)
    return tr, va, te


def build(clf):
    ct = ColumnTransformer([
        ("word", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), "svm"),
        ("char", TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3), "svm"),
        ("dom", Pipeline([("f", FunctionTransformer(_feat)), ("s", MaxAbsScaler())]), "text"),
    ])
    return Pipeline([("ct", ct), ("clf", clf)])


def pcf(y, p):
    fs = f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)
    return {l: round(v, 3) for l, v in zip(LABELS, fs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="outputs/labeling/balanced_3000.csv")
    ap.add_argument("--tag", default="balanced3k")
    ap.add_argument("--write-official", action="store_true")
    args = ap.parse_args()

    df = load(_connect(), args.subset)
    tr, va, te = split(df)
    print(f"[{args.tag}] n={len(df)} train={len(tr)} val={len(va)} test={len(te)}\n")

    # FAST PATH: tulis official via 1x fit (LinearSVC C=0.5), TANPA OOF/sweep (hemat CPU).
    if args.write_official:
        pipe = build(LinearSVC(class_weight="balanced", C=0.5, random_state=SEED))
        pipe.fit(pd.concat([tr, va])[["svm", "text"]], pd.concat([tr, va])["y"])
        yp = pipe.predict(te[["svm", "text"]])
        acc = accuracy_score(te["y"], yp); mf = f1_score(te["y"], yp, average="macro", zero_division=0)
        f = pcf(te["y"], yp)
        print(f"domfeat+linsvc test_acc={acc:.4f} macroF1={mf:.4f} | "
              + " ".join(f"{k[:3]}={f[k]:.2f}" for k in LABELS))
        from src.modeling.svm_batch_a import _write_official
        _write_official(args.tag, "domfeat+linsvc(word+char)", te, yp)
        return

    Xall = df[["svm", "text"]]
    clfs = {"linsvc": LinearSVC(class_weight="balanced", C=0.5, random_state=SEED),
            "logreg": LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)}
    best = None
    for name, clf in clfs.items():
        pipe = build(clf)
        pipe.fit(pd.concat([tr, va])[["svm", "text"]], pd.concat([tr, va])["y"])
        yp = pipe.predict(te[["svm", "text"]])
        acc = accuracy_score(te["y"], yp); mf = f1_score(te["y"], yp, average="macro", zero_division=0)
        oof = cross_val_predict(build(clf), Xall, df["y"],
                                cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        oofm = f1_score(df["y"], oof, average="macro", zero_division=0)
        f = pcf(te["y"], yp)
        print(f"{name:<7} test_acc={acc:.4f} test_macroF1={mf:.4f} oof_macroF1={oofm:.4f} | "
              + " ".join(f"{k[:3]}={f[k]:.2f}" for k in LABELS))
        if best is None or oofm > best[1]:
            best = (name, oofm, acc, mf, yp)

    print(f"\nBaseline neg+char (Batch A) = 0,7533. Batch B terbaik (OOF) = {best[0]} (acc {best[2]:.4f})")

    if args.write_official:
        from src.modeling.svm_batch_a import _write_official
        _write_official(args.tag, f"domfeat+{best[0]}", te, best[4])


if __name__ == "__main__":
    main()

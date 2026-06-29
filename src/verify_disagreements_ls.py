"""Surfacing baris untuk VERIFIKASI MANUAL: di mana model SVM tak setuju dgn label LLM.

Alur (dataset balanced 3000, label LLM confidence-tertinggi):
  1. Hitung prediksi **out-of-fold** SVM+TF-IDF (StratifiedKFold) atas SEMUA 3000 baris
     → tiap baris diprediksi model yang TIDAK melatihnya (sinyal jujur, bukan overfit).
  2. Ambil baris di mana ``pred != label LLM`` = kandidat label keliru / kasus sulit,
     diurut by margin keputusan (paling yakin di atas).
  3. Tulis CSV offline + (opsional) buat project Label Studio baru berisi baris itu,
     pra-anotasi = label LLM saat ini, dengan dugaan model ditampilkan sbg konteks.
     User tinggal cek: biarkan jika label benar, ubah hotkey 1/2/3 jika salah.

Default DRY-RUN (hanya tulis CSV + ringkasan). ``--commit`` membuat project LS.

  python -m src.verify_disagreements_ls                     # dry-run: CSV + statistik
  python -m src.verify_disagreements_ls --commit            # + buat project LS
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from configs.config import Config
from src.modeling.subset import load_subset_ids
from src.modeling.train_svm_full14k import (
    LAB,
    LABEL2ID,
    LABELS,
    TEXT,
    _connect,
    build,
    load_df,
)
from src.push_labels_to_label_studio import LSClient, _read_token, _result_payload

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "outputs" / "labeling"
CONFIG_XML = ROOT / "configs" / "label_studio_verify.xml"
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
TITLES = {
    "disagreement": "Verifikasi balanced3k — model vs LLM",
    "lowconf": "Verifikasi balanced3k — confidence rendah",
}
BATCH = 500
# load_df sudah membawa kolom `text`; meta hanya menambah kolom yg belum ada.
META_COLS = ["source_title", "like_count"]


def compute_oof(subset_csv: str, folds: int, seed: int) -> pd.DataFrame:
    """OOF SVM atas subset → DataFrame penuh + kolom model_pred, llm_label, margin.

    Tiap baris diprediksi model yang TIDAK melatihnya (StratifiedKFold) → sinyal jujur.
    """
    client = _connect()
    df = load_df(client)
    ids = load_subset_ids(subset_csv)
    df = df[df["comment_id"].astype(str).isin(ids)].reset_index(drop=True)

    # Meta untuk tampilan LS (text asli, judul, like, confidence) dari labeling_dataset.csv.
    meta = pd.read_csv(LBL / "labeling_dataset.csv",
                       usecols=["comment_id", *META_COLS, "confidence"])
    df = df.merge(meta, on="comment_id", how="left")

    X, y = df[TEXT].fillna(""), df[LAB].to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = cross_val_predict(build(), X, y, cv=skf, n_jobs=-1)
    scores = cross_val_predict(build(), X, y, cv=skf, n_jobs=-1, method="decision_function")
    # Margin = skor kelas-tertinggi − kelas-kedua (keyakinan relatif OvR).
    top2 = np.sort(scores, axis=1)[:, -2:]
    df["model_pred"] = [ID2LABEL[i] for i in oof]
    df["llm_label"] = df["label"]
    df["margin"] = np.round(top2[:, 1] - top2[:, 0], 4)
    df["setuju_llm"] = oof == y
    return df


def _build_tasks(rows: pd.DataFrame, model_version: str) -> list[dict]:
    tasks = []
    for _, r in rows.iterrows():
        data = {
            "comment_id": r["comment_id"],
            "text": r.get("text") or "",
            "source_title": r.get("source_title") or "",
            "like_count": int(r["like_count"]) if pd.notna(r.get("like_count")) else 0,
            "llm_label": r["llm_label"],
            "model_pred": r["model_pred"],
            "confidence": float(r["confidence"]) if pd.notna(r.get("confidence")) else None,
        }
        tasks.append({
            "data": data,
            # Pra-anotasi = label LLM saat ini (yang sedang diverifikasi).
            "predictions": [{
                "model_version": model_version,
                "result": _result_payload(r["llm_label"]),
            }],
        })
    return tasks


def _find_project(c: LSClient, title: str):
    r = c.request("GET", "/api/projects/?page_size=1000")
    r.raise_for_status()
    data = r.json()
    results = data.get("results", data) if isinstance(data, dict) else data
    return [p for p in results if p.get("title") == title]


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifikasi label balanced3k di Label Studio.")
    ap.add_argument("--mode", choices=["disagreement", "lowconf"], default="disagreement",
                    help="disagreement: model OOF != LLM | lowconf: confidence LLM rendah.")
    ap.add_argument("--subset", default=str(LBL / "balanced_3000.csv"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf-max", type=float, default=0.80,
                    help="(mode lowconf) ambang confidence; pilih baris <= nilai ini.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Batasi jumlah task (paling mencurigakan dulu).")
    ap.add_argument("--model-version", default="svm-oof-balanced3k")
    ap.add_argument("--commit", action="store_true", help="Benar-benar buat project LS + import.")
    args = ap.parse_args()
    title = TITLES[args.mode]

    df = compute_oof(args.subset, args.folds, args.seed)
    if args.mode == "lowconf":
        rows = df[df["confidence"] <= args.conf_max].sort_values(
            ["confidence", "margin"], ascending=[True, False]).reset_index(drop=True)
        out_csv = LBL / "verify_lowconf_balanced3k.csv"
        print(f"Confidence <= {args.conf_max}: {len(rows)} dari 3000. Distribusi kelas: "
              f"{rows['llm_label'].value_counts().to_dict()}")
    else:
        rows = df[~df["setuju_llm"]].sort_values("margin", ascending=False).reset_index(drop=True)
        out_csv = LBL / "verify_disagreements_balanced3k.csv"
        print(f"Disagreement OOF (model vs LLM): {len(rows)} dari 3000.")
        for (a, b), cnt in rows.groupby(["llm_label", "model_pred"]).size().sort_values(ascending=False).items():
            print(f"    {a:<8} → {b:<8} {cnt}")

    cols = ["comment_id", "llm_label", "model_pred", "margin", "confidence", "text"]
    rows[cols].to_csv(out_csv, index=False)
    print(f"CSV: {out_csv}")

    tasks = _build_tasks(rows if not args.limit else rows.head(args.limit), args.model_version)
    if not args.commit:
        print(f"\nDRY-RUN: {len(tasks)} task siap. Jalankan --commit untuk buat project '{title}'.")
        return

    c = LSClient(Config.label_studio.URL, _read_token(None))
    if _find_project(c, title):
        print(f"\nProject '{title}' SUDAH ADA (idempotent). Hapus dulu bila ingin re-import.")
        return
    label_config = CONFIG_XML.read_text(encoding="utf-8")
    r = c.request("POST", "/api/projects/", json={"title": title, "label_config": label_config})
    if r.status_code not in (200, 201):
        sys.exit(f"ERROR buat project: HTTP {r.status_code} {r.text[:400]}")
    pid = r.json()["id"]
    print(f"\nProject dibuat: id={pid}")

    total = 0
    for i in range(0, len(tasks), BATCH):
        batch = tasks[i:i + BATCH]
        r = c.request("POST", f"/api/projects/{pid}/import", json=batch, timeout=180)
        if r.status_code not in (200, 201):
            sys.exit(f"ERROR import @{i}: HTTP {r.status_code} {r.text[:400]}")
        total += r.json().get("task_count", len(batch))
        print(f"  batch {i // BATCH + 1}: total {total}")
    print(f"\nSELESAI: {total} task di project id={pid}.")
    print(f"Buka {Config.label_studio.URL} → '{title}' → Label All Tasks (hotkey 1/2/3/4).")


if __name__ == "__main__":
    main()

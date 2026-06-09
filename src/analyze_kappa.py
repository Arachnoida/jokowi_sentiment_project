"""Analisis kesepakatan label MANUAL (manusia) vs ``claude-llm`` pada test set v5.

Menarik anotasi dari project Label Studio "Test Set v5 — Verifikasi Manual" via
PAT, lalu membandingkannya dengan label LLM (``testset_v5_llm_reference.csv``).

Keluaran:
  - progres (berapa dari 1.411 sudah dilabeli manusia)
  - **Cohen's kappa** + raw agreement (%) manusia-vs-LLM
  - laporan per-kelas + confusion matrix (baris=manusia, kolom=LLM)
  - contoh ketidaksepakatan untuk inspeksi
  - snapshot export -> outputs/labeling/testset_v5_export.json
  - ringkasan JSON  -> outputs/reports/kappa_testset_v5.json

Aman dijalankan KAPAN SAJA selama pelabelan (hanya memakai task yang sudah
dianotasi). Kappa interpretasi (Landis & Koch): 0.2-0.4 fair, 0.4-0.6 moderate,
0.6-0.8 substantial, >0.8 almost perfect.

  python -m src.analyze_kappa
  python -m src.analyze_kappa --project 4
"""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from configs.config import Config
from src.modeling.labels import LABELS, parse_label_studio_export
from src.push_labels_to_label_studio import LSClient, _read_token

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "outputs" / "labeling"
REF_CSV = LBL / "testset_v5_llm_reference.csv"
EXPORT_SNAP = LBL / "testset_v5_export.json"
TOTAL_TARGET = 1411


def _kappa_label(k: float) -> str:
    if k < 0.0:
        return "poor (lebih buruk dari acak)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def _fetch_export(c: LSClient, pid: int) -> list:
    r = c.request("GET", f"/api/projects/{pid}/export?exportType=JSON")
    if r.status_code != 200:
        raise SystemExit(f"ERROR export project {pid}: HTTP {r.status_code} {r.text[:300]}")
    EXPORT_SNAP.write_bytes(r.content)
    return json.loads(r.content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=4, help="project id Label Studio")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "reports" / "kappa_testset_v5.json"))
    args = ap.parse_args()

    if not REF_CSV.exists():
        raise SystemExit(f"Referensi LLM tak ada: {REF_CSV}\n"
                         "Jalankan dulu: python -m src.export_testset_for_labeling")

    c = LSClient(Config.label_studio.URL, _read_token(None))
    tasks = _fetch_export(c, args.project)
    print(f"export: {len(tasks)} task ditarik dari project {args.project}")

    human = parse_label_studio_export(EXPORT_SNAP)
    # Export LS hanya memuat task yang sudah dianotasi -> kalau belum ada,
    # DataFrame kosong (tanpa kolom). Tangani sebelum mengakses kolom.
    if human.empty or "label" not in human.columns:
        print(f"\nBelum ada anotasi manusia (0/{TOTAL_TARGET}). Jalankan lagi setelah ada label.")
        return
    human = human.dropna(subset=["label"])[["comment_id", "label"]].rename(columns={"label": "human"})
    if human.empty:
        print(f"\nBelum ada anotasi manusia (0/{TOTAL_TARGET}). Jalankan lagi setelah ada label.")
        return

    ref = pd.read_csv(REF_CSV)[["comment_id", "llm_label"]]
    m = human.merge(ref, on="comment_id", how="inner")
    n = len(m)
    pct = 100.0 * n / TOTAL_TARGET

    kappa = cohen_kappa_score(m["human"], m["llm_label"], labels=LABELS)
    agree = accuracy_score(m["llm_label"], m["human"])
    cm = confusion_matrix(m["human"], m["llm_label"], labels=LABELS)

    # agreement per-kelas (basis label manusia)
    per_class = {}
    for lab in LABELS:
        sub = m[m["human"] == lab]
        per_class[lab] = {
            "n_human": int(len(sub)),
            "agree_pct": round(100.0 * (sub["human"] == sub["llm_label"]).mean(), 1) if len(sub) else None,
        }

    print(f"\n=== KESEPAKATAN MANUSIA vs LLM (test set v5) ===")
    print(f"progres        : {n}/{TOTAL_TARGET} dilabeli ({pct:.1f}%)")
    print(f"Cohen's kappa  : {kappa:.3f}  -> {_kappa_label(kappa)}")
    print(f"raw agreement  : {agree*100:.1f}%")
    print(f"\nper-kelas (basis label manusia):")
    for lab, d in per_class.items():
        print(f"  {lab:8} n={d['n_human']:<4} setuju LLM={d['agree_pct']}%")
    print(f"\nconfusion matrix (baris=MANUSIA, kolom=LLM) urut {LABELS}:")
    print(pd.DataFrame(cm, index=[f"H:{l}" for l in LABELS], columns=[f"L:{l}" for l in LABELS]).to_string())

    dis = m[m["human"] != m["llm_label"]]
    print(f"\nketidaksepakatan: {len(dis)} ({100*len(dis)/n:.1f}%). Contoh:")
    for _, r in dis.head(8).iterrows():
        print(f"  {r['comment_id'][:16]}  manusia={r['human']:8} llm={r['llm_label']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "project": args.project,
        "n_labeled": n,
        "total_target": TOTAL_TARGET,
        "progress_pct": round(pct, 1),
        "cohen_kappa": round(float(kappa), 4),
        "kappa_interpretation": _kappa_label(kappa),
        "raw_agreement": round(float(agree), 4),
        "per_class": per_class,
        "confusion_matrix": {"order": LABELS, "rows_human_cols_llm": cm.tolist()},
        "n_disagree": int(len(dis)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nringkasan -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

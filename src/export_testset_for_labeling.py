"""Ekspor TEST SET v5 (1.411 komentar) untuk PELABELAN ULANG MANUAL di Label Studio.

Kenapa: test set inilah yang dipakai semua model (SVM & IndoBERT) untuk mengukur
performa. Saat ini labelnya dari ``claude-llm`` (otomatis). Dengan melabeli ulang
test set secara MANUAL, metrik yang dilaporkan jadi terverifikasi manusia — bukan
sekadar "cocok dengan tebakan LLM".

Anotasi BUTA (penting secara metodologi): label LLM TIDAK disertakan ke task agar
anotator tak terpengaruh (anchoring bias). Label LLM disimpan terpisah untuk
menghitung agreement manusia-vs-LLM (Cohen's kappa) SETELAH anotasi selesai.

Test set direproduksi via ``split_version`` yang sama persis dengan training
(70/20/10 stratified, seed=42, urut comment_id) -> comment_id identik dengan yang
dievaluasi model.

Output:
  outputs/labeling/testset_v5_blind.json        -> import ke Label Studio (project baru)
  outputs/labeling/testset_v5_llm_reference.csv  -> comment_id + label LLM (untuk kappa; JANGAN diimpor)

Jalankan:
  python -m src.export_testset_for_labeling
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

from src.modeling.train_svm_full14k import split_version

ROOT = pathlib.Path(__file__).resolve().parents[1]
LBL = ROOT / "outputs" / "labeling"
SVM_PARQUET = ROOT / "data" / "spark_parquet" / "processed_svm.parquet"
META_CSV = LBL / "labeling_dataset.csv"

# Field yang ikut ke task LS. Hanya $text/$source_title/$like_count yang DITAMPILKAN
# (lihat configs/label_studio_sentiment.xml); comment_id dll. ikut tersimpan tapi
# tidak dirender -> berguna untuk join balik ke label LLM saat hitung kappa.
DATA_FIELDS = [
    "comment_id", "video_id", "source_title", "source_url",
    "published_at", "like_count", "text",
]


def main() -> None:
    feats = pd.read_parquet(SVM_PARQUET)               # 14.107 baris berlabel (v5 = semua)
    _, _, te = split_version(feats)                    # test set deterministik
    test_ids = set(te["comment_id"])
    print(f"test set v5: {len(test_ids)} komentar | dist:",
          te["label"].value_counts().to_dict())

    meta = pd.read_csv(META_CSV)
    sel = meta[meta["comment_id"].isin(test_ids)].copy()
    if len(sel) != len(test_ids):
        raise SystemExit(f"hanya {len(sel)}/{len(test_ids)} comment_id ketemu di {META_CSV.name}")

    # Urutan stratified round-robin: prefix berapa pun (mis. 500 pertama) tetap
    # representatif antar kelas, jadi progres harian tak bias.
    sel["__r"] = sel.groupby("label").cumcount()
    sel = sel.sort_values(["__r", "label"], kind="stable").drop(columns="__r").reset_index(drop=True)

    def _clean(v):
        return None if pd.isna(v) else (int(v) if isinstance(v, float) and v.is_integer() else v)

    tasks = [{"data": {k: _clean(row[k]) for k in DATA_FIELDS}} for _, row in sel.iterrows()]

    out_json = LBL / "testset_v5_blind.json"
    out_json.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    ref = sel[["comment_id", "label", "confidence"]].rename(columns={"label": "llm_label"})
    out_ref = LBL / "testset_v5_llm_reference.csv"
    ref.to_csv(out_ref, index=False)

    print(f"\n{len(tasks)} task buta -> {out_json.relative_to(ROOT)}")
    print(f"referensi LLM ({len(ref)}) -> {out_ref.relative_to(ROOT)}  (JANGAN diimpor)")
    print("Impor JSON itu ke project Label Studio baru (config: configs/label_studio_sentiment.xml).")


if __name__ == "__main__":
    main()

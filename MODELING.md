# Tahap Modeling — SVM vs IndoBERT

Klasifikasi sentimen komentar YouTube terhadap Jokowi ke **3 kelas**:
`Negatif (0)`, `Netral (1)`, `Positif (2)`.

Membandingkan dua paradigma pada dataset & split yang **identik**:

| Jalur | Paradigma | Input | Preprocessing | Tempat latih |
|-------|-----------|-------|---------------|--------------|
| **A — SVM + TF-IDF** | ML tradisional | `text_svm` | clean agresif + slang + buang stopword + **stemming** | Lokal (CPU) |
| **B — IndoBERT** | Deep learning (fine-tuning) | `text_bert` | cleaning minimal (morfologi terjaga) | Colab/Kaggle (GPU) |

## Desain dataset

- **Seimbang: 1.000 contoh/kelas → 3.000 total** (anotasi manual di Label Studio).
- **Split stratified 70/20/10** (train/val/test) menjaga proporsi kelas, seed `42`,
  dibuat **sekali** dan dipakai kedua model agar perbandingan adil.
- Metrik utama **macro-F1**; karena seimbang, **accuracy** juga bermakna. Selalu
  laporkan **per-kelas P/R/F1 + confusion matrix**.

> Catatan metodologi: test set seimbang mengukur performa pada *kondisi seimbang*,
> bukan distribusi dunia nyata yang timpang. Wajar untuk studi perbandingan model —
> laporkan apa adanya.

## Alur kerja

1. **Anotasi** di Label Studio (project id=1 "Sentimen Jokowi"). Hotkey `1/2/3`.
2. **Ekspor**: UI Label Studio → *Export → JSON* → simpan ke
   `outputs/labeling/label_studio_export.json`.
3. **`notebooks/05_build_dataset.ipynb`** — parse ekspor → preprocessing dua jalur →
   split → simpan ke `data/processed/splits/{train,val,test}.parquet`.
4. **`notebooks/06_train_svm.ipynb`** — latih + evaluasi SVM (lokal).
5. **`notebooks/07_indobert_finetune_colab.ipynb`** — unggah 3 parquet ke Colab,
   fine-tune IndoBERT, unduh `indobert_test_metrics.json`.
6. **Bandingkan** — taruh kedua file metrik di `outputs/reports/`:
   ```python
   import json
   from src.modeling.evaluate import compare_models
   compare_models({
       "SVM+TF-IDF": json.load(open("outputs/reports/svm_test_metrics.json")),
       "IndoBERT":   json.load(open("outputs/reports/indobert_test_metrics.json")),
   })
   ```

## Kode

```
src/modeling/
├── labels.py      # kosakata label + parser ekspor Label Studio
├── dataset.py     # preprocessing dua jalur + split stratified + simpan/muat
├── train_svm.py   # TF-IDF + LinearSVC (grid search di val)
└── evaluate.py    # metrik bersama (macro-F1, confusion matrix, compare)
```

## Dependency

- **Lokal (SVM):** `pip install -r requirements.txt` (sudah termasuk scikit-learn,
  joblib, matplotlib, pyarrow).
- **Colab (IndoBERT):** `requirements-colab.txt` (transformers, datasets, evaluate,
  accelerate). Torch sudah ada di Colab/Kaggle.

## Status saat ini

⏳ **Menunggu labeling.** Infrastruktur siap; training dimulai begitu `label_studio_export.json`
berisi cukup contoh per kelas (idealnya mendekati 1.000/kelas).

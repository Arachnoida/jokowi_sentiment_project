# Tahap Modeling — SVM vs IndoBERT

Klasifikasi sentimen komentar YouTube (isu ijazah Jokowi) ke **3 kelas**:
`Negatif (0)`, `Netral (1)`, `Positif (2)`. Polaritas di-anchor ke **narasi tuduhan**
(lihat `outputs/labeling/_RUBRIK.md`).

Membandingkan dua paradigma pada dataset & split yang **identik**:

| Jalur | Paradigma | Input (Mongo) | Preprocessing | Tempat latih |
|-------|-----------|---------------|---------------|--------------|
| **A — SVM + TF-IDF** | ML tradisional | `processed_svm.svm` | clean agresif + slang + buang stopword + **stemming**; **negasi dipertahankan** | Lokal (CPU) |
| **B — IndoBERT** | Deep learning (fine-tuning) | `processed_bert.bert` | cleaning minimal (morfologi & negasi terjaga) | Colab/Kaggle (GPU) |

## Desain dataset

- **Seimbang: 1.000/kelas → 3.000 total** (komentar `in_balanced_set=true` di
  `raw_comments`), label **LLM-assisted** (`claude-llm`).
- **Split stratified 70/20/10** (train/val/test), seed `42`, **urut `comment_id` +
  dilakukan SEBELUM preprocessing** → notebook SVM & IndoBERT menghasilkan **test/val
  identik** (perbandingan adil). Disimpan via field `split` di `processed_svm`/`processed_bert`.
- Metrik utama **macro-F1**; karena seimbang, **accuracy** juga bermakna. Selalu
  laporkan **per-kelas P/R/F1 + confusion matrix**.

> Catatan: test set seimbang mengukur performa pada *kondisi seimbang*, bukan distribusi
> dunia nyata yang timpang. Wajar untuk studi perbandingan model — laporkan apa adanya.

## Alur kerja

1. **Pelabelan** (sudah): label `claude-llm` di `raw_comments`, 3.000 ditandai balanced.
2. **`notebooks/preprocessing_svm.ipynb`** → `processed_svm` (kolom `svm` + `split`).
3. **`notebooks/preprocessing_indobert.ipynb`** → `processed_bert` (kolom `bert` + `split`).
4. **`notebooks/train_svm.ipynb`** — baca `processed_svm`, TF-IDF→LinearSVC, tuning di
   val (`PredefinedSplit`), refit train+val, evaluasi test. Lokal.
5. **`notebooks/indobert_finetune_colab.ipynb`** — baca `processed_bert`, fine-tune
   `indobert-base-p1` di Colab/GPU, evaluasi test set identik.
6. **Bandingkan** macro-F1 & per-kelas kedua model (deliverable utama).

Notebook modeling **self-contained** (tanpa `import src`) — metrik & confusion matrix
dihitung inline. Helper `compare_models()` tersedia di `src/modeling/evaluate.py`.

## Hasil

### SVM + TF-IDF (test) — ✅ selesai
- **Macro-F1 0,699** | Accuracy 0,697 | Weighted-F1 0,699.
- Per-kelas F1: **Negatif 0,761** (P 0,83), Netral 0,679, Positif 0,657.
- Kebingungan utama di seputar **Netral** (Positif↔Netral); Negatif vs Positif jarang tertukar.
- Param terbaik: `C=0.1`, unigram `(1,1)`, `min_df=2`.
- Artefak: `outputs/models/svm_tfidf.joblib` (gitignored), `outputs/reports/svm_metrics.json`,
  `svm_test_confusion.png`.

### IndoBERT (test) — ⏳ menunggu run di Colab/GPU
Target: mengungguli baseline SVM (> 0,70). Konfigurasi: 4 epoch, lr 2e-5, batch 16,
max_len 128, `metric_for_best_model="macro_f1"`, seed 42.

## Kode pendukung

```
src/modeling/
├── labels.py     # kosakata label (LABELS, LABEL2ID) + parser ekspor Label Studio
└── evaluate.py   # metrik bersama (macro-F1, confusion matrix, compare_models)
```

> Alur lama (`dataset.py` build parquet, `train_svm.py` baca parquet) diarsipkan di
> `archive/src/modeling/` — digantikan notebook berbasis Mongo.

## Dependency

- **Lokal (SVM):** `pip install -r requirements.txt` (scikit-learn, joblib, matplotlib,
  pymongo, dll). Tanpa Spark.
- **Colab (IndoBERT):** notebook meng-`%pip install` transformers + torch sendiri;
  set Runtime → GPU (T4).

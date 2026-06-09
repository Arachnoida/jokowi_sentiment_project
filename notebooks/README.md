# Notebooks — alur pipeline

Arsitektur: **MongoDB Atlas** (DB `youtube_sentiment`) sebagai sumber & tujuan data —
semua dokumen JSON di Mongo, tanpa file CSV/parquet sebagai dataset.

> **Penting (migrasi 2026-06-09):** tahap **preprocessing** dan **training SVM**
> sudah dimigrasi dari notebook ke **skrip `.py` / PySpark** (lihat `../MODELING.md`
> bagian "Jalur PySpark"). Notebook yang digantikan dipindah ke `../archive/notebooks/`.
> Notebook yang TERSISA di sini hanya yang belum/ tak dimigrasi: data collection,
> eksperimen, dan fine-tune berbasis Colab/GPU.

## Struktur folder

```
notebooks/
├── 1_data_collection/
│   ├── ingestion.ipynb               YouTube Data API → raw_comments
│   └── export_labeling.ipynb         raw_comments → ekspor ke Label Studio
├── 3_modeling/
│   ├── improve_svm.ipynb             eksperimen SVM (ensemble word+char, 5-fold CV)
│   ├── indobert_finetune_colab.ipynb processed_bert → IndoBERT + metrik (Colab/GPU)
│   └── indobertweet_improve_colab.ipynb  IndoBERTweet + weighted loss (Colab/GPU)
└── utils/
    ├── config.ipynb                  cek koneksi & konfigurasi
    ├── database_maintenance.ipynb    operasi pemeliharaan koleksi
    └── reset_database.ipynb          reset/kosongkan koleksi
```

## Tahap yang sudah jadi `.py` (bukan notebook lagi)

| Tahap | Sebelumnya (notebook) | Sekarang (`.py` / Spark) |
|-------|------------------------|--------------------------|
| Preprocessing SVM | `preprocessing_svm.ipynb` | `src/spark/preprocess_spark.py` (+ `src/modeling/_backfill_processed_svm.py` utk tulis Mongo) |
| Preprocessing BERT | `preprocessing_indobert.ipynb` | `src/spark/preprocess_spark.py` (+ `_backfill_processed_bert.py`) |
| EDA preprocessing | (sel EDA di notebook) | `src/spark/eda_spark.py` |
| Training SVM | `train_svm.ipynb` | `src/modeling/train_svm_full14k.py` (sklearn) + `src/spark/train_svm_spark.py` (Spark MLlib) |

Cara jalan jalur Spark: lihat `../MODELING.md` → "Jalur PySpark (Big Data)".

## Alur notebook yang tersisa

| Tahap | Notebook | Baca → Tulis |
|-------|----------|--------------|
| 1. Data collection | `1_data_collection/ingestion.ipynb` | YouTube API → `raw_comments` |
| 1. Labeling (bridge) | `1_data_collection/export_labeling.ipynb` | `raw_comments` → Label Studio |
| 3. Modeling (Colab/GPU) | `3_modeling/indobert_finetune_colab.ipynb` | `processed_bert` → model + metrik |
| 3. Eksperimen (Colab) | `3_modeling/indobertweet_improve_colab.ipynb` | `processed_bert` → IndoBERTweet |
| 3. Eksperimen (lokal) | `3_modeling/improve_svm.ipynb` | `processed_svm` → ensemble/CV |

IndoBERT tetap notebook karena fine-tune transformer butuh **GPU (Colab)** dan tak punya
padanan Spark MLlib (lihat catatan di `../MODELING.md`).

## Koneksi

Notebook membaca `MONGO_URI` dari `.env` (lokal) atau `getpass` (Colab); lihat
`../.env.example`. Notebook bersifat **self-contained** (tanpa `import src`) agar bisa
dijalankan lokal maupun di Colab.

## Arsip

Notebook alur lama (preprocessing, train_svm, serta percobaan Spark/parquet terdahulu)
ada di `../archive/notebooks/` — referensi sejarah, **bukan pipeline aktif**.

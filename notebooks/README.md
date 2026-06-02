# Notebooks — alur pipeline

Arsitektur: **MongoDB Atlas** (DB `youtube_sentiment`) sebagai sumber & tujuan data —
semua dokumen JSON di Mongo, tanpa file CSV/parquet sebagai dataset. Notebook
dikelompokkan per tahap dalam folder bernomor; **nama file** tetap tanpa nomor.

## Struktur folder

```
notebooks/
├── 1_data_collection/
│   ├── ingestion.ipynb              YouTube Data API → raw_comments
│   └── export_labeling.ipynb        raw_comments → ekspor ke Label Studio
├── 2_preprocessing/
│   ├── preprocessing_svm.ipynb      raw_comments → processed_svm  (kolom `svm`)
│   └── preprocessing_indobert.ipynb raw_comments → processed_bert (kolom `bert`)
├── 3_modeling/
│   ├── train_svm.ipynb              processed_svm  → model SVM + metrik (lokal)
│   └── indobert_finetune_colab.ipynb processed_bert → IndoBERT + metrik (Colab/GPU)
└── utils/
    ├── config.ipynb                 cek koneksi & konfigurasi
    ├── database_maintenance.ipynb   operasi pemeliharaan koleksi
    └── reset_database.ipynb         reset/kosongkan koleksi
```

## Urutan alur

| Tahap | Notebook | Baca → Tulis |
|-------|----------|--------------|
| 1. Data collection | `1_data_collection/ingestion.ipynb` | YouTube API → `raw_comments` |
| 1. Labeling (bridge) | `1_data_collection/export_labeling.ipynb` | `raw_comments` → Label Studio |
| 2. Preprocessing | `2_preprocessing/preprocessing_svm.ipynb` | `raw_comments` → `processed_svm` |
| 2. Preprocessing | `2_preprocessing/preprocessing_indobert.ipynb` | `raw_comments` → `processed_bert` |
| 3. Modeling (lokal) | `3_modeling/train_svm.ipynb` | `processed_svm` → model + metrik |
| 3. Modeling (Colab/GPU) | `3_modeling/indobert_finetune_colab.ipynb` | `processed_bert` → model + metrik |

Preprocessing SVM & IndoBERT memakai **split identik** (urut `comment_id` + `seed=42`,
split sebelum preprocessing) → test/val sama persis untuk perbandingan adil.

## Koneksi

Notebook membaca `MONGO_URI` dari `.env` (lokal) atau `getpass` (Colab); lihat
`../.env.example`. Notebook preprocessing & modeling bersifat **self-contained** (tanpa
`import src`) sehingga bisa dijalankan lokal maupun di Colab.

## Arsip

Notebook/skrip alur lama (Spark, parquet) ada di `../archive/` — disimpan untuk
referensi, **bukan bagian pipeline aktif**.

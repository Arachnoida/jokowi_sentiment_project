# Notebooks — alur pipeline

Arsitektur: **MongoDB Atlas** (DB `youtube_sentiment`) sebagai sumber & tujuan data.
Tidak pakai file CSV/parquet sebagai dataset; semua data dokumen JSON di Mongo.
Notebook tanpa nomor — urutan alur dijelaskan di bawah.

## Alur (urut dijalankan)

| # | Notebook | Tahap | Baca | Tulis |
|---|----------|-------|------|-------|
| 1 | `ingestion.ipynb` | Data collection | YouTube Data API | `raw_comments` |
| 2 | `export_labeling.ipynb` | Labeling (bridge) | `raw_comments` | ekspor ke Label Studio |
| 3 | `preprocessing_svm.ipynb` | Preprocessing | `raw_comments` (`in_balanced_set`) | `processed_svm` |
| 3 | `preprocessing_indobert.ipynb` | Preprocessing | `raw_comments` (`in_balanced_set`) | `processed_bert` |
| 4 | `train_svm.ipynb` | Modeling (lokal) | `processed_svm` | model + metrik |
| 4 | `indobert_finetune_colab.ipynb` | Modeling (Colab/GPU) | `processed_bert` | model + metrik |

Preprocessing SVM & IndoBERT memakai **split identik** (seed=42, urut `comment_id`,
split sebelum preprocessing) → test/val sama persis untuk perbandingan adil.

## Utilitas (situasional)
- `config.ipynb` — cek koneksi & konfigurasi.
- `database_maintenance.ipynb` — operasi pemeliharaan koleksi.
- `reset_database.ipynb` — reset/kosongkan koleksi.

## Koneksi
Notebook membaca `MONGO_URI` dari `.env` (lokal) atau `getpass` (Colab). Lihat
`.env.example`. Preprocessing & modeling bersifat **self-contained** (tanpa `import src`).

## Arsip
Notebook/skrip alur lama (Spark, parquet) ada di `../archive/` — disimpan untuk
referensi, **bukan bagian pipeline aktif**.

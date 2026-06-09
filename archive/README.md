# Archive — artefak alur lama (tidak dipakai)

Berkas di sini **bukan bagian pipeline aktif**. Disimpan untuk referensi/sejarah saja.
Lihat `../notebooks/README.md` (notebook tersisa) & `../MODELING.md` (jalur `.py`/Spark).

## Notebook yang digantikan skrip `.py`/PySpark (diarsipkan 2026-06-09)

Tahap preprocessing & training SVM dimigrasi ke `src/spark/` + `src/modeling/`:
- `notebooks/preprocessing_svm.ipynb` & `..._purecode.ipynb` — preprocessing jalur SVM
  (tulis `processed_svm`). Digantikan `src/spark/preprocess_spark.py` +
  `src/modeling/_backfill_processed_svm.py`.
- `notebooks/preprocessing_indobert.ipynb` & `..._purecode.ipynb` — preprocessing jalur
  BERT (tulis `processed_bert`). Digantikan `src/spark/preprocess_spark.py` +
  `src/modeling/_backfill_processed_bert.py`.
- `notebooks/train_svm.ipynb` — training SVM lintas versi. Digantikan
  `src/modeling/train_svm_full14k.py` (sklearn) & `src/spark/train_svm_spark.py` (Spark MLlib).
- `notebooks/indobert_finetune_colab_variant.ipynb` — varian lama notebook IndoBERT
  (untracked); notebook IndoBERT AKTIF tetap di `../notebooks/3_modeling/`.

> **Catatan penting — JANGAN rancu dua "Spark":** percobaan Spark LAMA di bawah ini
> (`preprocessing_spark.ipynb`, `export_labeling_spark.ipynb`, `src/preprocess_spark.py`)
> sudah usang. Jalur Spark **AKTIF** sekarang ada di `../src/spark/` (alasan: syarat
> akademik Big Data), bukan berkas arsip ini.

## Kenapa diarsipkan (percobaan terdahulu)

**Alur Spark LAMA (ditinggalkan)** — percobaan awal saat dataset masih 3.000 balanced:
- `notebooks/preprocessing_spark.ipynb` — preprocessing korpus via PySpark.
- `notebooks/export_labeling_spark.ipynb` — ekspor untuk labeling (versi Spark).
- `src/preprocess_spark.py` — UDF/pipeline Spark.

**Alur parquet (digantikan Mongo)** — split disimpan sebagai parquet lokal:
- `notebooks/build_dataset_parquet.ipynb` (eks `05_build_dataset`) — parse export
  Label Studio → split parquet. Digantikan `preprocessing_svm/indobert.ipynb` yang
  baca/tulis langsung ke Mongo (`processed_svm`/`processed_bert`).
- `src/modeling/dataset.py` — build_modeling_frame / stratified_split / build_and_save
  (alur parquet). Logikanya kini inline di notebook preprocessing.
- `src/modeling/train_svm.py` — training SVM yang baca parquet via `load_splits`.
  Digantikan `notebooks/train_svm.ipynb` yang baca `processed_svm` dari Mongo.

**Demo:**
- `notebooks/colab_preprocessing_demo.ipynb` — demo trace transformasi; kini redundan
  karena sel trace sudah ada di notebook preprocessing aktif.

> Catatan: import `from src...` di berkas arsip mengacu ke lokasi lama dan tidak
> dijamin jalan. Jangan dipakai untuk produksi.

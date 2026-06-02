# Archive — artefak alur lama (tidak dipakai)

Berkas di sini **bukan bagian pipeline aktif**. Disimpan untuk referensi/sejarah saja.
Pipeline sekarang berbasis MongoDB Atlas + pandas; lihat `../notebooks/README.md`.

## Kenapa diarsipkan

**Alur Spark (ditinggalkan)** — dataset kecil (3.000 balanced), Spark berlebihan:
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

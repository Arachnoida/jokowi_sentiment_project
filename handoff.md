# Handoff — Re-labeling Full 14k dengan Rubrik Final

Status per **2026-06-24**. Dataset "Sentimen Jokowi" (14.107 komentar) di-label ulang
oleh LLM (Claude Code) mengikuti `outputs/labeling/_RUBRIK.md` versi final yang sudah
disepakati (text-only stance, attack-on-person → Netral, dst.).

## 1. Ringkasan hasil

Distribusi label **berubah drastis** (52,5% label berganti) — efek rubrik konservatif,
dan kini **selaras dengan label manusia** (manual tim 82% Netral, Negatif 4,3%):

| Kelas | LAMA | BARU |
|---|---|---|
| Netral | 3.060 (22%) | **9.889 (70,1%)** |
| Positif | 5.516 (39%) | 3.005 (21,3%) |
| Negatif | 5.531 (39%) | **1.213 (8,6%)** |

Transisi utama: Negatif→Netral 4.364, Positif→Netral 2.597.

## 2. Yang SUDAH dikerjakan (otomatis)

- [x] Re-label 14.107 via workflow (142 batch × 100, model Sonnet, text-only).
- [x] Push label baru → MongoDB `raw_comments` (matched 14.107, modified 14.103).
- [x] Regen `processed_svm` & `processed_bert` di Mongo (Spark) — label baru.
- [x] Regen snapshot Parquet (`data/spark_parquet/*`) untuk jalur Spark.
- [x] **SVM sklearn** dilatih ulang → `outputs/reports/svm_full14k_metrics.json`.
- [x] **SVM Spark MLlib** dilatih ulang → `outputs/reports/svm_spark_metrics.json`.
- [~] **Label Studio id=1**: label baru di-push sebagai *predictions*
      (`model_version=claude-llm-rubrik-final-20260624`); anotasi lama tidak dihapus.

### Metrik SVM (test set 1.411; Neg 121 / Net 989 / Pos 301)

| Model | macro-F1 | Akurasi | F1 Neg | F1 Net | F1 Pos |
|---|---|---|---|---|---|
| SVM sklearn | **0,686** | 0,814 | 0,476 | 0,885 | 0,697 |
| SVM Spark MLlib | 0,635 | 0,768 | 0,41 | 0,85 | 0,65 |

> Akurasi naik (Netral dominan & mudah), tapi **Negatif anjlok** (imbalance: hanya 121
> sampel uji). macro-F1 sedikit naik dari 0,669 karena Netral menyumbang besar.

## 3. Yang HARUS kamu kerjakan — IndoBERT di Colab

IndoBERT belum dilatih ulang (kamu jalankan di Colab GPU). Notebook Colab membaca
`processed_bert` **langsung dari MongoDB Atlas** (sudah ter-update), split kanonik
(urut `comment_id`, seed=42, 70/20/10) **identik** dengan SVM.

1. Jalankan ulang notebook IndoBERT seperti biasa (pastikan `MONGO_URI` ter-set;
   IP allowlist Atlas sudah terbuka untuk Colab dari sesi sebelumnya).
   Setara lokal: `python -m src.modeling.train_indobert`.
2. Simpan hasil sebagai **`outputs/reports/indobert_metrics.json`** dengan struktur:
   `{"model":"IndoBERT","test":{"accuracy":..,"macro_f1":..,"per_class":{...},"confusion_matrix":[...],"labels":["Negatif","Netral","Positif"]}}`
3. Kabari/commit file itu → langkah lanjutan otomatis: regen tabel & chart perbandingan
   3 model + update laporan PDF.

> IndoBERT versi label LAMA sudah diarsipkan ke
> `outputs/reports/indobert_metrics_OLDLABELS_20260624.json` (jangan dipakai untuk
> perbandingan label baru).

## 4. Backup & cara revert

Label lama disimpan sebelum ditimpa (push ulang untuk revert):

- `outputs/labeling/labels_backup_20260624.csv` (comment_id, label lama, dst.)
- `outputs/labeling/labeling_dataset.backup_20260624.csv` (CSV lengkap lama)

Revert ke label lama:
```bash
cp outputs/labeling/labeling_dataset.backup_20260624.csv outputs/labeling/labeling_dataset.csv
.venv/bin/python -m src.push_labels_to_mongo --no-dry-run
.venv/bin/python -m src.spark.regenerate_processed_mongo   # regen processed_*
```

## 5. Perintah referensi (urutan pipeline)

```bash
# (label sudah ada di outputs/labeling/labeling_dataset.csv)
.venv/bin/python -m src.push_labels_to_mongo --no-dry-run            # -> Mongo raw_comments
.venv/bin/python -m src.spark.regenerate_processed_mongo            # -> processed_svm/bert (Mongo)
.venv/bin/python -m src.spark.export_mongo                          # -> data/spark_parquet/*
.venv/bin/python -m src.spark.preprocess_spark                     # -> features_spark.parquet
.venv/bin/python -m src.modeling.train_svm_full14k                  # -> svm_full14k_metrics.json
.venv/bin/python -m src.spark.train_svm_spark                       # -> svm_spark_metrics.json
# IndoBERT: Colab / python -m src.modeling.train_indobert           # -> indobert_metrics.json
```

> Catatan: jalur Spark membaca **Parquet** (`data/spark_parquet/`), BUKAN Mongo langsung.
> Setelah label berubah, WAJIB `export_mongo` + `preprocess_spark` sebelum `train_svm_spark`,
> kalau tidak Spark memakai fitur lama (pernah terjadi: hasil stale 0,607).

## 6. Sisa pekerjaan

- [ ] IndoBERT Colab (kamu) → `indobert_metrics.json`.
- [ ] Regen perbandingan 3 model + update laporan PDF (setelah IndoBERT).
      **Metrik utama = AKURASI** (keputusan user 2026-06-24; macro-F1 dibuang dari
      tabel/chart karena data timpang 70% Netral → akurasi yang dipakai).
- [ ] **(DITUNDA) Fix over-stemming Sastrawi** `setuju→tuju` (juga `sependapat→dapat`,
      `sepaham`). Cara: tambah kata terlindungi ke kamus stemmer di `src/spark/udf.py`
      via `stemmer.delegatedStemmer.dictionary.add(<kata>)`. Lalu **re-run** pipeline
      preprocessing→SVM (regenerate_processed_mongo → export_mongo → preprocess_spark →
      train_svm_full14k → train_svm_spark; ~12–20 mnt). Dampak KECIL: hanya 0,8% baris
      (112/14107), tidak mengubah label, IndoBERT tak terpengaruh (kolom `bert` tanpa stem).
      Lihat §5 untuk urutan perintah.
- [x] LS predictions id=1: ter-push ~59% (terpotong timeout 90 mnt; kosmetik, Mongo = truth).
- [x] Commit & push (commit `24791e6`, `8a56e67`, `116a887` di main).

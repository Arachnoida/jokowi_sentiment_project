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

- [x] **(SELESAI 2026-06-24) IndoBERT Colab** → `outputs/reports/indobert_metrics.json`
      (label baru, test set kanonik 1.411 identik SVM). Akurasi **0,8207**, macro-F1 0,698,
      F1 Neg 0,480 / Net 0,886 / Pos 0,730.
- [x] **(SELESAI 2026-06-24) Regen perbandingan 3 model** via `src/modeling/compare_models.py`
      → `outputs/reports/model_comparison_full14k.csv` + `model_comparison_accuracy.png`.
      **Metrik utama = AKURASI** (keputusan user; macro-F1 tetap dicatat di CSV tapi chart
      pakai akurasi). Hasil akurasi: **IndoBERT 0,8207** > SVM sklearn 0,8115 > SVM Spark
      0,7725 (IndoBERT menang tipis; SVM sklearn justru unggul F1 Negatif 0,502). Chart lama
      `svm_vs_indobert_full14k.png` (2-model macro-F1, IndoBERT label LAMA) DIHAPUS — stale.
- [ ] **Update laporan** (`laporan/BD II - 2305551076.docx/.pdf`, manual) dgn tabel+chart
      perbandingan 3 model di atas. Dokumen di-maintain manual → tempel artefak dari
      `outputs/reports/model_comparison_*`.
- [x] **(SELESAI 2026-06-24) Fix over-stemming Sastrawi** `setuju→tuju`,
      `sependapat→dapat`. Kata `setuju/sependapat/sepaham/sepakat` didaftarkan sbg root
      word ke kamus stemmer di `src/spark/udf.py` (+`_backfill_processed_svm.py`), WAJIB
      sebelum stem pertama karena `CachedStemmer` meng-cache. Pipeline penuh sudah di-re-run
      (regenerate_processed_mongo → export_mongo → preprocess_spark → train_svm_full14k →
      train_svm_spark; ~17 mnt). Dampak kecil tapi positif: **F1 Negatif naik** (sklearn
      0,476→0,502 +2,6pp; Spark 0,41→0,443 +3,3pp), akurasi ~datar (sklearn 0,814→0,8115;
      Spark 0,768→0,7725), macro-F1 naik di keduanya. IndoBERT tak terpengaruh (kolom `bert`
      tanpa stem) → tak perlu re-train ulang gara-gara ini.
      Catatan: `try_stem_sastrawi` di `text_normalizer.py` punya bug serupa tapi DEAD CODE
      (tak dipakai) → sengaja dibiarkan.
- [x] LS predictions id=1: ter-push ~59% (terpotong timeout 90 mnt; kosmetik, Mongo = truth).
- [x] Commit & push (commit `24791e6`, `8a56e67`, `116a887` di main).

## 7. Roadmap pelabelan & training (target bertahap)

Strategi: **mulai dari gold-set kecil yang BALANCED**, jangan tunggu pelabelan penuh.
Tujuannya punya baseline berbasis **label MANUSIA** (bukan LLM) secepatnya.

### Milestone 1 (PRIORITAS SEKARANG) — gold balanced ~300/kelas (~900 total)
- Labeling manual sampai **±300 per kelas** (Positif/Negatif/Netral).
  - Sudah terkumpul (project id=6, 230 task): Netral 188, Positif 32, Negatif 10.
  - Sisa kira-kira: **Negatif +290, Positif +268, Netral +112**.
  - Sumber: fokus **boost id=8 lalu id=7** untuk Pos/Neg; Netral mudah (dari id=6).
  - Negatif = bottleneck (kandidat total ~1043 di id=7+id=8) tapi 300 sangat tercapai.
- Lalu **train SVM + IndoBERT pada gold balanced ~900 ini** = eksperimen pertama
  berbasis label manusia. Catatan: data kecil → hasil kasar/variansi tinggi, tapi ini
  baseline "gold" yang jujur (bandingkan dengan model yang dilatih label LLM full 14k).

### Milestone 2 — perbesar ke balanced 1000/1000/1000
- Lanjut labeling sampai 1000/kelas (kalau Negatif cukup; **plafon Negatif = 1213**).
  Kalau mentok, turunkan target Neg atau scrape data baru.

### Milestone 3 — validasi & pelaporan
- Hitung **Cohen's kappa manusia-vs-LLM** (kualitas korpus).
- Evaluasi model (LLM-trained full 14k) terhadap **gold manusia** sebagai test jujur.
- Regen perbandingan 3 model (akurasi) + update laporan PDF.

## 8. Eksperimen BALANCED 3000 (2026-06-29)

Keputusan user: fokus dataset **balanced 1000/kelas (3000 total)**, di-*sample* dari
label LLM **confidence tertinggi** per kelas (bukan label manusia — masih jalur LLM).
Tujuan: ukur performa per-kelas yang jujur (test balanced 100/100/100 menghilangkan
bias "Netral dominan").

**Tooling baru (parametrik, default full14k tak berubah):**
- `src/modeling/build_balanced_subset.py` → `outputs/labeling/balanced_3000.csv`
  (top-1000/kelas by confidence; Neg conf 0.75–0.95, Net 0.90–1.0, Pos 0.85–0.99).
- `src/modeling/subset.py` — helper `load_subset_ids`.
- Trainer dapat `--subset <csv> --tag <tag>`: `train_svm_full14k`, `train_svm_spark`,
  `train_indobert`, dan `compare_models --tag`. Subset = FILTER baris setelah load
  (fitur sudah ada di processed_*/parquet → tak perlu regen). Artefak ber-suffix tag.

**Reproduksi:**
```bash
python -m src.modeling.build_balanced_subset                                    # -> balanced_3000.csv
python -m src.modeling.train_svm_full14k --subset outputs/labeling/balanced_3000.csv --tag balanced3k
python -m src.spark.train_svm_spark     --subset outputs/labeling/balanced_3000.csv --tag balanced3k
# IndoBERT (Colab, lihat outputs/reports/COLAB_indobert_balanced3k.md):
#   python -m src.modeling.train_indobert --subset outputs/labeling/balanced_3000.csv --tag balanced3k
python -m src.modeling.compare_models --tag balanced3k                           # setelah 3 JSON ada
```

**Update 2026-06-30 — filter READABLE:** `build_balanced_subset.py` kini menyaring
komentar tak-readable (default on; `is_readable`, tingkat SEDANG: buang emoji/simbol
murni & yg didominasi emoji, simpan kata tunggal bermakna). Pool 14107→14005 (buang 102).
`balanced_3000.csv` di-rebuild: 62 baris tak-readable ditukar baris readable (top-up
confidence), 2938 tetap, semua readable, 1000/kelas. Backup lama: `balanced_3000.prereadable.bak`.
`--no-readable` utk reproduksi versi lama. SVM/IndoBERT WAJIB re-train pada dataset ini.

**Hasil (test 300 = 100/kelas):**

| Model | Akurasi | macro-F1 | F1 Neg | F1 Net | F1 Pos |
|---|---|---|---|---|---|
| **IndoBERT** ★ | **0,8533** | 0,8525 | 0,79 | 0,91 | 0,86 |
| SVM sklearn | 0,8333 | 0,8334 | 0,78 | 0,87 | 0,85 |
| SVM Spark MLlib | 0,7933 | 0,7894 | 0,72 | 0,84 | 0,81 |

> F1 Negatif melonjak vs full14k (sklearn 0,50 → **0,78**) karena kelas seimbang.
> Akurasi balanced TIDAK comparable dgn akurasi full14k (0,81) yang ter-*inflate* Netral.
> Peringkat: **IndoBERT > SVM sklearn > SVM Spark**. Selisih IndoBERT vs SVM sklearn
> MELEBAR di data balanced (+2pp; full14k cuma +0,9pp) — IndoBERT unggul di Netral (F1 0,91).
> IndoBERT dijalankan di Colab GPU (T4) via notebook clone, ~4 mnt.

- [x] **(SELESAI 2026-06-30) IndoBERT balanced3k** (Colab T4) → `indobert_balanced3k_metrics.json`
      (acc 0,8533). `compare_models --tag balanced3k` → `model_comparison_balanced3k.{csv,png}`.
- [~] **Verifikasi disagreement** (LS project id=9, 551 task): user review manual mana
      label LLM yg salah → relabel → re-train. Lihat `verify_disagreements_balanced3k.csv`.
      Opsi **"🗑️ Tidak terbaca"** (hotkey 4) ditambahkan ke project utk komentar gibberish.
      Setelah verifikasi: `python -m src.rebuild_balanced_from_verification --commit` →
      tarik anotasi, buang yg "tidak terbaca", terapkan koreksi (manusia=gold), rebuild
      balanced 1000/kelas (prioritas gold lalu confidence; top-up otomatis) → re-train.
      `--patch-config` utk update label_config project yg sudah live.
- [ ] **Update laporan PDF** dgn tabel + chart 3-model balanced3k.

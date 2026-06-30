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

**Angka FINAL (dataset READABLE, test 300=100/kelas):**

| Model | Akurasi | macro-F1 | F1 Neg | F1 Net | F1 Pos |
|---|---|---|---|---|---|
| **IndoBERT** ★ | **0,8467** | 0,8469 | 0,79 | 0,885 | 0,865 |
| SVM sklearn | 0,8333 | 0,8331 | 0,79 | 0,854 | 0,857 |
| SVM Spark MLlib | 0,7667 | 0,7669 | 0,71 | 0,802 | 0,792 |

> (Angka pra-readable: IndoBERT 0,8533 / SVM 0,8333 / Spark 0,7933 — tak comparable,
> dataset beda.) **Fix over-stem 2026-06-30:** `diseting→ting`/`mentri→tri` ditambah ke
> `_PROTECTED_ROOTS` (udf.py + _backfill_processed_svm.py). Sebagian (`dibantah→ban`,
> `seandainya→anda`) TAK bisa diperbaiki via kamus = keterbatasan rule Sastrawi (dokumentasikan
> di laporan). Impak kecil; fix berlaku saat regen processed_svm berikutnya (di-batch dgn
> re-train pasca-verifikasi — hindari regen Mongo redundan). SVM di tabel ini masih fitur lama.

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

## 10. Re-label LLM Opus + rubrik domain-aware (2026-07-01)

**Keputusan user:** fokus dataset balanced 3000 (14k ditinggalkan), perbaiki dataset utk
naikkan akurasi. Dijalankan dua pass re-label dgn **Claude Opus** (bukan Sonnet pass-1),
blind, mengikuti `_RUBRIK.md`.

**Temuan kunci — leher botol Negatif:** readable Negatif cuma 1.213; memaksa 1.000/kelas
menyeret Negatif ke conf 0.75 (kotor). Re-label mengungkap Sonnet pass-1 **menggelembungkan
Negatif**.

**Pass-1 (1.056 suspect, rubrik lama):** 316 berubah (29,9%) — dominan **288 Negatif→Netral**
(serang-orang/tuntut-penjara tanpa basis klaim) + 13 Neg→Pos. Pool Negatif bersih turun ke
~905. SVM balanced (900/kelas, dedup): **0,84→0,767**. Penurunan = **pembongkaran inflasi**:
0,84 lama mengukur label yg salah (pintasan leksikal Negatif=bahasa-serangan).

**Rubrik REVISI 2026-06-30 (domain-aware, di `_RUBRIK.md`):** user pilih "Penuh + cerminan".
Karena aktor korpus tetap & dikenal: **menyerang/menuntut-hukuman PENUDUH** (Roy Suryo/Rismon/
Tifa/Rizal/RRT) walau tanpa kata isu → **Negatif**; menyerang/menuntut Jokowi/pembela →
**Positif**; pujian/doa/sapaan telanjang + arah ambigu → tetap Netral.

**Pass-2b (3.240 Netral-menyebut-aktor, rubrik baru):** 2.221 berubah (68,5%) → **1.936 Negatif**
+ 285 Positif. Pool Negatif melonjak 913→2.828 (2.396 Opus-verified) → balanced kembali ke
**1.000/kelas**, Negatif & Netral 100% Opus-verified, Positif 455 verified (sisa pass-1 claim-based).

**Dampak SVM (test 100/kelas):** 0,84(inflasi) → 0,767(pass-1 bersih, 900) → **0,730**(domain-aware, 1000).
Confusion: Negatif terlemah (R=0,67), error tersebar Neg→Net 18 / Neg→Pos 15 / Net→Pos 19.
**Sebab:** label makin VALID secara stance (siapa-diserang + serang-vs-puji) tapi makin SULIT utk
bag-of-words → SVM TF-IDF mentok ~0,73. **Ini justru titik di mana IndoBERT (kontekstual)
seharusnya unggul** — gap diperkirakan MELEBAR.

**Tooling baru (reusable):**
- `src/modeling/apply_relabel_rebuild.py` — terapkan koreksi + dedup + rebuild (dipakai pass-1).
- `src/modeling/push_relabel_to_mongo.py` — push koreksi → master CSV + Mongo `raw_comments`
  & `processed_bert` (label hidup di KEDUA koleksi: SVM baca raw, IndoBERT baca processed_bert).
- `src/modeling/rebuild_balanced_from_master.py` — rebuild balanced dari master (verified-first,
  dedup). `--per-class 1000`.
- Catatan durable koreksi: `outputs/labeling/relabel_pass2_opus_20260630.csv` (pass-1),
  `relabel_pass2b_domainaware_20260630.csv` (pass-2b). Backup balanced: `.prerelabel.bak`/`.prerebuild.bak`.

**Batch A SVM (model-side, TANPA regen) — SELESAI 2026-07-01.** Harness baru
`src/modeling/svm_batch_a.py` (eval test split kanonik + OOF 5-fold). Hasil balanced3k
(test 100/kelas): base word(1,2)=0,713 → **+char n-gram(3-5)=0,743** → **+per-class
threshold (bias decision_function)=0,767**. Negation-merge sendiri kecil (+0,7pp; char sudah
serap). **SVM pulih 0,730→0,767** murni model-side. Eksperimen → `outputs/reports/svm_batch_a_balanced3k.csv`.
char n-gram = lever utama; ceiling bag-of-words ~0,767.

**KEPUTUSAN PENDING (user):** promosi char+threshold jadi SVM resmi → **TUNGGU IndoBERT dulu**,
baru tentukan konfigurasi final SEMUA model bersama. Trainer kanonik `train_svm_full14k.py`
BELUM diubah (masih word-only, svm_balanced3k_metrics.json=0,730).

**WAJIB BERIKUTNYA (user, Colab) — ANGKA PENENTU:** jalankan **IndoBERT balanced3k** pada label
baru. Notebook **self-contained** (gaya indobert_finetune_colab_variant): **`indobert_balanced3k_colab.ipynb`**
(root repo) — TANPA upload CSV / clone, baca `processed_bert` via flag **`in_balanced3k=True`**,
cuma butuh MONGO_URI. Flag diset oleh `python -m src.modeling.push_subset_ids` (sudah dijalankan;
3000 doc ter-flag). Output `indobert_balanced3k_metrics.json` → kirim ke Claude. Lalu putuskan
config final + `compare_models --tag balanced3k`. Hipotesis: IndoBERT (kontekstual) unggul jauh di
task stance ini → dataset domain-aware = win.

## 11. Hasil final domain-aware + IndoBERTweet (2026-07-01)

**IndoBERT balanced3k (label domain-aware, Colab) SELESAI:** acc **0,7733**, macro-F1 0,774
(F1 Neg 0,785 / Net 0,773 / Pos 0,764). Notebook dipakai: `notebooks/3_modeling/indobert_finetune_colab.ipynb`
(clone via PAT + `train_indobert --tag balanced3k --subset`).

**HIPOTESIS GUGUR:** IndoBERT TIDAK unggul jauh. Perbandingan final (2 model, Spark di-drop):

| Model | Akurasi | macro-F1 | F1 Neg | F1 Net | F1 Pos |
|---|---|---|---|---|---|
| SVM (char+thr, Batch A) | 0,7667 | 0,765 | 0,761 | 0,742 | **0,793** |
| **IndoBERT** ★ | **0,7733** | 0,774 | **0,785** | **0,773** | 0,764 |

IndoBERT menang **cuma +0,66pp**. Sebab: aturan "serang penuduh→Negatif" butuh world-knowledge
(Roy Suryo=penuduh) yang TAK ada di teks → bahkan IndoBERT (2100 train) cuma belajar sebagian →
**kedua model mentok ~0,77**. **0,84 lama = fatamorgana** (label salah). Plafon jujur ~0,77 utk
label valid, apa pun filosofi labelnya.

**KEPUTUSAN (user): keep domain-aware + coba IndoBERTweet.**
- **SVM Batch A DIPROMOSIKAN jadi resmi.** `svm_balanced3k_metrics.json` kini = char+thr (0,7667),
  diproduksi `src/modeling/svm_batch_a.py --write-official` (BUKAN train_svm_full14k yg masih
  word-only/baseline; trainer kanonik TIDAK diubah agar full14k aman). Varian: FeatureUnion
  word(1,2)+char_wb(3,5) + LinearSVC(C=0.5,balanced) + bias per-kelas [0.3,0,0.4] (tuned val).
- `compare_models` kini **skip model yg file metriknya tak ada** (Spark dilewati). Output
  `model_comparison_balanced3k.{csv,png}` = 2 model.

**Track C IndoBERTweet — SELESAI & MENANG.** `indolem/indobertweet-base-uncased` + weighted loss
(6 epoch) via `notebooks/3_modeling/indobertweet_balanced3k_colab.ipynb`. **acc 0,79** — TEMBUS
plafon ~0,77. Kunci: Netral F1 **0,820** (model domain-medsos paham bahasa alay/medsos).

**TABEL FINAL balanced3k (test 100/kelas, label domain-aware):**

| Model | Akurasi | macro-F1 | F1 Neg | F1 Net | F1 Pos |
|---|---|---|---|---|---|
| SVM sklearn (char+thr) | 0,7667 | 0,765 | 0,761 | 0,742 | **0,793** |
| IndoBERT | 0,7733 | 0,774 | 0,785 | 0,773 | 0,764 |
| **IndoBERTweet** ★ | **0,7900** | 0,789 | **0,796** | **0,820** | 0,751 |

→ `model_comparison_balanced3k.{csv,png}` (compare_models kini sertakan IndoBERTweet bila JSON-nya ada).
Peringkat final: **IndoBERTweet > IndoBERT > SVM.** Naik bertahap; IndoBERTweet menang krn domain-fit
(Twitter Indonesia) menolong kelas Netral/Negatif yang butuh nuansa medsos.

**SISA (opsional):** update laporan PDF dgn tabel+chart 3 model; ensemble; tambah data latih.



## 13. Loop improve v2 — audit OOF + re-label v3 (2026-07-01)

Audit kualitas label v2 (keputusan user "terus improve"): OOF 5-fold SVM(char) di seluruh 3000 →
769 disagreement (25,6%); ambil yg model PERCAYA-DIRI beda (margin>=0.6, 260) + konsensus 3 model
di test (13) = 264 kandidat → re-adjudikasi Opus blind (`src/modeling/svm_batch_a.py` OOF; kandidat
di scratchpad). **41 koreksi (16%)**: Positif→Netral 30 (mayoritas Positif non-verified/Sonnet),
Netral→Negatif 11; 223 dikonfirmasi (model yg salah). NOL koreksi Negatif (sudah 100% Opus-verified).
Push ke master+Mongo (durable `relabel_pass2_opus_v3audit_20260701.csv`), rebuild balanced, refresh
flag `in_balanced3k`, re-train.

**SVM (neg+char, stabil, TANPA threshold): 0,743 → 0,7533 (+1pp)** — fixing label MEMBANTU.
**Per-class threshold DI-DROP dari official**: overfit val kecil (pra-v3 +2pp jadi 0,767, pasca-v3
JUSTRU -3pp jadi 0,72). Official SVM kini = neg+char 0,7533 (svm_balanced3k_metrics.json).
IndoBERT & IndoBERTweet **perlu re-run Colab** (label/membership berubah) — notebook sama, dataset
auto-update (Mongo+balanced_3000.csv). Tabel 3-model final menunggu hasil itu.

## 12. DUA VERSI dataset disimpan — v1 (label lama) vs v2 (domain-aware) (2026-07-01)

Keputusan user: simpan **dua-duanya** agar laporan bisa pakai angka tinggi v1 **dan** punya
analisis jujur v2. **Tidak ada re-train** — artefak v1 dipulihkan dari git `6462568`.

| Versi | Model | Akurasi | macro-F1 | Catatan |
|---|---|---|---|---|
| **v1** (Sonnet, label lama) | SVM sklearn | **0,84** | 0,840 | ⚠️ label sebagian salah (288 Neg→Net) |
| v1 | SVM Spark | 0,7767 | 0,774 | |
| v1 | **IndoBERT** | **0,8467** | 0,847 | ⚠️ angka ter-inflasi label keliru |
| **v2** (domain-aware, valid) | SVM sklearn (char+thr) | 0,7667 | 0,765 | label valid (Opus) |
| v2 | IndoBERT | 0,7733 | 0,774 | |
| v2 | **IndoBERTweet** ★ | **0,79** | 0,789 | terbaik di label valid |

**⚠️ CAVEAT (penting utk laporan):** angka v1 lebih tinggi karena diuji pada **label lama yang
sebagian KELIRU** (288 komentar serang-orang salah ditandai Negatif; terverifikasi Opus). Jadi v1
sebagian mengukur "meniru label salah". v2 = label valid (stance domain-aware), plafon jujur ~0,77–0,79.
Disarankan: kalau pakai v1 di laporan, sebut keterbatasannya; v2 lebih defensibel secara ilmiah.

**Artefak:**
- v1: `outputs/reports/*_balanced3k_v1sonnet_*` (metrics/confusion/predictions) + dataset
  `outputs/labeling/balanced_3000_v1sonnet.csv`. Mongo TIDAK diubah (tetap label v2 domain-aware).
- v2: `*_balanced3k_*` (default, aktif di Mongo).
- Gabungan: `model_comparison_balanced3k_v1_vs_v2.{csv,png}`.

**Revert PENUH ke v1 (bila perlu):** restore label lama ke Mongo dari git
`git show 6462568:outputs/labeling/labeling_dataset.csv` → push_relabel (atau push manual) +
`cp balanced_3000_v1sonnet.csv balanced_3000.csv`. Saat ini Mongo = v2 (label valid).

## 9. Regen processed_svm + roadmap akurasi (2026-06-30, akhir sesi)

**Regen penuh `processed_svm` (14.107 baris, Spark) + re-train SVM** dengan fix preprocessing
(ijasah→ijazah, slang +24, stemming `diseting→seting`/`mentri`). Dampak ke balanced3k:

| Model | Sebelum | Sesudah | Δ |
|---|---|---|---|
| SVM sklearn | 0,8333 | **0,84** | +0,67pp |
| SVM Spark | 0,7667 | **0,7767** | +1,0pp |
| IndoBERT | 0,8467 | 0,8467 | — (jalur `bert` non-stemmed, tak diregen) |

Naik kecil tapi konsisten (sifatnya konsolidasi fitur TF-IDF, bukan sinyal baru). Artefak
ter-commit (`36b7478`). Peringkat tetap: **IndoBERT > SVM sklearn > SVM Spark**.

**Diagnosis confusion (konsisten di KETIGA model):** error #1 = **Negatif → Positif (stance
flip)** (SVM 13/100, BERT 15/100, Spark 23/100); **Negatif = kelas terlemah**. Neg & Pos
tumpang-tindih kata (ijazah/palsu/asli) → pembeda cuma negasi + arah sikap.

**Roadmap akurasi → `docs/roadmap_akurasi.md`** (commit `c0cb965`), berlapis per leverage:
1. Kualitas label (tertinggi, SEDANG JALAN: verifikasi LS id=9/id=10 → rebuild → re-train).
2. Preprocessing/fitur SVM (negation tagging, emoji→token, lexicon InSet, char n-gram).
3. Model & tuning (sweep classifier; sweep IndoBERT LR/epoch/early-stop/large).
4. Ensemble IndoBERT+SVM, k-fold, per-class threshold.

**KEPUTUSAN AKHIR SESI:** **SVM Spark di-DROP dari fokus** (gap 0,7767 vs 0,84 itu soal
implementasi HashingTF, bukan riset). **Fokus = SVM sklearn + IndoBERT.**

**Menu pengembangan SVM sklearn (next session)** — dipisah biaya:
- **Batch A (TANPA regen, cuma ubah `train_svm_full14k.py build()`, re-train ~1mnt):**
  - A1 **Negation merge** `tidak palsu`→`tidak_palsu` sbg analyzer (kata negasi sudah tersimpan
    di kolom `svm`, jadi tak perlu regen) — serang stance-flip #1. *Paling menjanjikan.*
  - A2 FeatureUnion word + char n-gram (3–5). A3 sweep `LogisticRegression`/`ComplementNB`.
  - A4 **Per-class threshold** via `decision_function` (angkat recall Negatif). A5 perluas grid
    (`C` halus, `max_df`, `(1,3)`, `max_features`). A6 `CalibratedClassifierCV` (utk ensemble).
  - Ukur via **OOF cross-val** (bukan single split), metrik utama **f1 Negatif**.
- **Batch B (BUTUH regen processed_svm ~15mnt):** B1 emoji→token sentimen (kini dibuang di
  `clean_for_svm`), B2 fitur lexicon InSet, B3 fix error stemmer Sastrawi (`dibantah→ban`).

**Lanjutan yang masih nunggu user:** selesaikan verifikasi LS id=9 + id=10 →
`python -m src.rebuild_balanced_from_verification --commit` → re-train ketiga model (label
berubah memengaruhi semua). Catatan: Batch A bisa dikerjakan paralel TANPA nunggu verifikasi.

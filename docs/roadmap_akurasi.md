# Roadmap Peningkatan Akurasi — SVM & IndoBERT

> Tujuan: menaikkan akurasi **SVM (sklearn + Spark)** dan **IndoBERT** pada dataset
> balanced 3000 (1000/kelas). Dokumen ini terurut **berdasarkan leverage** (dampak ÷ usaha),
> bukan abjad. Mulai dari Lapisan 1.

## Status awal (baseline balanced3k, 2026-06-30)

| Model | Akurasi | macro-F1 | f1 Negatif | f1 Netral | f1 Positif |
|---|---|---|---|---|---|
| IndoBERT | **0.8467** | 0.8469 | 0.790 | 0.885 | 0.865 |
| SVM sklearn | 0.8400 | 0.8397 | 0.802 | 0.842 | 0.875 |
| SVM Spark MLlib | 0.7767 | 0.7739 | 0.707 | 0.822 | 0.793 |

## Diagnosis dari confusion matrix (akar masalah)

Baris = label asli, kolom = prediksi. Angka tebal = error terbesar.

```
            SVM sklearn         IndoBERT            SVM Spark
          Neg  Net  Pos       Neg  Net  Pos       Neg  Net  Pos
true Neg [ 81    6  *13 ]    [ 79    6  *15 ]    [ 64   13  *23 ]
true Net [*16   80    4 ]    [*12   85    3 ]    [  9   83    8 ]
true Pos [  5    4   91 ]    [  9    1   90 ]    [  8    6   86 ]
```

**Tiga temuan yang menyetir seluruh roadmap:**

1. **Negatif → Positif (stance flip) = error #1 di SEMUA model.** Negatif (membantah
   tuduhan / "ijazah asli") dan Positif (percaya tuduhan / "ijazah palsu") **lexically
   tumpang-tindih** — sama-sama menyebut *ijazah, jokowi, palsu, asli, fitnah*. Pembedanya
   cuma **negasi + arah sikap**, justru sinyal yang paling mudah hilang di TF-IDF unigram.
2. **Negatif = kelas terlemah** di semua model (recall 81/79/64%). Ini bottleneck akurasi.
3. **Netral → Negatif** signifikan di model linear (SVM 16, BERT 12) — Netral "bocor" jadi Negatif.

→ Konsekuensi: lever paling efektif bukan "model lebih besar", melainkan **(a) kualitas
label Negatif** dan **(b) menangkap negasi/sikap** di preprocessing.

---

## Lapisan 1 — Kualitas Label  ⭐ leverage tertinggi (SEDANG JALAN)

Pada plafon ~85% dengan label dari LLM, **noise label = atap akurasi**. Memperbaiki 1 label
Negatif salah lebih bernilai daripada tuning hyperparameter apa pun.

| # | Aksi | Target | Usaha | Status |
|---|---|---|---|---|
| 1.1 | Verifikasi manual disagreement (LS id=9, 551 task) | label | sedang | 🔄 jalan |
| 1.2 | Verifikasi manual low-conf Negatif (LS id=10, 699 task) | label | sedang | 🔄 jalan |
| 1.3 | `rebuild_balanced_from_verification --commit` → buang "Tidak terbaca" + koreksi gold + top-up 1000/kelas → re-train 3 model | semua | rendah | ⏳ setelah 1.1/1.2 |
| 1.4 | **Consensus-wrong audit**: baris yang ditebak Positif oleh KETIGA model padahal label Negatif → kandidat kuat label salah. Relabel tertarget (murah, presisi tinggi). | label | rendah | 📋 baru |
| 1.5 | **Gold test-set manusia** untuk eval jujur (project id=6) — ukur akurasi vs label MANUSIA, bukan vs LLM. Mengubah arti "akurasi" jadi kredibel utk laporan. | eval | sedang | 📋 baru |

**Ekspektasi:** +2–4pp akurasi, terkonsentrasi di Negatif. Ini langkah dengan rasio
dampak/usaha terbaik dan sudah berjalan.

---

## Lapisan 2 — Preprocessing & Fitur (khusus SVM)

Menyerang langsung temuan #1 (negasi/sikap hilang). IndoBERT relatif kebal di sini karena
kontekstual, jadi fokus ke jalur SVM.

| # | Aksi | Kenapa | Usaha | Ekspektasi |
|---|---|---|---|---|
| 2.1 | **Negation tagging**: `tidak palsu` → token `tidak_palsu` (gabung negasi+kata berikut sebelum stopword removal) | langsung serang stance flip Neg↔Pos | sedang | +1–2pp SVM |
| 2.2 | **Emoji → token sentimen** (jangan dibuang di `clean_for_svm`): ❤😍👍 → `emoji_positif`, 😡👎 → `emoji_negatif` via `demoji`/peta manual | emoji = sinyal sentimen kuat, sekarang DIBUANG | sedang | +0.5–1pp |
| 2.3 | **Fitur lexicon InSet** (kamus sentimen Indonesia): tambah skor pos/neg per komentar sbg fitur numerik di samping TF-IDF | prior eksplisit utk kelas lemah | sedang | +0.5–1pp |
| 2.4 | **Char n-gram (3–5)** digabung word n-gram (`FeatureUnion`) | robust ke typo/varian ejaan tanpa stemming | rendah | +0.3–0.8pp |
| 2.5 | Stemmer: dokumentasikan/atasi error rule-based Sastrawi (`dibantah→ban`, `seandainya→anda`); pertimbangkan stemmer alternatif (mpstemmer) atau lemmatizer | mengurangi token rusak | sedang | +0.3pp |
| 2.6 | `sublinear_tf=True` + tuning `max_features` di TF-IDF | redam dominasi kata sering | rendah | +0–0.5pp |

**Catatan:** 2.1 dan 2.2 paling menjanjikan karena menyasar akar error #1. Lakukan satu-satu,
ukur via OOF cross-val agar atribusi jelas.

---

## Lapisan 3 — Model & Hyperparameter

### 3a. SVM sklearn
| # | Aksi | Usaha |
|---|---|---|
| 3.1 | Coba `LogisticRegression` & `ComplementNB` (kuat utk teks pendek/imbalance) sbg pembanding `LinearSVC` | rendah |
| 3.2 | Perluas grid: `C` lebih halus, `min_df`/`max_df`, `ngram (1,3)` | rendah |
| 3.3 | Kalibrasi probabilitas (`CalibratedClassifierCV`) → confidence layak utk thresholding & ensemble | rendah |

### 3b. SVM Spark — **tutup gap 0.7767 vs 0.84**
Gap 6pp vs sklearn pada data IDENTIK = ada beda implementasi, bukan beda data.
| # | Aksi | Usaha |
|---|---|---|
| 3.4 | Audit paritas: tokenizer, `HashingTF` vs `CountVectorizer`, jumlah fitur hash (collision!), `minDF`, `regParam`, normalisasi IDF | sedang |
| 3.5 | Ganti `HashingTF`→`CountVectorizer` (hindari collision) + samakan vocab dgn sklearn | sedang |
| 3.6 | Samakan regularisasi (`LinearSVC` Spark OvR vs sklearn) + class weighting | rendah |

**Ekspektasi 3b:** Spark naik mendekati sklearn (~+4–5pp) tanpa data baru. Win murah & nyata.

### 3c. IndoBERT (headroom paling besar di antara model)
| # | Aksi | Usaha | Ekspektasi |
|---|---|---|---|
| 3.7 | Sweep hyperparameter: LR {1e-5,2e-5,3e-5}, epoch {3,4,5}, warmup, weight decay, batch {16,32} | sedang | +0.5–1.5pp |
| 3.8 | **Early stopping** on val macro-F1 (hindari overfit/underfit epoch tetap) | rendah | +0.5pp |
| 3.9 | Coba `indobert-large-p1` atau `indobert-base-p2` | sedang | +0.5–1pp |
| 3.10 | Tuning `max_length` (cek distribusi panjang token; jangan truncate sinyal) | rendah | +0–0.5pp |
| 3.11 | **Pertahankan emoji** di `clean_for_bert` (demojize ke teks) — IndoBERT bisa pakai konteks emoji | rendah | +0.3pp |
| 3.12 | Augmentasi data Negatif (EDA: sinonim/swap, atau back-translation id→en→id) → seimbangkan kesulitan kelas | sedang | +0.5–1pp |

---

## Lapisan 4 — Ensemble & Evaluasi

| # | Aksi | Kenapa | Usaha |
|---|---|---|---|
| 4.1 | **Ensemble IndoBERT + SVM** (soft-vote pakai prob terkalibrasi, atau stacking LogReg) | error keduanya beda-beda → komplementer | sedang |
| 4.2 | **k-fold CV** (bukan single split) utk estimasi akurasi stabil ± std | angka laporan lebih kredibel | rendah |
| 4.3 | **Per-class threshold tuning** (geser batas Neg vs Pos) untuk angkat recall Negatif | langsung serang bottleneck | rendah |
| 4.4 | Error analysis berkelanjutan: tiap iterasi, ekspor 20 error Neg→Pos teratas → cari pola → umpan balik ke rubrik/preprocessing | loop perbaikan | rendah |

---

## Urutan eksekusi yang disarankan

```
SEKARANG  → [Lapisan 1] selesaikan verifikasi LS id=9/id=10 → rebuild → re-train   (sudah jalan)
            └ sekalian 1.4 consensus-wrong audit (murah, presisi tinggi)
BERIKUT   → [Lapisan 3b] tutup gap SVM Spark (win termurah, +4–5pp Spark)
            → [Lapisan 2.1 + 2.2] negation tagging + emoji token (serang stance flip)
LALU      → [Lapisan 3c] sweep IndoBERT + early stopping (+ coba large/p2)
AKHIR     → [Lapisan 4.1] ensemble IndoBERT+SVM untuk perasan terakhir
SEPANJANG → [1.5] gold test-set manusia + [4.2] k-fold + [4.4] error analysis tiap iterasi
```

## Prinsip ukur (agar atribusi jujur)

- Ubah **satu** hal per eksperimen; bandingkan via **OOF cross-val / k-fold**, bukan satu split.
- Lacak **f1 Negatif** sebagai metrik utama (bottleneck), bukan cuma akurasi global.
- IndoBERT tak terpengaruh perubahan jalur SVM (teks `bert` non-stemmed) — tak perlu re-run
  BERT saat eksperimen preprocessing SVM. Sebaliknya, perubahan label (Lapisan 1) memengaruhi
  **ketiga** model → re-train semua.
- Catat setiap eksperimen di `handoff.md` + `outputs/reports/` (sudah ada konvensi tag).

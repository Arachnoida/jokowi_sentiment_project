# Penjelasan Notebook Modeling — SVM, IndoBERT, IndoBERTweet

> Rangkuman untuk presentasi/laporan. Dataset fokus: **balanced 3.000 v1audited**
> (label LLM Sonnet + audit manusia-LLM, seimbang 3 kelas: Negatif / Netral / Positif).
> Acuan polaritas = sikap terhadap **NARASI tuduhan "ijazah Jokowi palsu"**, bukan sosok Jokowi.

## Hasil final (test 300 = 100/kelas, split identik)

| Model | Akurasi | macro-F1 | F1 Neg | F1 Net | F1 Pos |
|---|---|---|---|---|---|
| SVM + TF-IDF | 0,8567 | 0,8556 | 0,815 | 0,887 | 0,865 |
| IndoBERT | 0,8633 | 0,8615 | 0,811 | 0,916 | 0,857 |
| **IndoBERTweet** ★ | **0,8733** | 0,8721 | 0,825 | 0,898 | 0,893 |

Peringkat: **IndoBERTweet > IndoBERT > SVM**. Ketiganya diuji pada **test set 300 yang sama persis**.

---

## Split data 70/20/10 (sama untuk ketiga model)

- **Train 70% (2.100)** — melatih model.
- **Validation 20% (600)** — tuning hyperparameter / pilih checkpoint terbaik (early stopping).
- **Test 10% (300 = 100/kelas)** — evaluasi akhir, belum pernah dilihat model.

Prosedur kanonik (identik → adil): urut `comment_id` → potong test 10% → val (`0.20/0.90` sisa) → train,
`stratify` + `random_state=42`. Ini **konvensi holdout standar** (bukan turunan rumus).

**Lokasi kode split:**
- SVM: `src/modeling/train_svm.py` → fungsi `split_version()` (baris ~89–93).
- IndoBERT: `src/modeling/train_indobert.py` → `load_splits()`.
- IndoBERTweet: inline di sel "Muat data" notebook.

> Catatan: SVM menggabungkan train+val (2.700) untuk `GridSearchCV` (pakai `PredefinedSplit`
> menandai val), test 300 tetap disisihkan → output tampil `n_train=2700 n_test=300`.

---

## 1. SVM + TF-IDF — `notebooks/3_modeling/svm_train_colab.ipynb`

**Model klasik (bag-of-words), cepat, CPU cukup, interpretable.**

- **Fitur:** TF-IDF dari kolom `svm` (teks ter-stem, buang stopword), word n-gram `(1,2)`, `sublinear_tf`.
- **Classifier:** `LinearSVC(class_weight="balanced")`.
- **Tuning:** `GridSearchCV` 24 kombinasi (`ngram × min_df × C`), skor macro-F1 validasi.
- **Runtime:** memanggil script repo `!python -m src.modeling.train_svm --tag v1audited --subset balanced_3000_v1audited.csv`.

**Isi 8 bagian:**
1. Clone repo + dependensi
2. Set MONGO_URI + jalankan training (grid search)
3. **Hasil grid search** — tabel + heatmap macro-F1 (C × min_df)
4. Evaluasi test — confusion matrix + classification report + confusion ternormalisasi + bar P/R/F1
5. **Top fitur diskriminatif per kelas** — kata/bigram penanda tiap kelas (interpretabilitas)
6. Contoh komentar + analisis error (✅/❌)
7. Analisis keyakinan — prediksi salah paling yakin (kandidat label keliru) + histogram
8. Perbandingan 3 model

**Keunggulan demo:** grid heatmap + top fitur (SVM *interpretable*, beda dari BERT). Hasil **0,8567**.

---

## 2. IndoBERT — `notebooks/3_modeling/indobert_finetune_colab.ipynb`

**Deep learning kontekstual (Transformer), butuh GPU.**

- **Model:** `indobenchmark/indobert-base-p1` (di-*pretrain* korpus Indonesia besar: Wikipedia/berita).
- **Fine-tuning:** melanjutkan pelatihan bobot pretrained ke tugas sentimen (bukan sekadar pilih hyperparameter).
- **Hyperparameter:** 4 epoch, LR `2e-5`, batch 16, `MAX_LEN=128`, `load_best_model_at_end` (checkpoint val terbaik).
- **Input:** teks `bert` (preproses ringan: lowercase, buang URL/emoji; **tanda baca & imbuhan dipertahankan** agar tokenizer subword optimal).
- **Runtime:** memanggil script repo `train_indobert --tag v1audited --subset ...`.

**Isi:** overview + tabel hyperparameter, clone, training, **loss & metrik per epoch**, confusion + classification report, contoh komentar, analisis keyakinan, perbandingan 3 model.

**Hasil 0,8633** — unggul di kelas Netral (F1 0,916).

---

## 3. IndoBERTweet + Weighted Loss — `notebooks/3_modeling/indobertweet_v1audited_colab.ipynb`

**Model TERBAIK proyek. Deep learning + domain medsos + tangani imbalance.**

- **Model:** `indolem/indobertweet-base-uncased` — di-*pretrain* pada **Twitter Indonesia** (bahasa
  medsos/alay/singkatan) → paling cocok untuk komentar YouTube.
- **Weighted cross-entropy:** bobot kelas terbalik thd frekuensi (kelas minoritas dapat bobot lebih besar).
- **Hyperparameter:** 6 epoch + early stopping, LR `2e-5`, batch 16, `MAX_LEN=128`, checkpoint val terbaik.
- **Data:** label dari `balanced_3000_v1audited.csv` (override), teks `bert` dari Mongo.
- **Struktur:** training **inline** di sel (bukan panggil script) + banyak visualisasi.

**Isi 11 bagian (26 sel):**
0. Setup GPU + dependency
1. Clone repo (PAT) + MONGO_URI
2. Muat data + split + **bar distribusi kelas**
3. Tokenisasi WordPiece + **contoh 1 komentar → sub-kata**
4. Model + **Weighted Trainer** (cetak bobot kelas)
5. Fine-tuning
6. **Kurva loss & metrik per epoch** (lihat interpretasi di bawah)
7. Evaluasi test + tabel P/R/F1 per kelas
8. Confusion ternormalisasi + bar F1 per kelas
9. Contoh prediksi benar/salah (salah-paling-yakin)
10. Perbandingan 3 model
11. Simpan + unduh artefak

**Hasil 0,8733** — tertinggi; Positif F1 0,893.

---

## Membaca kurva pelatihan (Bagian 6 IndoBERT/IndoBERTweet)

Dua panel: **Loss** (kiri) & **Metrik validasi** (kanan).

- **train loss** turun terus → model makin cocok dgn data latih.
- **val loss** turun lalu **naik** (mis. minimum ~epoch 3) → titik model mulai **overfit**.
- **akurasi & macro-F1 validasi** naik lalu **mendatar** (~0,85 sejak epoch 3) → model "matang" cepat.

**Korelasi penting:**
1. train loss ↓ terus + val loss ↓-lalu-↑ = **tanda klasik overfitting** setelah titik divergensi.
2. Val loss naik **tapi akurasi datar** → model jadi *makin yakin pada sebagian tebakan salah*
   (loss menghukum kepercayaan-salah; argmax/akurasi tak berubah). Artinya epoch tambahan tak menambah
   kualitas, cuma overfit.
3. **Mitigasi:** `load_best_model_at_end=True` + `metric_for_best_model='macro_f1'` → model final =
   **checkpoint val-terbaik** (bukan epoch terakhir) + `EarlyStopping`. Jadi overfit di epoch akhir
   tak merusak hasil test.

**Kesimpulan:** model belajar efektif di ~3 epoch pertama; sisanya mulai overfit, tetapi mekanisme
best-checkpoint menjaga hasil akhir tetap optimal.

---

## Perbandingan pendekatan

| Aspek | SVM | IndoBERT | IndoBERTweet |
|---|---|---|---|
| Jenis | klasik (bag-of-words) | Transformer | Transformer |
| Domain pretraining | — | Wikipedia/berita | **Twitter Indonesia** |
| Loss | hinge (SVM) | cross-entropy | **weighted** cross-entropy |
| Input fitur | `svm` (stem, no stopword) | `bert` (imbuhan/tanda baca utuh) | `bert` |
| Latih di | script repo (CPU) | script repo (GPU) | inline notebook (GPU) |
| Interpretable | **ya** (top fitur) | tidak (black box) | tidak |
| Akurasi v1audited | 0,8567 | 0,8633 | **0,8733** |

**Insight:** semua diuji pada test set identik (split kanonik seed=42). Model domain-medsos
(IndoBERTweet) + label yang sudah diaudit = hasil terbaik. SVM tetap kompetitif & interpretable
sebagai baseline.

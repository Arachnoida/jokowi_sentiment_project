# Cheat-Sheet Presentasi — Preprocessing & Modeling

> Buka file ini di samping editor saat presentasi. Fokus: **preprocessing** (utama),
> **modeling** (sedikit). Proyek: sentimen 14.107 komentar YouTube soal narasi
> *"ijazah Jokowi palsu"*, 3 kelas (Negatif / Netral / Positif).

---

## 0. Pembuka (±30 detik)

> "Proyek analisis sentimen **14.107 komentar** soal narasi *ijazah Jokowi palsu*,
> 3 kelas. Empat tahap: pengumpulan data → **preprocessing** → **modeling**
> (SVM vs IndoBERT). Yang khas: ada **dua jalur preprocessing berbeda** untuk dua
> jenis model — dan itu yang akan saya jelaskan."

**Acuan polaritas (kalau ditanya):** sikap terhadap **NARASI tuduhan**, bukan terhadap
sosok Jokowi. Positif = percaya/mendukung tuduhan; Negatif = menolak/membantah;
Netral = tak jelas/bertanya/info.

---

## 1. `src/text_normalizer.py` — JANTUNG preprocessing ⭐ (paling lama di sini)

Satu sumber kebenaran semua logika. Tunjukkan fungsi-fungsinya, jelaskan **kenapa**-nya.

| Fungsi | Baris | Peran |
|---|---|---|
| `clean_for_svm()` | ~105 | cleaning agresif jalur SVM |
| `clean_for_bert()` | ~120 | cleaning minimal jalur IndoBERT |
| `normalize_slang()` | ~133 | normalisasi slang (gk→tidak, sdh→sudah) |
| `tokenize()` / `remove_stopwords()` | ~143 / ~148 | tokenisasi + buang stopword |
| `preprocess_svm_python()` | ~153 | pipeline SVM lengkap |
| `try_stem_sastrawi()` | ~167 | stemming Sastrawi |

### Poin bicara utama — **KENAPA dua jalur?**
- **Jalur SVM** (`preprocess_svm_python`): `clean_for_svm` → `normalize_slang` →
  `tokenize` → `remove_stopwords` → **stemming Sastrawi (string utuh)**.
  → SVM = *bag-of-words* (TF-IDF), tak paham konteks → teks harus "diseragamkan"
  agresif supaya kata seakar dianggap sama.
- **Jalur IndoBERT** (`clean_for_bert`): cleaning **minimal** — lowercase, buang
  URL/mention/emoji; **tanda baca, imbuhan, negasi DIPERTAHANKAN**.
  → IndoBERT paham konteks (tokenizer subword/WordPiece) → morfologi malah membantu;
  stemming justru merusak.

### 3 keputusan desain yang layak dibanggakan (juri suka ini)
1. **Negasi dipertahankan** — `tidak / bukan / belum` sengaja DIKELUARKAN dari
   `STOPWORDS_ID`. Kalau ikut dibuang, *"tidak palsu"* jadi == *"palsu"* → **fatal**
   untuk sentimen. (Ini temuan kualitas yang kami perbaiki.)
2. **SLANG_DICT** (baris ~10) — kamus slang diperluas berbasis frekuensi korpus
   (gk/ga→tidak, sdh→sudah, org→orang, kalo→kalau, dst).
3. **Stemming pada string utuh** (bukan per-token) → hasil konsisten, satu instance
   stemmer (efisien).

---

## 2. Demo transformasi LANGSUNG (jalankan live — sangat meyakinkan) ⭐

Ketik di terminal:

```bash
.venv/bin/python -c "from src.spark.udf import make_svm_text, make_bert_text; \
t='Jokowi ijazahnya palsu bgt sih'; print('Input:', t); \
print('SVM :', make_svm_text(t)); print('BERT:', make_bert_text(t))"
```

Output:
```
Input: Jokowi ijazahnya palsu bgt sih
SVM : jokowi ijazah palsu              <- stem (ijazahnya->ijazah), buang slang(bgt)/stopword(sih)
BERT: jokowi ijazahnya palsu bgt sih   <- utuh, cuma lowercase
```

→ Bukti konkret beda dua jalur dalam **satu layar**. Kalimat penutup demo:
*"SVM dipadatkan ke akar kata; BERT dibiarkan utuh karena dia paham konteks."*

---

## 3. Di MANA preprocessing dijalankan (kalau ditanya arsitektur)

Logika di `text_normalizer.py` **satu**, dieksekusi di **dua tempat (hasil identik)**:

- **Jalur kanonik** → hasil disimpan di **MongoDB Atlas**: koleksi `processed_svm`
  (kolom `svm`) & `processed_bert` (kolom `bert`). Inilah yang dibaca training.
- **Jalur Spark** (`src/spark/preprocess_spark.py` + `udf.py`) → menghitung ulang
  terdistribusi; **diverifikasi 100% cocok** dengan jalur kanonik.

---

## 4. Modeling — buka cepat (sedikit saja)

### `src/modeling/train_svm_full14k.py` — SVM (baseline)
- **TF-IDF + `LinearSVC`**, `GridSearchCV` (ngram × min_df × C), tuning di validation
  (`PredefinedSplit`), refit train+val, evaluasi test.
- **Split deterministik 70/20/10** (`split_version`): urut `comment_id` + `seed=42`
  → **test set IDENTIK** untuk SVM & IndoBERT → **perbandingan adil**. test = **1411**.
- Hasil: **macro-F1 0.669** (acc 0.682). Best: C=0.1, ngram (1,2), min_df=3.

### `src/modeling/train_indobert.py` — IndoBERT (deep learning)
- Fine-tune `indobenchmark/indobert-base-p1`, 3 kelas, 4 epoch,
  `load_best_model_at_end` (pilih checkpoint val terbaik).
- Butuh **GPU** → dijalankan di Colab (file `.py` ini bisa dipanggil dari notebook
  Colab atau langsung di mesin ber-GPU).

### Metrik utama: **macro-F1**
Karena 3 kelas tak seimbang (Netral paling langka) → macro-F1 menghargai semua kelas
setara, tak hanya kelas mayoritas.

---

## 5. (Opsional) Jalur Big Data — kalau ditanya

`src/spark/` = pipeline terdistribusi **PySpark** di **cluster 2 mesin** (via Tailscale):
- `preprocess_spark.py` — preprocessing terdistribusi (100% cocok vs kanonik).
- `train_svm_spark.py` — SVM MLlib (CountVectorizer→IDF→OneVsRest LinearSVC).
- Riwayat run tersimpan di **Spark History Server** (`:18080`).
- **Posisi jujur:** angka final = **sklearn**; Spark = demonstrasi pipeline Big Data
  setara metodologi (selisih kecil karena beda engine: IDF, loss, regParam).

---

## 6. Antisipasi pertanyaan dosen/juri

| Pertanyaan | Jawaban singkat |
|---|---|
| Kenapa **dua jalur** preprocessing? | SVM bag-of-words butuh normalisasi agresif + stem; IndoBERT paham konteks → teks dibiarkan utuh. |
| Kenapa **negasi tidak dibuang**? | "tidak palsu" ≠ "palsu" — negasi penentu polaritas. Stopword negasi sengaja dipertahankan. |
| Kenapa **macro-F1**, bukan accuracy? | Kelas tak seimbang (Netral langka); macro-F1 adil ke semua kelas. |
| Kenapa **test set sama** antar model? | Split deterministik (sort comment_id + seed=42) → perbandingan apple-to-apple. |
| Kenapa pakai **satu dataset** (full 14k)? | Awalnya uji beberapa versi subset; disederhanakan ke satu dataset penuh agar fokus & konsisten. |
| **SVM vs IndoBERT**, mana menang? | Pada data relatif kecil + sinyal sangat leksikal (palsu/bohong/fitnah), SVM kompetitif. (Angka final menyusul setelah IndoBERT full-14k selesai.) |
| Kenapa **Sastrawi**? | Stemmer bahasa Indonesia standar — menyatukan imbuhan ke akar kata untuk fitur SVM. |
| **Di mana labeling** dilakukan? | Di **Label Studio** (hosted di HF Spaces): project "Sentimen Jokowi" (14.107 task) + "Verifikasi Manual" (1.411 task), lengkap jejak audit (annotator + timestamp). |
| **Mana label manual, mana AI?** | Dilacak via kolom **`label_source`** di MongoDB (`llm` / `manual`). Saat ini seluruh 14.107 = **LLM-assisted** (`claude-llm`); verifikasi manual dilakukan bertahap (~1013 ditargetkan) → otomatis ditandai `manual`. |
| Kenapa pakai **label LLM**? | LLM-assisted labeling: skalabel & konsisten untuk 14k komentar; **divalidasi manusia** pada sampel via Label Studio (Cohen's kappa). Jujur disebut "LLM-assisted", bukan gold manual. |

---

## 7. Tips eksekusi
- **Jangan baca kode baris-per-baris** — tunjuk fungsi, jelaskan *kenapa* (keputusan desain).
- Nyalakan **nomor baris** di editor biar mudah menunjuk.
- Urutan aman: `text_normalizer.py` → **demo live** → `train_svm_full14k.py`.
- Kalau gugup, 3 kalimat kunci:
  1. **"Dua jalur preprocessing"** (SVM agresif vs BERT minimal).
  2. **"Negasi dipertahankan"** karena menentukan sentimen.
  3. **"Test set identik"** supaya perbandingan model adil.

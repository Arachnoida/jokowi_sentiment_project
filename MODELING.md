# Tahap Modeling — SVM vs IndoBERT

Klasifikasi sentimen komentar YouTube (isu ijazah Jokowi) ke **3 kelas**:
`Negatif (0)`, `Netral (1)`, `Positif (2)`. Polaritas di-anchor ke **narasi tuduhan**
(lihat `outputs/labeling/_RUBRIK.md`).

Membandingkan dua paradigma pada dataset & split yang **identik**:

| Jalur | Paradigma | Input (Mongo) | Preprocessing | Tempat latih |
|-------|-----------|---------------|---------------|--------------|
| **A — SVM + TF-IDF** | ML tradisional | `processed_svm.svm` | clean agresif + slang + buang stopword + **stemming**; **negasi dipertahankan** | Lokal (CPU) |
| **B — IndoBERT** | Deep learning (fine-tuning) | `processed_bert.bert` | cleaning minimal (morfologi & negasi terjaga) | Colab/Kaggle (GPU) |

## Desain dataset

- **Seimbang: 1.000/kelas → 3.000 total** (komentar `in_balanced_set=true` di
  `raw_comments`), label **LLM-assisted** (`claude-llm`).
- **Split stratified 70/20/10** (train/val/test), seed `42`, **urut `comment_id` +
  dilakukan SEBELUM preprocessing** → notebook SVM & IndoBERT menghasilkan **test/val
  identik** (perbandingan adil). Disimpan via field `split` di `processed_svm`/`processed_bert`.
- Metrik utama **macro-F1**; karena seimbang, **accuracy** juga bermakna. Selalu
  laporkan **per-kelas P/R/F1 + confusion matrix**.

> Catatan: test set seimbang mengukur performa pada *kondisi seimbang*, bukan distribusi
> dunia nyata yang timpang. Wajar untuk studi perbandingan model — laporkan apa adanya.

## Alur kerja

1. **Pelabelan** (sudah): label `claude-llm` di `raw_comments`, 3.000 ditandai balanced.
2. **`notebooks/2_preprocessing/preprocessing_svm.ipynb`** → `processed_svm` (kolom `svm` + `split`).
3. **`notebooks/2_preprocessing/preprocessing_indobert.ipynb`** → `processed_bert` (kolom `bert` + `split`).
4. **`notebooks/3_modeling/train_svm.ipynb`** — baca `processed_svm`, TF-IDF→LinearSVC, tuning di
   val (`PredefinedSplit`), refit train+val, evaluasi test. Lokal.
5. **`notebooks/3_modeling/indobert_finetune_colab.ipynb`** — baca `processed_bert`, fine-tune
   `indobert-base-p1` di Colab/GPU, evaluasi test set identik.
6. **Bandingkan** macro-F1 & per-kelas kedua model (deliverable utama).

Notebook modeling **self-contained** (tanpa `import src`) — metrik & confusion matrix
dihitung inline. Helper `compare_models()` tersedia di `src/modeling/evaluate.py`.

## Hasil

### SVM + TF-IDF (test) — ✅ selesai
- **Macro-F1 0,699** | Accuracy 0,697 | Weighted-F1 0,699.
- Per-kelas F1: **Negatif 0,761** (P 0,83), Netral 0,679, Positif 0,657.
- Kebingungan utama di seputar **Netral** (Positif↔Netral); Negatif vs Positif jarang tertukar.
- Param terbaik: `C=0.1`, unigram `(1,1)`, `min_df=2`.
- Artefak: `outputs/models/svm_tfidf.joblib` (gitignored), `outputs/reports/svm_metrics.json`,
  `svm_test_confusion.png`.

### IndoBERT (test) — ⏳ menunggu run di Colab/GPU
Target: mengungguli baseline SVM (> 0,70). Konfigurasi: 4 epoch, lr 2e-5, batch 16,
max_len 128, `metric_for_best_model="macro_f1"`, seed 42.

## Eksperimen lintas versi dataset (SVM)

`train_svm.ipynb` melatih SVM pada **4 versi** (lihat README untuk definisi flag).
Split kanonik per versi (urut `comment_id` + `seed=42`) → identik dengan IndoBERT.
Artefak: `outputs/reports/svm_versions_{metrics.json,table.csv,compare.png,confusion.png}`.

| Versi | n_train | macro-F1 | accuracy | F1 Neg/Net/Pos |
|-------|---------|----------|----------|-----------------|
| v1 imbalanced 6k | 4198 | 0,602 | 0,643 | 0,70 / 0,45 / 0,65 |
| **v2 balanced 3k** | 2098 | **0,694** | 0,693 | 0,76 / 0,66 / 0,66 |
| v3 imbalanced 10k | 6997 | 0,626 | 0,656 | 0,71 / 0,49 / 0,68 |
| v4 balanced 10k | 4063 | 0,651 | 0,651 | 0,66 / 0,67 / 0,62 |

Temuan: (1) **balance > imbalance** pada macro-F1 — versi imbalanced membuat F1 kelas
minoritas (Netral) anjlok; (2) **kualitas > kuantitas** — v2 (balanced 3k, high-conf)
mengungguli v4 (balanced 10k) karena v4 menyertakan banyak Netral low-confidence;
(3) accuracy pada versi imbalanced bias ke kelas mayoritas (selalu > macro-F1). IndoBERT
per versi: set `VERSION_FLAG` di `indobert_finetune_colab.ipynb`.

## Perbandingan final: SVM vs IndoBERT × 4 versi

IndoBERT difine-tune per versi (`VERSION_FLAG` di `indobert_finetune_colab.ipynb`),
dievaluasi pada test set **identik** dengan SVM. Artefak:
`outputs/reports/{model_comparison.csv, svm_vs_indobert.png}` +
`indobert_metrics_{is6,ibs,is10,ib10}.json`.

| Versi | SVM macro-F1 | IndoBERT macro-F1 | Pemenang |
|-------|--------------|-------------------|----------|
| v1 imbalanced 6k | 0,602 | 0,597 | SVM |
| v2 balanced 3k | **0,694** | 0,633 | SVM |
| v3 imbalanced 10k | 0,626 | **0,644** | IndoBERT |
| v4 balanced 10k | 0,651 | **0,666** | IndoBERT |

**Temuan utama — *crossover* ukuran data:** pada data **kecil** (3k–6k) **SVM unggul**
(paling tajam di v2: 0,694 vs 0,633); pada data **10k** **IndoBERT menyalip** (v3 & v4).
Khas literatur: transformer butuh data lebih banyak untuk mengungguli ML klasik —
keunggulannya tumbuh seiring data & pada kelas sulit (Netral, di mana konteks membantu).
**macro-F1 tertinggi keseluruhan tetap SVM v2 (0,694)**; **IndoBERT terbaik = v4 (0,666)**.

![Perbandingan semua hasil model](outputs/reports/all_models_comparison.png)

*Atas:* macro-F1 SVM vs IndoBERT per versi (crossover terlihat). *Bawah:* heatmap semua
metrik (8 konfigurasi × 5 metrik). Dihasilkan dari `svm_versions_metrics.json` +
`indobert_metrics_{is6,ibs,is10,ib10}.json`.

## Eksperimen peningkatan akurasi (ringkasan)

| Track | Tindakan | Hasil |
|-------|----------|-------|
| **B** Fitur+ensemble SVM | char n-gram + ComplementNB/LogReg + voting (`improve_svm.ipynb`) | Ensemble word+char terbaik, **+2–3%** (CV) |
| **D** Cross-validation | 5-fold CV (`improve_svm.ipynb`) | **macro-F1 jujur ~0,59–0,62**; single-split 0,694 ternyata optimistik |
| **C** IndoBERTweet | ganti model domain-medsos + weighted loss (`indobertweet_improve_colab.ipynb`) | siap dijalankan di Colab |
| **A** Kualitas label | 3-pass LLM consensus voting pada 2.525 baris conf<0,6 (877 label dikoreksi) | **Tidak ada gain terukur** → label dikembalikan |

**Temuan track A (penting & jujur):** relabel konsensus LLM mengoreksi 877 label dan
menaikkan cakupan conf≥0,6 (74%→98%), **tetapi macro-F1 SVM tidak membaik** (v1/v3/v4
datar; "penurunan" v2 0,694→0,59 sebenarnya menyingkap nilai CV sejati — 0,694 itu split
beruntung). **Kesimpulan:** tanpa *gold-standard* manusia, relabel LLM tidak menaikkan
performa terukur; plafon ~0,6 lebih ditentukan **ambiguitas tugas (Positif↔Netral)**
daripada noise label yang bisa dibersihkan. Label asli dipertahankan demi konsistensi.

## Jalur PySpark (Big Data) — migrasi dari notebook ke `.py`

Sebagai pemenuhan komponen **Big Data**, jalur SVM dimigrasikan dari notebook ke
**skrip PySpark** (`src/spark/`). Spark **4.x** dipakai karena environment memakai
**Java 21** (Spark 3.5 tidak didukung resmi di Java 21); dijalankan mode lokal
`local[*]`. Dataset 14k kecil — Spark di sini untuk **mendemonstrasikan pipeline
ML terdistribusi**, bukan karena volume menuntutnya (dicatat jujur).

```
src/spark/
├── session.py          # builder SparkSession (local[*]/cluster, UI, log diredam)
├── export_mongo.py     # one-time: Mongo -> Parquet (lepas dari konektor Mongo-Spark)
├── udf.py              # bungkus text_normalizer + Sastrawi jadi UDF Spark
├── preprocess_spark.py # teks mentah -> fitur svm/bert via UDF -> features_spark.parquet
├── eda_spark.py        # EDA terdistribusi: distribusi/panjang/vocab/leakage/lift
├── train_svm_spark.py  # Spark ML: Tokenizer->CountVectorizer->IDF->OneVsRest(LinearSVC)
└── cluster.sh          # start/stop standalone cluster (Master :8080 + Worker)
```

**Cara jalan (pipeline self-contained, semua via Spark):**
`python -m src.spark.export_mongo` → `python -m src.spark.preprocess_spark` →
`python -m src.spark.eda_spark` → `python -m src.spark.train_svm_spark`.
`preprocess_spark` menulis `features_spark.parquet` (fitur + flag versi) yang dibaca
langsung oleh `train_svm_spark` → preprocessing→training **sepenuhnya Spark** (tak
kembali ke Mongo). Artefak: `outputs/reports/{svm_spark_*, svm_sklearn_vs_spark.csv,
eda_spark.json, eda_spark.png}`.

**EDA terdistribusi (`eda_spark.py`)** menghitung via agregasi Spark (groupBy/explode/
join): distribusi kelas (Neg 5531 / Net 3060 / Pos 5516), rata-rata kata (mentah 15,3 →
svm 11,6 → bert 15,2), **vocab SVM 14.844**, svm kosong 0,52%, duplikat 1,45%, top term
diskriminatif per kelas (skor lift), dan **cek leakage** split v5: `comment_id` antar
split disjoint (0 overlap), tapi **61 teks identik** (198 baris, 1,4%) muncul lintas
split — komentar pendek beda `comment_id` yang menyusut jadi token sama setelah cleaning
agresif (mis. "mantap"); minor & dilaporkan apa adanya.

### Spark Web UI & standalone cluster

Dua level tampilan (lihat juga `session.py`, `cluster.sh`):

| UI | URL | Isi | Cara |
|----|-----|-----|------|
| **Application UI** | `localhost:4040` | Jobs/Stages/**Executors**/SQL/DAG per-aplikasi | otomatis ON saat skrip jalan; cepat hilang saat selesai |
| **Master UI** | `localhost:8080` | **Worker node** terdaftar, app aktif/selesai, resource | jalankan standalone cluster |

- **Tahan UI :4040** agar sempat dibuka (job `local[*]` cuma beberapa detik):
  `SPARK_HOLD=1 python -m src.spark.train_svm_spark` lalu buka `localhost:4040`,
  tekan ENTER utk menutup. Matikan UI `SPARK_UI=0`, ganti port `SPARK_UI_PORT=...`.
- **Standalone cluster (Master :8080 + Worker)** — `local[*]` tidak punya Worker
  terpisah, jadi untuk screenshot "Worker node" skripsi jalankan cluster nyata:
  ```bash
  bash src/spark/cluster.sh start          # Master :8080 + Worker :8081 (loopback)
  SPARK_MASTER=spark://127.0.0.1:7077 python -m src.spark.train_svm_spark
  bash src/spark/cluster.sh status         # cek Worker UP
  bash src/spark/cluster.sh stop
  ```
  Saat `SPARK_MASTER` diset, executor (JVM Worker terpisah) otomatis dapat
  `PYSPARK_PYTHON`=venv + `PYTHONPATH`=akar proyek → UDF Sastrava bisa diimpor di
  Worker (terverifikasi: `preprocess_spark` di cluster tetap 100% cocok). pip-pyspark
  4.x tak punya `start-master.sh`, jadi `cluster.sh` meluncurkan Master/Worker via
  `bin/spark-class` (cara yang sama di balik layar). Atur resource Worker via env
  `WORKER_CORES`/`WORKER_MEM`. Log daemon di `logs/spark/` (gitignored).

**Kesetaraan dijaga:** split train/val/test deterministik dihitung sekali (logika
identik `train_svm_full14k.py`) lalu di-join ke Spark DataFrame → **test set identik**
lintas sklearn/Spark/IndoBERT. Preprocessing Spark **terverifikasi 100% cocok** dengan
fitur Mongo (`svm` & `bert`).

### SVM Spark vs SVM sklearn (test macro-F1)

| Versi | sklearn | **PySpark** | selisih | F1 Neg/Net/Pos (Spark) |
|-------|---------|-------------|---------|------------------------|
| v1 imbalanced 6k | 0,602 | 0,562 | −0,039 | 0,65 / 0,41 / 0,63 |
| v2 balanced 3k | 0,694 | 0,545 | −0,150 | 0,59 / 0,54 / 0,50 |
| v3 imbalanced 10k | 0,626 | 0,561 | −0,065 | 0,64 / 0,43 / 0,61 |
| v4 balanced 10k | 0,651 | 0,627 | −0,024 | 0,62 / 0,67 / 0,59 |
| v5 full 14k | 0,669 | 0,615 | −0,054 | 0,65 / 0,56 / 0,63 |

**Kenapa Spark sedikit di bawah sklearn (struktural, bukan bug/tuning):**
Spark MLlib `CountVectorizer+IDF` **tidak punya `sublinear_tf`** (TfidfVectorizer
sklearn memakainya), multiclass lewat **OneVsRest** (Spark hanya SVM biner), dan
regularisasi diparametri `regParam` (bukan `C`). Menambah `regParam=0.001` ke grid
**tidak terpilih** → gap memang dari pembobotan TF-IDF, bukan tuning. **Pola peringkat
tetap sama** (v4/v5 terbaik di antara multi-kelas; Netral kelas tersulit). Untuk angka
final skripsi tetap pakai **sklearn**; Spark dilaporkan sebagai jalur Big Data setara
metodologi.

> **IndoBERT tidak dimigrasi ke Spark.** Spark MLlib tidak punya jalur fine-tune
> transformer; IndoBERT tetap HuggingFace/PyTorch di Colab GPU. Ini batas wajar — Spark
> menangani jalur SVM/feature-engineering, transformer di luar cakupannya.

## Kode pendukung

```
src/modeling/
├── labels.py     # kosakata label (LABELS, LABEL2ID) + parser ekspor Label Studio
└── evaluate.py   # metrik bersama (macro-F1, confusion matrix, compare_models)
```

> Alur lama (`dataset.py` build parquet, `train_svm.py` baca parquet) diarsipkan di
> `archive/src/modeling/` — digantikan notebook berbasis Mongo.

## Dependency

- **Lokal (SVM sklearn):** `pip install -r requirements.txt` (scikit-learn, joblib,
  matplotlib, pymongo, dll).
- **Lokal (SVM PySpark):** sama `requirements.txt` (sudah memuat `pyspark>=4.0`) +
  **JDK 17/21** terpasang (`java -version`). Jalan mode lokal `local[*]`, tanpa cluster.
- **Colab (IndoBERT):** notebook meng-`%pip install` transformers + torch sendiri;
  set Runtime → GPU (T4).

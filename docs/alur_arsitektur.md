# Gambaran Alur Arsitektur — Kondisi Sekarang

> Ringkas: **mana yang pakai Spark (master + workers), mana yang lokal (single-node).**
> Dipakai untuk menjawab "ini jalannya di mana?" saat presentasi.

---

## 1. Jawaban cepat

| Tahap | Engine | Pakai Spark master/workers? |
|---|---|---|
| Preprocessing (bikin `processed_svm` & `processed_bert`) | **Spark cluster** | ✅ Ya — ravi-zorin + rocky-server |
| Training **SVM sklearn** (baseline utama) | scikit-learn | ❌ Tidak — lokal, 1 proses Python |
| Training **SVM Spark MLlib** (varian Big Data) | **Spark MLlib** | ✅ Ya — bisa cluster |
| Training **IndoBERT** (fine-tune) | PyTorch / HuggingFace | ❌ Tidak — lokal, CPU |

**Inti:** Spark dipakai untuk **data engineering (preprocessing)** dan **satu varian model (SVM MLlib)**.
Model "juara" (IndoBERT) dan baseline (SVM sklearn) dilatih **lokal** — karena library-nya
(PyTorch, scikit-learn) memang jalan single-node, bukan terdistribusi.

---

## 2. Diagram alur penuh

```
                         ┌─────────────────────────────┐
                         │   MongoDB Atlas (cloud)      │
                         │   collection: raw_comments   │
                         │   14.107 komentar + label    │
                         └──────────────┬──────────────┘
                                        │  baca (driver)
                                        ▼
        ╔═══════════════════════════════════════════════════════════╗
        ║              TAHAP 1 — PREPROCESSING (SPARK)              ║
        ║  regenerate_processed_mongo.py  /  preprocess_spark.py    ║
        ║                                                           ║
        ║   Spark Master :8080  ──┬── Worker lokal (ravi-zorin)     ║
        ║                         └── Worker remote (rocky-server)   ║
        ║                              via Tailscale                 ║
        ║                                                           ║
        ║   text_normalizer.py (UDF di executor):                   ║
        ║     casefold → bersihkan URL/emoji → normalisasi slang    ║
        ║     → tokenisasi → (SVM: stopword + stemming Sastrawi)    ║
        ╚════════════════════════════┬══════════════════════════════╝
                                     │  tulis balik (pymongo)
                                     ▼
              ┌──────────────────────┴──────────────────────┐
              ▼                                              ▼
   ┌────────────────────┐                        ┌────────────────────┐
   │  processed_svm     │                        │  processed_bert    │
   │  (teks ter-stem)   │                        │  (teks ringan)     │
   └─────────┬──────────┘                        └─────────┬──────────┘
             │                                             │
   split deterministik (sort comment_id, seed=42, 70/20/10) — SAMA untuk semua
             │                                             │
   ┌─────────┴───────────┐                                 │
   ▼                     ▼                                 ▼
╔═══════════════╗  ╔═══════════════════╗      ╔══════════════════════════╗
║  TAHAP 2a     ║  ║  TAHAP 2b         ║      ║  TAHAP 2c                ║
║  SVM sklearn  ║  ║  SVM Spark MLlib  ║      ║  IndoBERT fine-tune      ║
║  (LOKAL)      ║  ║  (SPARK)          ║      ║  (LOKAL)                 ║
║               ║  ║                   ║      ║                          ║
║ TF-IDF        ║  ║ CountVectorizer   ║      ║ indobert-base-p1         ║
║ sublinear+L2  ║  ║ →SublinearTF→IDF  ║      ║ 4 epoch, lr 2e-5         ║
║ LinearSVC     ║  ║ →L2→OneVsRest     ║      ║ PyTorch (CPU)            ║
║ GridSearchCV  ║  ║ LinearSVC         ║      ║                          ║
║               ║  ║                   ║      ║                          ║
║ 1 proses      ║  ║ Spark Master +    ║      ║ 1 proses Python          ║
║ Python        ║  ║ Workers           ║      ║ (transformers Trainer)   ║
╚═══════╤═══════╝  ╚═════════╤═════════╝      ╚════════════╤═════════════╝
        │                    │                             │
        ▼                    ▼                             ▼
  macro-F1 0.669       macro-F1 0.607               macro-F1 0.683
  (baseline)           (varian Big Data)            (TERBAIK)
        │                    │                             │
        └────────────────────┴──────────────┬──────────────┘
                                             ▼
                              Perbandingan akhir 3 model
                              (svm_full14k_metrics.json +
                               indobert_metrics.json)
```

---

## 3. Kenapa IndoBERT & SVM sklearn TIDAK pakai Spark?

- **Spark unggul untuk data paralel** (transform jutaan baris, ETL, agregasi).
  Preprocessing 14k komentar = cocok → kita pakai cluster.
- **Training deep learning (IndoBERT)** butuh GPU/CPU dengan PyTorch — modelnya satu,
  bobotnya tidak dipecah ke banyak worker. Jadi single-node, bukan tugas Spark.
- **scikit-learn LinearSVC** juga single-node by design (in-memory).
- **SVM Spark MLlib** sengaja dibuat sebagai **pembanding "versi terdistribusi"** —
  untuk menunjukkan SVM yang sama, tapi diimplementasi di engine Big Data.
  Hasilnya sedikit lebih rendah (0.607) karena beda formula IDF & loss function
  (hinge vs squared-hinge), bukan karena Spark "lebih jelek".

---

## 4. Kalau ditanya "tadi yang di lokal itu apa artinya?"

> "Preprocessing-nya saya distribusikan ke Spark cluster (2 mesin lewat Tailscale)
> karena itu beban data paralel. Tapi training IndoBERT dan SVM baseline saya jalankan
> lokal, karena PyTorch dan scikit-learn memang single-node — bobot model tidak dipecah
> ke worker. Saya tetap bikin satu varian SVM di Spark MLlib supaya ada pembanding sisi
> Big Data-nya."

---

## 5. Status cluster saat ini

- Semua training & evaluasi **sudah selesai dan ter-commit**.
- Worker remote (rocky-server) sesi SSH-nya sudah ditutup → cluster sekarang
  tinggal node lokal. **Tidak masalah** karena tidak ada job yang sedang jalan.
- Mau hidupkan cluster lagi: `bash src/spark/cluster.sh start` (+ sambungkan rocky).
- Riwayat aplikasi yang sudah selesai tetap bisa dilihat lewat History Server
  (`bash src/spark/cluster.sh history` → http://localhost:18080), bukan dari Master :8080.
```

# YouTube Comment Sentiment Analysis (Big Data Pipeline)

## Ringkasan Proyek

Pipeline end-to-end untuk analisis sentimen komentar YouTube berbahasa Indonesia.
Sistem mendukung ingestion dari **banyak video sekaligus** (sekitar 20 video),
dengan pelacakan status per video, mekanisme resume, dan pemrosesan data
agregat lintas video menggunakan Apache Spark (PySpark).

## Arsitektur Alur Data

```
configs/video_urls.txt  (daftar ~20 URL YouTube)
        |
        v
[01_config.ipynb]
  - validasi semua URL
  - ekstrak videoId dari tiap URL
  - daftarkan job ke MongoDB (ingestion_jobs collection)
        |
        v
[02_ingestion.ipynb]  <-- loop per video dengan resume support
  - cek status job (skip jika sudah "completed")
  - youtube_ingest.py: ambil komentar via API + pagination
  - mongo_utils.py: simpan ke raw_comments (dengan video_id sebagai partisi logis)
  - ingestion_tracker.py: update status job (pending -> running -> completed/failed)
        |
        v
[03_preprocessing_spark.ipynb]
  - baca SEMUA raw_comments dari MongoDB (lintas video)
  - PySpark DataFrame transformations
  |-- Jalur A (SVM + TF-IDF): cleaning agresif + stemming PySastrawi
  |-- Jalur B (IndoBERT): cleaning minimal, morfologi terjaga
        |
        v
[04_export_labeling.ipynb]
  - gabungkan hasil dua jalur
  - ekspor ke JSONL / CSV / Parquet
  - siap untuk labeling manual atau hybrid
```

## Struktur Folder

```
youtube_sentiment_project/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── __init__.py
│   ├── config.py
│   └── video_urls.txt          <-- daftar URL video (satu per baris)
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── mongo_utils.py
│   ├── ingestion_tracker.py    <-- pelacak status ingestion per video
│   ├── youtube_ingest.py
│   ├── text_normalizer.py
│   ├── preprocess_spark.py
│   └── notebook_helpers.py
├── notebooks/
│   ├── 01_config.ipynb
│   ├── 02_ingestion.ipynb
│   ├── 03_preprocessing_spark.ipynb
│   └── 04_export_labeling.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
└── logs/
```

## Instalasi di Windows

### 1. Buat dan aktifkan virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Konfigurasi environment

```powershell
copy .env.example .env
```

Isi `.env`:

```env
YOUTUBE_API_KEY=your_api_key_here
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=youtube_sentiment
MONGO_COLLECTION_RAW=raw_comments
MONGO_COLLECTION_SVM=processed_svm
MONGO_COLLECTION_BERT=processed_bert
MONGO_COLLECTION_JOBS=ingestion_jobs
TARGET_COMMENT_COUNT=10000
```

### 4. Isi daftar video

Edit `configs/video_urls.txt`, satu URL per baris:

```
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://youtu.be/VIDEO_ID_3
# Baris yang diawali # adalah komentar dan akan diabaikan
```

### 5. Pastikan MongoDB berjalan

```powershell
net start MongoDB
```

### 6. Jalankan notebook secara berurutan

```
01_config.ipynb                  -- validasi URL dan daftarkan job
02_ingestion.ipynb               -- ambil komentar semua video
03_preprocessing_spark.ipynb     -- preprocessing dua jalur
04_export_labeling.ipynb         -- ekspor dataset siap labeling
```

## Mekanisme Resume Ingestion

Jika ingestion terputus di tengah jalan (quota habis, koneksi error, dsb.),
cukup jalankan ulang `02_ingestion.ipynb`. Video dengan status `completed`
otomatis dilewati, sehingga hanya video yang `pending` atau `failed` yang
diproses ulang.

Status per video tersimpan di MongoDB collection `ingestion_jobs`:

| Status      | Keterangan                                      |
|-------------|--------------------------------------------------|
| `pending`   | Belum diproses                                   |
| `running`   | Sedang diproses (atau crash tanpa update status) |
| `completed` | Berhasil selesai                                 |
| `failed`    | Gagal dengan error message                       |

## Skema Data

### ingestion_jobs (MongoDB)

```json
{
  "video_id": "dQw4w9WgXcQ",
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "video_title": "Judul Video",
  "status": "completed",
  "comment_count": 9847,
  "started_at": "2025-01-20T08:00:00",
  "completed_at": "2025-01-20T08:05:30",
  "error_message": null
}
```

### raw_comments (MongoDB)

```json
{
  "comment_id": "UgxABC123_xyz",
  "video_id": "dQw4w9WgXcQ",
  "text": "Mantap banget videonya bro!! 🔥",
  "published_at": "2024-01-15T10:23:45.000Z",
  "like_count": 42,
  "author_channel_id": "UCxxxxxxxxxxxx",
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "source_title": "Judul Video",
  "fetched_at": "2025-01-20T08:00:00.123456"
}
```

### Output Labeling

```json
{
  "comment_id": "UgxABC123_xyz",
  "video_id": "dQw4w9WgXcQ",
  "text_original": "Mantap banget videonya bro!! 🔥",
  "text_svm": "mantap video",
  "text_bert": "mantap banget videonya bro",
  "label": null
}
```

## Catatan Kompatibilitas

### Python 3.14.3
- PySastrawi: install dari source jika wheel belum tersedia:
  `pip install git+https://github.com/har07/PySastrawi.git`
- Hindari library yang bergantung pada `pkg_resources` lama.

### Apache Spark 4.1.x
- Butuh Java 11 atau 17. Set `JAVA_HOME` dengan benar.
- Set `PYSPARK_PYTHON` ke interpreter aktif:
  `set PYSPARK_PYTHON=.venv\Scripts\python.exe`
- MongoDB Spark Connector JAR: https://www.mongodb.com/try/download/spark-connector

### Jalur SVM vs IndoBERT
- **SVM + TF-IDF**: preprocessing agresif untuk bag-of-words yang bersih.
- **IndoBERT**: preprocessing minimal, jangan stem atau hapus stopword.

"""
configs/config.py
Konfigurasi terpusat. Semua nilai dibaca dari environment variable atau .env.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

VIDEO_URLS_FILE = _ROOT / "configs" / "video_urls.txt"


def load_video_urls() -> List[str]:
    """
    Baca daftar URL video dari configs/video_urls.txt.
    Baris kosong dan baris yang diawali '#' diabaikan.

    Returns:
        List URL YouTube yang valid secara format (belum divalidasi ke API).

    Raises:
        FileNotFoundError: jika video_urls.txt tidak ditemukan.
    """
    if not VIDEO_URLS_FILE.exists():
        raise FileNotFoundError(
            f"File daftar URL tidak ditemukan: {VIDEO_URLS_FILE}\n"
            "Buat file tersebut dan isi dengan URL video YouTube (satu per baris)."
        )
    urls = []
    for line in VIDEO_URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


class YoutubeConfig:
    API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
    BASE_URL: str = "https://www.googleapis.com/youtube/v3"
    COMMENTS_ENDPOINT: str = f"{BASE_URL}/commentThreads"
    TARGET_COMMENT_COUNT: int = int(os.getenv("TARGET_COMMENT_COUNT", "10000"))
    MAX_RESULTS_PER_PAGE: int = 100


class MongoConfig:
    URI: str = os.getenv("MONGO_URI", "")
    DB_NAME: str = os.getenv("MONGO_DB_NAME", "youtube_sentiment")
    COLLECTION_RAW: str = os.getenv("MONGO_COLLECTION_RAW", "raw_comments")
    COLLECTION_SVM: str = os.getenv("MONGO_COLLECTION_SVM", "processed_svm")
    COLLECTION_BERT: str = os.getenv("MONGO_COLLECTION_BERT", "processed_bert")
    COLLECTION_JOBS: str = os.getenv("MONGO_COLLECTION_JOBS", "ingestion_jobs")


class SparkConfig:
    APP_NAME: str = "YoutubeSentimentPipeline"
    MASTER: str = "local[*]"
    MONGO_CONNECTOR_JAR: str = os.getenv("MONGO_CONNECTOR_JAR", "")
    MONGO_SPARK_PACKAGE: str = (
        "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0"
    )


class PathConfig:
    ROOT: Path = _ROOT
    DATA_RAW: Path = _ROOT / "data" / "raw"
    DATA_PROCESSED: Path = _ROOT / "data" / "processed"
    OUTPUTS: Path = _ROOT / "outputs"
    LOGS: Path = _ROOT / "logs"

    # --- Tahap modeling ---
    LABELING: Path = _ROOT / "outputs" / "labeling"
    SPLITS: Path = _ROOT / "data" / "processed" / "splits"
    MODELS: Path = _ROOT / "outputs" / "models"
    REPORTS: Path = _ROOT / "outputs" / "reports"

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in [cls.DATA_RAW, cls.DATA_PROCESSED, cls.OUTPUTS, cls.LOGS,
                  cls.LABELING, cls.SPLITS, cls.MODELS, cls.REPORTS]:
            d.mkdir(parents=True, exist_ok=True)


class LabelStudioConfig:
    """Sumber label hasil anotasi manusia di Label Studio."""
    URL: str = os.getenv(
        "LABEL_STUDIO_URL",
        "https://raviarnan-jokowi-label-studio.hf.space",
    )
    # Token auth legacy dinonaktifkan untuk org ini; ekspor manual via UI
    # tetap jadi jalur utama. Token hanya dipakai bila API diaktifkan kembali.
    API_TOKEN: str = os.getenv("LABEL_STUDIO_API_TOKEN", "")
    PROJECT_ID: int = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))
    # Nama Choices di configs/label_studio_sentiment.xml
    FROM_NAME: str = "sentiment"
    # File JSON hasil tombol "Export" di UI Label Studio
    EXPORT_FILE: Path = _ROOT / "outputs" / "labeling" / "label_studio_export.json"


class ModelingConfig:
    """Parameter bersama untuk kedua jalur model (SVM & IndoBERT)."""
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

    # Rasio split — total dataset seimbang 1.000/kelas (lihat README modeling).
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.20
    TEST_RATIO: float = 0.10

    # Kosakata label. URUTAN menentukan id kelas (jangan diubah sembarangan).
    LABELS: List[str] = ["Negatif", "Netral", "Positif"]

    # Target jumlah contoh per kelas untuk dataset seimbang.
    TARGET_PER_CLASS: int = int(os.getenv("TARGET_PER_CLASS", "1000"))

    # IndoBERT (fine-tuning, dijalankan di Colab/Kaggle).
    INDOBERT_MODEL: str = os.getenv("INDOBERT_MODEL", "indobenchmark/indobert-base-p1")
    MAX_SEQ_LEN: int = int(os.getenv("MAX_SEQ_LEN", "128"))


class Config:
    youtube = YoutubeConfig
    mongo = MongoConfig
    spark = SparkConfig
    paths = PathConfig
    label_studio = LabelStudioConfig
    modeling = ModelingConfig
    load_video_urls = staticmethod(load_video_urls)

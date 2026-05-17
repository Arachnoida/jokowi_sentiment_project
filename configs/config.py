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

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in [cls.DATA_RAW, cls.DATA_PROCESSED, cls.OUTPUTS, cls.LOGS]:
            d.mkdir(parents=True, exist_ok=True)


class Config:
    youtube = YoutubeConfig
    mongo = MongoConfig
    spark = SparkConfig
    paths = PathConfig
    load_video_urls = staticmethod(load_video_urls)

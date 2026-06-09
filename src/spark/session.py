"""Builder SparkSession + helper path proyek.

Spark 4.x dipakai karena environment ini memakai Java 21 (Spark 3.5 tidak
didukung resmi di Java 21). Dipusatkan di sini agar semua skrip Spark memakai
konfigurasi yang sama (mode lokal, UI mati, log diredam).
"""
from __future__ import annotations

import pathlib

from pyspark.sql import SparkSession

# Partisi shuffle default Spark = 200, kelewat banyak untuk 14k baris di satu
# mesin (overhead task >> kerja). 8 sudah cukup dan jauh lebih cepat lokal.
DEFAULT_SHUFFLE_PARTITIONS = "8"


def project_root() -> pathlib.Path:
    """Akar repo: naik dari file ini sampai ketemu .git / configs."""
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "configs").exists():
            return p
    return here.parents[2]


def reports_dir() -> pathlib.Path:
    d = project_root() / "outputs" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parquet_dir() -> pathlib.Path:
    """Lokasi snapshot Parquet hasil ekspor Mongo (sumber baca Spark)."""
    d = project_root() / "data" / "spark_parquet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_spark(app_name: str, shuffle_partitions: str = DEFAULT_SHUFFLE_PARTITIONS) -> SparkSession:
    """SparkSession lokal yang tenang (tanpa banjir log/UI)."""
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

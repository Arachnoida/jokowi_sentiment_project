"""
archive/src/spark_leftovers.py

Sisa kode terkait Spark yang DIKELUARKAN dari kode aktif saat pindah ke alur
MongoDB + pandas. Disimpan untuk referensi — TIDAK dipakai pipeline aktif.

Asal:
- `SparkConfig` + `Config.spark`  <- dulu di configs/config.py
- `check_spark_session`, `dataframe_info`  <- dulu di src/notebook_helpers.py
"""

import os


class SparkConfig:
    APP_NAME: str = "YoutubeSentimentPipeline"
    MASTER: str = "local[*]"
    MONGO_CONNECTOR_JAR: str = os.getenv("MONGO_CONNECTOR_JAR", "")
    MONGO_SPARK_PACKAGE: str = (
        "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0"
    )


def check_spark_session(spark) -> bool:
    """Verifikasi SparkSession aktif."""
    try:
        print(f"\n[OK] SparkSession aktif")
        print(f"  Versi Spark : {spark.version}")
        print(f"  App Name    : {spark.sparkContext.appName}")
        print(f"  Master      : {spark.sparkContext.master}")
        return True
    except Exception as exc:
        print(f"\n[GAGAL] SparkSession tidak aktif: {exc}")
        return False


def dataframe_info(df, label: str = "DataFrame") -> None:
    """Tampilkan informasi ringkas Spark DataFrame."""
    print(f"\n--- {label} ---")
    print(f"Jumlah baris : {df.count()}")
    print(f"Kolom        : {df.columns}")
    df.printSchema()
    df.show(5, truncate=50)

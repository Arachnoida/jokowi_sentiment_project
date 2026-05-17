"""
src/preprocess_spark.py
Preprocessing komentar YouTube menggunakan Apache Spark (PySpark).
Mendukung dua jalur: SVM + TF-IDF dan IndoBERT.
Data lintas banyak video diproses sekaligus dalam satu pipeline Spark.
"""

from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType

from src.utils import setup_logger
from src.text_normalizer import (
    preprocess_svm_python,
    preprocess_bert_python,
    try_stem_sastrawi,
    tokenize,
)

logger = setup_logger("preprocess_spark")


def build_spark_session(
    app_name: str,
    master: str = "local[*]",
    mongo_package: Optional[str] = None,
    extra_configs: Optional[dict] = None,
) -> SparkSession:
    """Bangun SparkSession untuk local mode."""
    builder = SparkSession.builder.appName(app_name).master(master)
    builder = builder.config("spark.ui.enabled", "false")
    builder = builder.config("spark.sql.shuffle.partitions", "8")

    if mongo_package:
        builder = builder.config("spark.jars.packages", mongo_package)
    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession dibuat: {app_name} | {master}")
    return spark


def load_from_list_to_spark(
    spark: SparkSession,
    records: List[dict],
    select_columns: Optional[List[str]] = None,
) -> DataFrame:
    """Muat data dari list Python ke Spark DataFrame."""
    if not records:
        raise ValueError("records tidak boleh kosong")
    df = spark.createDataFrame(records)
    if select_columns:
        existing = [c for c in select_columns if c in df.columns]
        df = df.select(existing)
    return df


# ---- UDF -------------------------------------------------------------------

@F.udf(returnType=StringType())
def udf_preprocess_svm(text):
    if text is None:
        return ""
    return preprocess_svm_python(str(text))


@F.udf(returnType=StringType())
def udf_preprocess_bert(text):
    if text is None:
        return ""
    return preprocess_bert_python(str(text))


@F.udf(returnType=StringType())
def udf_stem_sastrawi(text):
    if text is None:
        return ""
    return try_stem_sastrawi(str(text))


@F.udf(returnType=ArrayType(StringType()))
def udf_tokenize(text):
    if text is None:
        return []
    return tokenize(str(text))


# ---- Fungsi preprocessing --------------------------------------------------

def preprocess_svm_path(df: DataFrame) -> DataFrame:
    """
    Jalur A: Preprocessing SVM + TF-IDF.
    Input DataFrame harus memiliki kolom: comment_id, video_id, text.
    """
    logger.info("Menjalankan Jalur A (SVM + TF-IDF)...")
    df_svm = df.select("comment_id", "video_id", "text")
    df_svm = df_svm.withColumn("text_pre_stem", udf_preprocess_svm(F.col("text")))
    df_svm = df_svm.withColumn("text_svm", udf_stem_sastrawi(F.col("text_pre_stem")))
    df_svm = df_svm.withColumn("tokens_svm", udf_tokenize(F.col("text_svm")))
    df_svm = df_svm.drop("text_pre_stem")
    df_svm = df_svm.filter(F.length(F.col("text_svm")) > 0)
    logger.info(f"Jalur SVM selesai: {df_svm.count()} komentar valid.")
    return df_svm


def preprocess_bert_path(df: DataFrame) -> DataFrame:
    """
    Jalur B: Preprocessing IndoBERT.
    Input DataFrame harus memiliki kolom: comment_id, video_id, text.
    """
    logger.info("Menjalankan Jalur B (IndoBERT)...")
    df_bert = df.select("comment_id", "video_id", "text")
    df_bert = df_bert.withColumn("text_bert", udf_preprocess_bert(F.col("text")))
    df_bert = df_bert.filter(F.length(F.col("text_bert")) > 0)
    logger.info(f"Jalur BERT selesai: {df_bert.count()} komentar valid.")
    return df_bert


def merge_preprocessed_paths(
    df_original: DataFrame,
    df_svm: DataFrame,
    df_bert: DataFrame,
) -> DataFrame:
    """Gabungkan hasil dua jalur menjadi satu DataFrame siap labeling."""
    df_base = df_original.select(
        F.col("comment_id"),
        F.col("video_id"),
        F.col("text").alias("text_original"),
    )
    df_merged = (
        df_base
        .join(df_svm.select("comment_id", "text_svm", "tokens_svm"),
              on="comment_id", how="left")
        .join(df_bert.select("comment_id", "text_bert"),
              on="comment_id", how="left")
    )
    df_merged = df_merged.withColumn("label", F.lit(None).cast("string"))
    return df_merged

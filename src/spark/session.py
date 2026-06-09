"""Builder SparkSession + helper path proyek.

Spark 4.x dipakai karena environment ini memakai Java 21 (Spark 3.5 tidak
didukung resmi di Java 21). Dipusatkan di sini agar semua skrip Spark memakai
konfigurasi yang sama (mode lokal, log diredam).

Spark Web UI (Application UI) tersedia di http://localhost:4040 selama aplikasi
hidup. Default ON; matikan dengan env ``SPARK_UI=0``. Port via ``SPARK_UI_PORT``.
Karena skrip cepat selesai (lalu spark.stop() -> UI hilang), set env
``SPARK_HOLD=1`` agar skrip menahan sesi di akhir supaya UI sempat dibuka.
"""
from __future__ import annotations

import os
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
    """SparkSession lokal yang tenang. Web UI ON kecuali env SPARK_UI=0.

    Master default ``local[*]``. Set env ``SPARK_MASTER`` (mis.
    ``spark://127.0.0.1:7077``) untuk submit ke Standalone cluster — lihat
    ``src/spark/cluster.sh``. Saat memakai cluster, executor (JVM Worker
    terpisah) harus bisa ``import src`` dan memakai Python venv yang sama, jadi
    PYSPARK_PYTHON + executorEnv.PYTHONPATH diset otomatis.
    """
    ui_enabled = os.environ.get("SPARK_UI", "1") != "0"
    ui_port = os.environ.get("SPARK_UI_PORT", "4040")
    master = os.environ.get("SPARK_MASTER", "local[*]")
    is_cluster = not master.startswith("local")

    # Arrow mempercepat transfer JVM<->Python (collect/toPandas/createDataFrame),
    # TAPI memaksa setiap executor men-spawn Python worker -> pada cluster
    # ANTAR-MESIN itu mengharuskan interpreter+lib Python identik di SETIAP node.
    # Job SVM tak punya Python-UDF, jadi dengan Arrow OFF ia jadi 100% JVM di
    # executor (node remote cukup Spark+Java). Maka: Arrow ON hanya di local[*].
    arrow_enabled = "false" if is_cluster else "true"

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "true" if ui_enabled else "false")
        .config("spark.ui.port", ui_port)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.execution.arrow.pyspark.enabled", arrow_enabled)
    )

    if not master.startswith("local"):
        # Cluster: executor butuh interpreter venv + akar proyek di PYTHONPATH
        # agar UDF (src.text_normalizer, Sastrawi) bisa diimpor di Worker.
        venv_py = project_root() / ".venv" / "bin" / "python"
        os.environ.setdefault("PYSPARK_PYTHON", str(venv_py))
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", str(venv_py))
        builder = builder.config("spark.executorEnv.PYTHONPATH", str(project_root()))
        # Cluster ANTAR-MESIN: executor remote harus bisa menghubungi balik driver.
        # Set SPARK_DRIVER_HOST=<IP LAN mesin ini> saat submit (lihat join_worker.sh).
        driver_host = os.environ.get("SPARK_DRIVER_HOST")
        if driver_host:
            builder = builder.config("spark.driver.host", driver_host)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    if ui_enabled:
        port = spark.sparkContext.uiWebUrl.split(":")[-1]
        print(f"[Spark UI] http://localhost:{port}  (master={master})")
    return spark


def hold_for_ui(spark: SparkSession) -> None:
    """Tahan aplikasi di akhir skrip agar Spark UI (:4040) sempat dibuka.

    Aktif hanya bila env ``SPARK_HOLD=1``. Tanpa ini, ``spark.stop()`` langsung
    menutup UI begitu skrip selesai (job di local[*] cuma beberapa detik).
    """
    if os.environ.get("SPARK_HOLD", "0") != "1":
        return
    url = spark.sparkContext.uiWebUrl
    print(f"\n[Spark UI] Sesi DITAHAN. Buka {url} (tab Jobs/Stages/Executors/SQL).")
    try:
        input("Tekan ENTER untuk menutup SparkSession...")
    except (EOFError, KeyboardInterrupt):
        pass

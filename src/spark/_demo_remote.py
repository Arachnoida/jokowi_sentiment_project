"""Demo: paksa seluruh task Spark berjalan di executor REMOTE (rocky-server).

Sengaja TANPA membaca file lokal (murni komputasi spark.range) supaya tidak
bergantung pada replikasi data — buktinya bersih: task benar-benar dieksekusi
di mesin lain lewat Tailscale. Jalankan dengan worker lokal dimatikan sehingga
satu-satunya executor ada di rocky.

  SPARK_MASTER=spark://100.95.198.108:7077 SPARK_DRIVER_HOST=100.95.198.108 \\
  SPARK_LOCAL_IP=100.95.198.108 .venv/bin/python -m src.spark._demo_remote
"""
from __future__ import annotations

import urllib.request
import json

from pyspark.sql import functions as F

from src.spark.session import get_spark

spark = get_spark("demo-remote-rocky", shuffle_partitions="16")
sc = spark.sparkContext

# Komputasi terdistribusi: 8 juta baris, 16 partisi, agregasi (memicu shuffle).
df = spark.range(0, 8_000_000, numPartitions=16).withColumn("g", F.col("id") % 7)
res = df.groupBy("g").agg(F.sum("id").alias("s"), F.count("*").alias("c")).orderBy("g")
rows = res.collect()
print("\n=== HASIL AGREGASI (dihitung di executor remote) ===")
for r in rows:
    print(f"  g={r['g']}  count={r['c']:>9}  sum={r['s']}")

# Lapor di mana task dieksekusi (host executor).
aid = sc.applicationId
ui = sc.uiWebUrl  # http://<driver>:4040
try:
    data = json.load(urllib.request.urlopen(f"{ui}/api/v1/applications/{aid}/executors"))
    print("\n=== EXECUTOR (pembuktian mesin) ===")
    for e in data:
        host = e.get("hostPort", "").split(":")[0]
        if e.get("id") == "driver":
            tag = "driver (mesin kamu)"
        elif host.startswith("100.79"):
            tag = "ROCKY-SERVER (remote, via Tailscale)"
        else:
            tag = host
        print(f"  exec {e.get('id'):>4} @ {host:14} {tag:38} "
              f"tasks_selesai={e.get('completedTasks')}")
except Exception as exc:  # pragma: no cover - diagnostik saja
    print("(gagal baca executors API:", exc, ")")

spark.stop()

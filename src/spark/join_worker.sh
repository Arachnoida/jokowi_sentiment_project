#!/usr/bin/env bash
# Sambungkan mesin INI sebagai Worker ke Master Spark di mesin LAIN (tugas kelompok).
#
# Dijalankan di KOMPUTER TEMAN (bukan mesin Master). Worker akan daftar ke Master
# lewat jaringan, lalu ikut mengeksekusi task -> benar-benar terdistribusi antar-mesin.
#
# Prasyarat di mesin ini (teman):
#   1. Java 17/21 + pip install "pyspark>=4.0,<4.1"  (versi Spark HARUS sama: 4.0.x)
#   2. Satu jaringan/LAN dengan Master (atau IP Master terjangkau).
#   3. Untuk job ber-UDF Python (preprocess/eda): butuh PySastrawi + folder src/ juga.
#      Untuk job SVM (train_svm_spark): TIDAK perlu UDF Python -> cukup Spark+Java.
#
# Pakai:
#   MASTER=spark://192.168.1.50:7077 bash join_worker.sh
#   # opsional atur resource & IP sendiri:
#   MASTER=spark://192.168.1.50:7077 WORKER_CORES=4 WORKER_MEM=4g SPARK_LOCAL_IP=192.168.1.77 bash join_worker.sh
set -euo pipefail

MASTER="${MASTER:-${1:-}}"
if [ -z "$MASTER" ]; then
  echo "ERROR: set MASTER, mis: MASTER=spark://192.168.1.50:7077 bash join_worker.sh"
  exit 1
fi

# Cari SPARK_HOME dari pyspark yang terpasang (venv lokal kalau ada, lalu python sistem).
PY="${PYSPARK_PYTHON:-python3}"
[ -x "./.venv/bin/python" ] && PY="./.venv/bin/python"
export SPARK_HOME="$("$PY" -c 'import os,pyspark;print(os.path.dirname(pyspark.__file__))')"

# IP mesin ini yang dijangkau Master (auto-deteksi IP LAN pertama; override via env).
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
WORKER_CORES="${WORKER_CORES:-4}"
WORKER_MEM="${WORKER_MEM:-4g}"

echo "SPARK_HOME   : $SPARK_HOME"
echo "IP mesin ini : $SPARK_LOCAL_IP"
echo "Master       : $MASTER"
echo "Resource     : $WORKER_CORES core / $WORKER_MEM"
echo "Menyambung sebagai Worker... (Ctrl+C untuk berhenti)"

# Foreground supaya teman lihat log & bisa Ctrl+C. Worker akan muncul di Master UI :8080.
exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker "$MASTER" \
  --cores "$WORKER_CORES" --memory "$WORKER_MEM"

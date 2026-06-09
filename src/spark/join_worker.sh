#!/usr/bin/env bash
# Sambungkan mesin INI sebagai Worker ke Master Spark di mesin LAIN (tugas kelompok).
#
# Dijalankan di KOMPUTER TEMAN (bukan mesin Master). Worker akan daftar ke Master
# lewat jaringan, lalu ikut mengeksekusi task -> benar-benar terdistribusi antar-mesin.
# Mendukung dua jenis jaringan:
#   - LAN/WiFi sama  -> pakai IP 192.168.x / 10.x
#   - Tailscale (VPN mesh, beda lokasi pun bisa) -> pakai IP 100.x. Skrip ini
#     OTOMATIS memilih IP Tailscale mesin ini bila Master beralamat 100.x.
#
# Prasyarat di mesin ini (teman):
#   1. Java 17/21 + pip install "pyspark>=4.0,<4.1"  (versi Spark HARUS sama: 4.0.x)
#   2. Master terjangkau: satu LAN, ATAU satu tailnet Tailscale.
#   3. Untuk job ber-UDF Python (preprocess/eda): butuh PySastrawi + folder src/ juga.
#      Untuk job SVM (train_svm_spark): TIDAK perlu UDF Python -> cukup Spark+Java.
#   4. Job yang membaca Parquet (SVM) perlu data di path absolut yang SAMA di mesin
#      ini -> jalankan dulu `sync_data_to_worker.sh` dari koordinator (no-HDFS).
#
# Pakai:
#   MASTER=spark://192.168.1.50:7077 bash join_worker.sh          # LAN
#   MASTER=spark://100.95.198.108:7077 bash join_worker.sh        # Tailscale (auto IP 100.x)
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

# IP mesin ini yang dijangkau Master. Bila Master beralamat Tailscale (100.x) dan
# IP belum di-override, pilih IP Tailscale mesin ini (bukan IP LAN dari hostname -I,
# yang biasanya 192.168.x dan TAK terjangkau lewat tailnet).
if [ -z "${SPARK_LOCAL_IP:-}" ]; then
  master_host="${MASTER#spark://}"; master_host="${master_host%%:*}"
  if [[ "$master_host" == 100.* ]] && command -v tailscale >/dev/null 2>&1; then
    SPARK_LOCAL_IP="$(tailscale ip -4 2>/dev/null | head -1)"
  fi
  SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
fi
export SPARK_LOCAL_IP
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

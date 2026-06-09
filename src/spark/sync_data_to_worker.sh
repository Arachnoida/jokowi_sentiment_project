#!/usr/bin/env bash
# Replikasi snapshot Parquet ke mesin Worker pada path absolut yang SAMA.
#
# Kenapa perlu: cluster Spark Standalone TANPA HDFS/S3 tidak punya storage bersama.
# Saat `spark.read.parquet(path)`, task baca dijalankan di executor masing-masing,
# dan tiap executor membuka `path` dari filesystem LOKALNYA. Jadi file harus ada di
# SETIAP node di path identik, kalau tidak: `FileNotFoundException` di node remote.
#
# Job SVM (`train_svm_spark`) membaca `data/spark_parquet/features_spark.parquet`,
# jadi file itu harus direplikasi. Job demo `_demo_remote` murni komputasi (tak baca
# file) -> tak perlu sync.
#
# Jalankan dari mesin KOORDINATOR (yang punya data hasil export+preprocess):
#   bash src/spark/sync_data_to_worker.sh ravi@rocky-server
#   bash src/spark/sync_data_to_worker.sh ravi@host-a ravi@host-b   # banyak worker
#
# Catatan: path absolut repo di mesin worker harus sama dengan di koordinator
# (mis. /home/<user>/Projects/jokowi_sentiment_project). Skrip membuat folder bila perlu.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Pakai: bash src/spark/sync_data_to_worker.sh user@host [user@host ...]"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBDIR="data/spark_parquet"
SRC="$ROOT/$SUBDIR"
[ -d "$SRC" ] || { echo "ERROR: $SRC belum ada — jalankan dulu: python -m src.spark.export_mongo && python -m src.spark.preprocess_spark"; exit 1; }

for host in "$@"; do
  echo ">>> sync $SUBDIR -> $host:$ROOT/$SUBDIR"
  # pastikan tar ada di remote (Rocky/RHEL minimal kadang tanpa tar) + buat folder
  ssh "$host" "command -v tar >/dev/null 2>&1 || (sudo dnf install -y tar >/dev/null 2>&1 || sudo apt-get install -y tar >/dev/null 2>&1) || true; mkdir -p '$ROOT/data'"
  # tar-pipe lewat SSH (kompatibel Tailscale SSH yang tak dukung scp/sftp)
  tar -C "$ROOT/data" -cf - "$(basename "$SUBDIR")" | ssh "$host" "tar -C '$ROOT/data' -xf - && echo '    OK: $host'"
done
echo "Selesai. Submit job SVM dari koordinator seperti biasa."

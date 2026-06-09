#!/usr/bin/env bash
# Kelola Spark Standalone cluster LOKAL (Master + N Worker) untuk demo Big Data.
#
# pip-pyspark 4.x tidak menyertakan start-master.sh/start-worker.sh, jadi Master &
# Worker diluncurkan langsung via `bin/spark-class` (persis yang dilakukan skrip
# resmi di balik layar). Semua di loopback 127.0.0.1 (satu mesin).
#
#   Master UI : http://localhost:8080   (daftar Worker node -> screenshot skripsi)
#   Worker UI : http://localhost:8081, 8082, ...  (satu per worker)
#   Master URL: spark://127.0.0.1:7077
#
# Bisa BANYAK worker (tiap worker = 1 proses JVM, daftar ke Master yang sama):
#   WORKER_COUNT=3 bash src/spark/cluster.sh start    # 3 Worker node di :8080
#   bash src/spark/cluster.sh status
#   bash src/spark/cluster.sh stop
#
# Resource bersifat PER-worker. Total core = WORKER_COUNT * WORKER_CORES; jangan
# melebihi core fisik (mesin ini 16). Contoh aman: 4 worker x 4 core = 16.
#   WORKER_COUNT=4 WORKER_CORES=4 WORKER_MEM=2g bash src/spark/cluster.sh start
#
# Submit job ke cluster (bukan local[*]):
#   SPARK_MASTER=spark://127.0.0.1:7077 python -m src.spark.train_svm_spark
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
export SPARK_HOME="$("$VENV_PY" -c 'import os,pyspark;print(os.path.dirname(pyspark.__file__))')"

# Host Master. Default 127.0.0.1 (cuma worker lokal). Untuk cluster ANTAR-MESIN
# (tugas kelompok), set ke IP LAN mesin ini, mis:
#   SPARK_MASTER_HOST=192.168.1.50 bash src/spark/cluster.sh start
# lalu teman menjalankan join_worker.sh menunjuk spark://192.168.1.50:7077.
MASTER_HOST="${SPARK_MASTER_HOST:-127.0.0.1}"
# SPARK_LOCAL_IP harus IP yang bisa dijangkau worker remote -> samakan dgn MASTER_HOST.
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-$MASTER_HOST}"

MASTER_PORT=7077
MASTER_WEBUI=8080
WORKER_WEBUI_BASE=8081
WORKER_COUNT="${WORKER_COUNT:-1}"     # jumlah worker
WORKER_CORES="${WORKER_CORES:-4}"     # core PER worker
WORKER_MEM="${WORKER_MEM:-2g}"        # memori PER worker
MASTER_URL="spark://${MASTER_HOST}:${MASTER_PORT}"

RUN="$ROOT/logs/spark"
mkdir -p "$RUN"
MPID="$RUN/master.pid"

_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

_spark_class() {  # launch a daemon class in background, write pid
  local pidfile="$1" log="$2"; shift 2
  nohup "$SPARK_HOME/bin/spark-class" "$@" >"$log" 2>&1 &
  echo $! >"$pidfile"
}

start() {
  local ncpu; ncpu="$("$VENV_PY" -c 'import os;print(os.cpu_count())')"
  local total=$(( WORKER_COUNT * WORKER_CORES ))
  if [ "$total" -gt "$ncpu" ]; then
    echo "PERINGATAN: total core ($WORKER_COUNT x $WORKER_CORES = $total) > core fisik ($ncpu) -> oversubscribe."
  fi

  if _alive "$MPID"; then echo "Master sudah jalan (pid $(cat "$MPID"))."; else
    _spark_class "$MPID" "$RUN/master.log" \
      org.apache.spark.deploy.master.Master \
      --host "$MASTER_HOST" --port "$MASTER_PORT" --webui-port "$MASTER_WEBUI"
    echo "Master start -> $MASTER_URL (UI http://localhost:$MASTER_WEBUI)"
  fi
  sleep 6

  for i in $(seq 1 "$WORKER_COUNT"); do
    local wpid="$RUN/worker-$i.pid"
    local port=$(( WORKER_WEBUI_BASE + i - 1 ))
    local wdir="$RUN/work-$i"
    mkdir -p "$wdir"
    if _alive "$wpid"; then echo "Worker $i sudah jalan (pid $(cat "$wpid"))."; continue; fi
    _spark_class "$wpid" "$RUN/worker-$i.log" \
      org.apache.spark.deploy.worker.Worker "$MASTER_URL" \
      --cores "$WORKER_CORES" --memory "$WORKER_MEM" --webui-port "$port" --work-dir "$wdir"
    echo "Worker $i start -> ${WORKER_CORES} core / ${WORKER_MEM} (UI http://localhost:$port)"
  done
  sleep 4
  echo ""
  echo "Cluster siap: $WORKER_COUNT worker, $total core total. Submit job:"
  echo "  SPARK_MASTER=$MASTER_URL $VENV_PY -m src.spark.train_svm_spark"
}

stop() {
  for p in "$RUN"/worker-*.pid "$MPID"; do
    [ -e "$p" ] || continue
    if _alive "$p"; then kill "$(cat "$p")" 2>/dev/null || true; echo "Stop pid $(cat "$p") ($(basename "$p"))"; fi
    rm -f "$p"
  done
  echo "Cluster dimatikan."
}

status() {
  _alive "$MPID" && echo "Master: UP (pid $(cat "$MPID"))" || echo "Master: DOWN"
  local up=0
  for p in "$RUN"/worker-*.pid; do
    [ -e "$p" ] || continue
    if _alive "$p"; then up=$((up+1)); fi
  done
  echo "Worker UP: $up"
  echo -n "Master UI :$MASTER_WEBUI -> "; curl -s -o /dev/null -w "HTTP %{http_code}\n" "http://localhost:$MASTER_WEBUI/" || echo "tak terjangkau"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 2; start ;;
  status) status ;;
  *) echo "Pakai: [WORKER_COUNT=n] bash src/spark/cluster.sh {start|stop|restart|status}"; exit 1 ;;
esac

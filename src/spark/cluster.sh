#!/usr/bin/env bash
# Kelola Spark Standalone cluster LOKAL (Master + 1 Worker) untuk demo Big Data.
#
# pip-pyspark 4.x tidak menyertakan start-master.sh/start-worker.sh, jadi Master &
# Worker diluncurkan langsung via `bin/spark-class` (persis yang dilakukan skrip
# resmi di balik layar). Semua di loopback 127.0.0.1 (satu mesin).
#
#   Master UI : http://localhost:8080   (daftar Worker node -> screenshot skripsi)
#   Worker UI : http://localhost:8081
#   Master URL: spark://127.0.0.1:7077
#
# Pakai:
#   bash src/spark/cluster.sh start     # nyalakan Master + Worker
#   bash src/spark/cluster.sh status    # cek proses + UI
#   bash src/spark/cluster.sh stop      # matikan keduanya
#
# Lalu submit job ke cluster (bukan local[*]):
#   SPARK_MASTER=spark://127.0.0.1:7077 python -m src.spark.train_svm_spark
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
export SPARK_HOME="$("$VENV_PY" -c 'import os,pyspark;print(os.path.dirname(pyspark.__file__))')"
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"

MASTER_HOST=127.0.0.1
MASTER_PORT=7077
MASTER_WEBUI=8080
WORKER_WEBUI=8081
WORKER_CORES="${WORKER_CORES:-4}"
WORKER_MEM="${WORKER_MEM:-2g}"
MASTER_URL="spark://${MASTER_HOST}:${MASTER_PORT}"

RUN="$ROOT/logs/spark"
mkdir -p "$RUN"
MPID="$RUN/master.pid"
WPID="$RUN/worker.pid"

_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

_spark_class() {  # launch a daemon class in background, write pid
  local pidfile="$1" log="$2"; shift 2
  nohup "$SPARK_HOME/bin/spark-class" "$@" >"$log" 2>&1 &
  echo $! >"$pidfile"
}

start() {
  if _alive "$MPID"; then echo "Master sudah jalan (pid $(cat "$MPID"))."; else
    _spark_class "$MPID" "$RUN/master.log" \
      org.apache.spark.deploy.master.Master \
      --host "$MASTER_HOST" --port "$MASTER_PORT" --webui-port "$MASTER_WEBUI"
    echo "Master start -> $MASTER_URL (UI http://localhost:$MASTER_WEBUI)"
  fi
  sleep 6
  if _alive "$WPID"; then echo "Worker sudah jalan (pid $(cat "$WPID"))."; else
    _spark_class "$WPID" "$RUN/worker.log" \
      org.apache.spark.deploy.worker.Worker "$MASTER_URL" \
      --cores "$WORKER_CORES" --memory "$WORKER_MEM" --webui-port "$WORKER_WEBUI"
    echo "Worker start -> ${WORKER_CORES} core / ${WORKER_MEM} (UI http://localhost:$WORKER_WEBUI)"
  fi
  sleep 4
  echo ""
  echo "Cluster siap. Submit job:"
  echo "  SPARK_MASTER=$MASTER_URL $VENV_PY -m src.spark.train_svm_spark"
}

stop() {
  for p in "$WPID" "$MPID"; do
    if _alive "$p"; then kill "$(cat "$p")" 2>/dev/null || true; echo "Stop pid $(cat "$p")"; fi
    rm -f "$p"
  done
  echo "Cluster dimatikan."
}

status() {
  _alive "$MPID" && echo "Master: UP (pid $(cat "$MPID"))" || echo "Master: DOWN"
  _alive "$WPID" && echo "Worker: UP (pid $(cat "$WPID"))" || echo "Worker: DOWN"
  echo -n "Master UI :$MASTER_WEBUI -> "; curl -s -o /dev/null -w "HTTP %{http_code}\n" "http://localhost:$MASTER_WEBUI/" || echo "tak terjangkau"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 2; start ;;
  status) status ;;
  *) echo "Pakai: bash src/spark/cluster.sh {start|stop|restart|status}"; exit 1 ;;
esac

#!/usr/bin/env bash
#
# Jalankan Label Studio di balik tunnel ngrok dengan setting CSRF/host yang benar.
# Mengatasi error "Forbidden (403) CSRF verification failed" saat sign up lewat ngrok.
#
# Pakai:  ./start_label_studio_ngrok.sh [PORT]    (default PORT=8080)
# Syarat: 'ngrok' dan 'label-studio' sudah terinstall, dan ngrok authtoken sudah diset
#         (ngrok config add-authtoken <TOKEN>).

set -euo pipefail

PORT="${1:-8080}"
NGROK_API="http://127.0.0.1:4040/api/tunnels"
NGROK_PID=""

cleanup() {
  # Hanya matikan ngrok jika skrip ini yang menjalankannya.
  [ -n "${NGROK_PID}" ] && kill "${NGROK_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# 1. Pastikan ngrok berjalan (kalau belum, jalankan di background).
if ! curl -s "${NGROK_API}" >/dev/null 2>&1; then
  echo "==> Menjalankan ngrok di port ${PORT}..."
  ngrok http "${PORT}" --log=stdout >/tmp/ngrok.log 2>&1 &
  NGROK_PID=$!
fi

# 2. Ambil URL publik HTTPS dari API lokal ngrok (retry sampai ~15 detik).
echo "==> Mengambil URL publik ngrok..."
URL=""
for _ in $(seq 1 15); do
  URL=$(curl -s "${NGROK_API}" \
    | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(next((x['public_url'] for x in t if x['public_url'].startswith('https')), ''))" 2>/dev/null || true)
  [ -n "${URL}" ] && break
  sleep 1
done

if [ -z "${URL}" ]; then
  echo "ERROR: Gagal mendapatkan URL ngrok. Pastikan ngrok berjalan & authtoken sudah diset." >&2
  exit 1
fi

echo "==> URL ngrok: ${URL}"
echo "==> Buka URL itu di browser untuk sign up / login."

# 3. Jalankan Label Studio dengan CSRF & host yang benar.
export CSRF_TRUSTED_ORIGINS="${URL}"
export LABEL_STUDIO_HOST="${URL}"
echo "==> Menjalankan Label Studio (Ctrl+C untuk berhenti)..."
label-studio start --port "${PORT}"

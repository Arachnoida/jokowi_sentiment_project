# Jalankan Label Studio di balik tunnel ngrok dengan setting CSRF/host yang benar (Windows).
# Mengatasi error "Forbidden (403) CSRF verification failed" saat sign up lewat ngrok.
#
# Pakai:  powershell -ExecutionPolicy Bypass -File .\start_label_studio_ngrok.ps1 [-Port 8080]
# Syarat: 'ngrok' dan 'label-studio' ada di PATH, dan ngrok authtoken sudah diset
#         (ngrok config add-authtoken <TOKEN>).

param(
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$ngrokApi = "http://127.0.0.1:4040/api/tunnels"
$ngrokProc = $null

# 1. Pastikan ngrok berjalan (kalau belum, jalankan di background).
try {
    Invoke-RestMethod -Uri $ngrokApi -TimeoutSec 2 | Out-Null
    Write-Host "==> ngrok sudah berjalan."
} catch {
    Write-Host "==> Menjalankan ngrok di port $Port..."
    $ngrokProc = Start-Process -FilePath "ngrok" -ArgumentList "http", "$Port" -PassThru -WindowStyle Hidden
}

# 2. Ambil URL publik HTTPS dari API lokal ngrok (retry sampai ~15 detik).
Write-Host "==> Mengambil URL publik ngrok..."
$url = $null
for ($i = 0; $i -lt 15; $i++) {
    try {
        $tunnels = (Invoke-RestMethod -Uri $ngrokApi -TimeoutSec 2).tunnels
        $url = ($tunnels | Where-Object { $_.public_url -like "https*" } | Select-Object -First 1).public_url
        if ($url) { break }
    } catch { }
    Start-Sleep -Seconds 1
}

if (-not $url) {
    Write-Error "Gagal mendapatkan URL ngrok. Pastikan ngrok berjalan & authtoken sudah diset."
    if ($ngrokProc) { Stop-Process -Id $ngrokProc.Id -ErrorAction SilentlyContinue }
    exit 1
}

Write-Host "==> URL ngrok: $url"
Write-Host "==> Buka URL itu di browser untuk sign up / login."

# 3. Jalankan Label Studio dengan CSRF & host yang benar.
$env:CSRF_TRUSTED_ORIGINS = $url
$env:LABEL_STUDIO_HOST = $url
Write-Host "==> Menjalankan Label Studio (Ctrl+C untuk berhenti)..."
try {
    if (Get-Command label-studio -ErrorAction SilentlyContinue) {
        # 'label-studio' ada di PATH.
        label-studio start --port $Port
    } else {
        # Tidak di PATH -> panggil lewat Python (cari 'python' atau 'py').
        Write-Host "==> 'label-studio' tidak ada di PATH; mencoba lewat Python..."
        $pythonCmd = "python"
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            if (Get-Command py -ErrorAction SilentlyContinue) { $pythonCmd = "py" }
        }
        & $pythonCmd -c "import label_studio" 2>$null
        if ($LASTEXITCODE -eq 0) {
            & $pythonCmd -c "import sys; from label_studio.server import main; sys.argv=['label-studio','start','--port','$Port']; sys.exit(main())"
        } else {
            Write-Error "Label Studio belum terinstall di Python ini. Jalankan dulu:  pip install -U label-studio"
        }
    }
} finally {
    # Matikan ngrok hanya jika skrip ini yang menjalankannya.
    if ($ngrokProc) { Stop-Process -Id $ngrokProc.Id -ErrorAction SilentlyContinue }
}

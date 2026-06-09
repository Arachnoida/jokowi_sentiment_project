# 🖥️🖥️ Spark Cluster Antar-Mesin — Panduan & Troubleshooting (Tugas Kelompok)

Tujuan: jalankan **satu Master** (di mesin koordinator) + **beberapa Worker** dari
komputer anggota lain lewat LAN, lalu submit job SVM yang tereksekusi terdistribusi.

> **Istilah:** `MASTER_IP` = IP LAN mesin yang menjalankan Master (contoh di repo ini
> `192.168.1.50`). Ganti dengan IP mesin koordinator kalian.

---

## ✅ Checklist prasyarat (SEMUA mesin)

- [ ] **Satu jaringan/LAN sama** (WiFi yang sama paling gampang). Bukan hotspot terisolasi.
- [ ] **Java 17 atau 21** terpasang → `java -version`.
- [ ] **PySpark versi SAMA persis** di semua mesin → `python -c "import pyspark; print(pyspark.__version__)"` harus `4.0.x` di semua. **Beda versi = pasti gagal.**
- [ ] Mesin koordinator tahu **IP LAN-nya** → `hostname -I` (ambil yang `192.168.x.x`/`10.x.x.x`).
- [ ] (Khusus job ber-UDF) PySastrawi + folder `src/` ada di mesin Worker — lihat catatan di bawah.

---

## 🚀 Langkah jalan (urut)

### 1. Mesin koordinator (Master)
```bash
# bind Master ke IP LAN (WAJIB; jangan 127.0.0.1 atau worker remote tak bisa konek)
SPARK_MASTER_HOST=192.168.1.50 WORKER_COUNT=1 bash src/spark/cluster.sh start
```
Buka `http://localhost:8080` → harus muncul **Master ALIVE**.

### 2. Mesin tiap anggota (Worker)
```bash
# di folder repo, MASTER = spark://<MASTER_IP>:7077
MASTER=spark://192.168.1.50:7077 bash src/spark/join_worker.sh
```
Cek di `:8080` mesin koordinator → Worker baru muncul di tabel **Workers**.

### 3. Submit job (dari mesin koordinator)
```bash
SPARK_MASTER=spark://192.168.1.50:7077 SPARK_DRIVER_HOST=192.168.1.50 \
  python -m src.spark.train_svm_spark
```
Selama jalan, `:8080` → **Running Applications (1)** + executor tersebar ke semua Worker.

---

## 🔌 Cek konektivitas dulu (sebelum menyalahkan Spark)

Dari **mesin Worker**, pastikan Master terjangkau:
```bash
ping 192.168.1.50                      # ada balasan?
nc -zv 192.168.1.50 7077               # port Master terbuka? (atau: telnet 192.168.1.50 7077)
curl -s -o /dev/null -w "%{http_code}\n" http://192.168.1.50:8080/   # harus 200
```
- `ping` gagal → **beda jaringan** (cek WiFi, isolasi AP/"client isolation").
- `ping` ok tapi `nc` gagal → **firewall** memblok `7077` (lihat bawah).

---

## 🧯 Tabel masalah → solusi

| Gejala | Penyebab | Solusi |
|--------|----------|--------|
| Worker **tak muncul** di `:8080` | Master bind ke `127.0.0.1` | restart Master dgn `SPARK_MASTER_HOST=<MASTER_IP>` |
| Worker log: `Failed to connect to /127.0.0.1:7077` | `MASTER` salah / pakai loopback | pakai `spark://<MASTER_IP>:7077`, bukan `127.0.0.1` |
| Worker log: `Connection refused :7077` | firewall / Master belum jalan | buka port (lihat bawah) / cek Master ALIVE di `:8080` |
| `Incompatible` / serialVersion / aneh saat connect | **versi Spark beda** antar mesin | samakan: semua `pip install "pyspark>=4.0,<4.1"` |
| Worker **DEAD** di `:8080` | proses worker mati / di-`stop` | jalankan ulang `join_worker.sh`; jangan tutup terminalnya |
| Job jalan tapi **stuck/`TaskSchedulerImpl: Initial job has not accepted resources`** | resource kurang / worker tak nyambung benar | pastikan worker ALIVE & punya core; kurangi beban |
| Executor remote error `Failed to connect to driver` | driver tak terjangkau worker | submit dgn `SPARK_DRIVER_HOST=<MASTER_IP>` |
| Worker bind ke IP salah (`127.0.1.1`) | `SPARK_LOCAL_IP` belum diset | di mesin worker: `SPARK_LOCAL_IP=<IP_mesin_itu> bash join_worker.sh` |
| `JAVA_HOME is not set` / `java: command not found` | Java belum ada | install JDK 17/21, cek `java -version` |
| `ModuleNotFoundError: pyspark` | pyspark belum dipasang di mesin itu | `pip install "pyspark>=4.0,<4.1"` (venv yg sama dipakai skrip) |
| Job UDF error `No module named 'Sastrawi'` / `src` di executor remote | mesin worker tak punya dependency Python | pakai job **SVM** (tanpa UDF), atau pasang PySastrawi + sediakan `src/` di worker |

---

## 🔥 Firewall (penyebab tersering)

Spark butuh **lebih dari port 7077**: Master (7077, UI 8080), Worker (UI 8081+), plus
**port acak** untuk driver & block-manager. Untuk demo di LAN tepercaya, paling praktis:

```bash
# Ubuntu/Zorin — izinkan semua dari subnet LAN (sementara, saat demo):
sudo ufw allow from 192.168.1.0/24
# atau nonaktifkan sementara (HANYA di jaringan tepercaya):
sudo ufw disable        # ingat ufw enable lagi setelah selesai
```
Windows: izinkan `java`/`python` di Windows Defender Firewall (Private network), atau
matikan sementara untuk profil Private.

---

## 🔍 Verifikasi cepat (mesin koordinator)

```bash
# berapa worker ALIVE + total core (lewat REST Master):
python - <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen("http://localhost:8080/json/"))
print("status:", d["status"], "| worker alive:", d["aliveworkers"], "| core total:", d["cores"])
for w in d["workers"]:
    print(f"  - {w['host']} | {w['cores']} core | {w['state']}")
PY
```
Atau cukup `bash src/spark/cluster.sh status`.

---

## 🧹 Selesai / bersih-bersih

```bash
# mesin koordinator:
bash src/spark/cluster.sh stop
# mesin worker: cukup Ctrl+C di terminal join_worker.sh
# jangan lupa: sudo ufw enable  (kalau tadi dimatikan)
```

---

## 💡 Saran untuk demo yang mulus

1. **Pakai job SVM** (`train_svm_spark`) untuk demo antar-mesin — murni JVM, mesin teman
   **tak perlu** PySastrawi/`src/`, cukup Spark+Java. Paling sedikit yang bisa salah.
2. Uji **2 mesin dulu** (Master+1 Worker remote) sebelum menambah anggota.
3. Samakan versi Spark **sebelum** hari-H — ini error paling sering & paling membingungkan.
4. Screenshot `:8080` **saat job berjalan** (Worker ALIVE dari beberapa mesin + Running
   Application) untuk laporan.

> Catatan: bagian antar-mesin belum diuji dari repo ini (hanya 1 mesin tersedia);
> multi-worker di 1 mesin sudah terverifikasi. Kalau ada error saat 2 mesin, salin log
> Worker (`logs/spark/worker-*.log` di Master, atau output terminal `join_worker.sh`).

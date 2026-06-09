# 🖥️🖥️ Spark Cluster Antar-Mesin — Panduan & Troubleshooting (Tugas Kelompok)

Tujuan: jalankan **satu Master** (di mesin koordinator) + **beberapa Worker** dari
komputer anggota lain, lalu submit job SVM yang tereksekusi terdistribusi.

> **Istilah:** `MASTER_IP` = IP mesin yang menjalankan Master. Bisa **IP LAN**
> (`192.168.x` bila satu WiFi) atau **IP Tailscale** (`100.x` bila lewat VPN mesh).
> Ganti contoh di bawah dengan IP mesin koordinator kalian.

> ✅ **Status: SUDAH DIUJI antar-mesin** dari repo ini — Master + 1 Worker remote
> (`rocky-server`) lewat **Tailscale**, job terbukti dieksekusi di mesin lain
> (lihat bagian "Bukti & temuan" di bawah).

---

## ✅ Checklist prasyarat (SEMUA mesin)

- [ ] **Saling terjangkau jaringan** — TIDAK wajib WiFi sama. Dua opsi:
  - **LAN/WiFi sama** (paling gampang, latency rendah) → IP `192.168.x`/`10.x`.
  - **Tailscale** (beda lokasi/jaringan pun bisa) → install Tailscale di semua mesin, satu tailnet, pakai IP `100.x`.
- [ ] **Java 17 atau 21** terpasang → `java -version`. (Rocky/RHEL: `sudo dnf install -y java-21-openjdk-headless`)
- [ ] **PySpark versi SAMA persis** di semua mesin → `python -c "import pyspark; print(pyspark.__version__)"` harus `4.0.x` di semua. **Beda versi = pasti gagal.**
- [ ] Koordinator tahu **IP-nya** → LAN: `hostname -I`; Tailscale: `tailscale ip -4`.
- [ ] (Job baca Parquet, mis. SVM) data sudah **direplikasi** ke tiap Worker → `sync_data_to_worker.sh` (cluster tanpa HDFS, lihat langkah 2b).
- [ ] (Khusus job ber-UDF) PySastrawi + folder `src/` ada di mesin Worker — lihat catatan di bawah.

---

## 🚀 Langkah jalan (urut)

> Ganti `MASTER_IP` di bawah dengan IP koordinator: **LAN** `192.168.1.50` **atau**
> **Tailscale** `100.95.198.108`. Polanya identik, cuma beda alamat.

### 1. Mesin koordinator (Master)
```bash
# bind Master ke IP yang dijangkau worker (WAJIB; jangan 127.0.0.1)
SPARK_MASTER_HOST=MASTER_IP WORKER_COUNT=1 bash src/spark/cluster.sh start
```
Buka `http://localhost:8080` → harus muncul **Master ALIVE**.

### 2a. Replikasi data ke tiap Worker (untuk job baca Parquet, mis. SVM)
```bash
# dari KOORDINATOR — cluster tanpa HDFS: data harus ada di tiap node, path sama
bash src/spark/sync_data_to_worker.sh ravi@worker-host
```
(Job demo `_demo_remote` murni komputasi → langkah ini boleh dilewati.)

### 2b. Mesin tiap anggota (Worker)
```bash
# di folder repo, MASTER = spark://MASTER_IP:7077
MASTER=spark://MASTER_IP:7077 bash src/spark/join_worker.sh
```
Skrip auto-pilih IP Tailscale (`100.x`) bila MASTER beralamat `100.x`. Cek di
`:8080` koordinator → Worker baru muncul **ALIVE** di tabel **Workers**.
Biarkan terminal ini **terbuka** (worker hidup selama terminal hidup).

### 3. Submit job (dari mesin koordinator)
```bash
SPARK_MASTER=spark://MASTER_IP:7077 SPARK_DRIVER_HOST=MASTER_IP \
  python -m src.spark.train_svm_spark
```
Selama jalan, `:8080` → **Running Applications (1)** + executor tersebar ke semua Worker.

> **Bukti task di mesin remote** (paling meyakinkan untuk laporan): matikan worker
> lokal lalu jalankan job demo — seluruh task wajib mendarat di worker remote:
> ```bash
> kill $(cat logs/spark/worker-1.pid); rm logs/spark/worker-1.pid
> SPARK_MASTER=spark://MASTER_IP:7077 SPARK_DRIVER_HOST=MASTER_IP \
>   SPARK_LOCAL_IP=MASTER_IP python -m src.spark._demo_remote
> ```
> Output mencetak tabel executor + `tasks_selesai` per host.

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
| Banyak entri **DEAD** / `Address already in use :7077` / status `:8080` kacau | JVM Spark **orphan** dari run sebelumnya nyangkut | `bash src/spark/cluster.sh sweep` (sapu paksa semua master/worker), lalu `start` ulang |
| Job jalan tapi **stuck/`TaskSchedulerImpl: Initial job has not accepted resources`** | resource kurang / worker tak nyambung benar | pastikan worker ALIVE & punya core; kurangi beban |
| Executor remote error `Failed to connect to driver` | driver tak terjangkau worker | submit dgn `SPARK_DRIVER_HOST=<MASTER_IP>` |
| Worker bind ke IP salah (`127.0.1.1`) | `SPARK_LOCAL_IP` belum diset | di mesin worker: `SPARK_LOCAL_IP=<IP_mesin_itu> bash join_worker.sh` |
| `JAVA_HOME is not set` / `java: command not found` | Java belum ada | install JDK 17/21, cek `java -version` |
| `ModuleNotFoundError: pyspark` | pyspark belum dipasang di mesin itu | `pip install "pyspark>=4.0,<4.1"` (venv yg sama dipakai skrip) |
| Job UDF error `No module named 'Sastrawi'` / `src` di executor remote | mesin worker tak punya dependency Python | pakai job **SVM** (tanpa UDF), atau pasang PySastrawi + sediakan `src/` di worker |
| Executor remote: `Cannot run program ".venv/bin/python": No such file` | **Arrow** memaksa executor spawn Python pakai path venv koordinator yg tak ada di worker | sudah ditangani: `session.py` **mematikan Arrow di mode cluster** otomatis → job SVM jadi murni-JVM, worker tak perlu Python |
| Executor remote: `FileNotFoundException ... features_spark.parquet` | cluster tanpa HDFS, file cuma ada di koordinator | replikasi dulu: `bash src/spark/sync_data_to_worker.sh user@worker` |
| Worker remote **mati sendiri** begitu SSH ditutup (khusus **Tailscale SSH**) | Tailscale SSH membunuh proses background saat sesi tutup (nohup/setsid pun kena) | jalankan `join_worker.sh` di **foreground** pada terminal yang dibiarkan terbuka (jangan background) |
| Worker join via Tailscale bind ke IP `192.168.x` (tak terjangkau Master) | `hostname -I` mengembalikan IP LAN, bukan Tailscale | `join_worker.sh` kini auto-pilih `tailscale ip -4` bila MASTER `100.x`; atau set `SPARK_LOCAL_IP=<IP_tailscale_mesin_itu>` manual |

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

**Tailscale:** secara default semua port antar perangkat di tailnet kalian terbuka
(kecuali kalian pasang ACL ketat), jadi biasanya **tak perlu utak-atik firewall** —
trafik lewat interface `tailscale0`. Pastikan saja `ufw`/`firewalld` lokal tak memblok
interface itu. Cek dari worker: `nc -zv <MASTER_IP_tailscale> 7077` harus *succeeded*.

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

---

## 🔬 Bukti & temuan (dari uji nyata antar-mesin)

Diuji: koordinator + `rocky-server` (mesin remote, Rocky Linux 9.7) lewat **Tailscale**.

**Terbukti:** Worker remote mendaftar ke Master, dan job sintetis (`_demo_remote`,
agregasi 8 juta baris) **seluruh task-nya dieksekusi di mesin remote** (driver di
koordinator cuma mengkoordinir). Cross-machine Spark **berfungsi**.

**3 jebakan yang ditemukan & sudah ditangani (lihat tabel di atas):**
1. **Arrow** memaksa Python di executor → dimatikan otomatis di mode cluster (`session.py`).
2. **No-HDFS** → data Parquet harus direplikasi (`sync_data_to_worker.sh`).
3. **Tailscale SSH** membunuh worker background → jalankan di foreground.

**Catatan kejujuran untuk laporan (penting):** pada job **SVM 14k baris**, hampir
semua task justru jatuh ke **node lokal**; worker remote "kelaparan" task. Sebabnya
data terlalu kecil/cepat + latency antar-mesin (mis. ~100 ms lewat Tailscale) →
node lokal menyelesaikan partisi sebelum scheduler sempat menawarkannya ke node
remote (*locality/latency starvation*). **Distribusi antar-mesin baru benar-benar
"menang" ketika volume data cukup besar** untuk menutup overhead jaringan — ini
justru ilustrasi tepat kapan Big Data terdistribusi relevan dan kapan tidak.

> Kalau ada error saat menambah mesin, salin log Worker (`logs/spark/worker-*.log`
> di koordinator, atau output terminal `join_worker.sh`) saat melapor.

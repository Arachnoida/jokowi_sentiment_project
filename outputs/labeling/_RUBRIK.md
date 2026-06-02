# Rubrik Pelabelan Sentimen — Isu Ijazah Jokowi

Komentar diambil dari video YouTube seputar **dugaan ijazah Jojowi palsu**.
**ACUAN POLARITAS = SIKAP TERHADAP ISU/NARASI TUDUHAN (ijazah palsu)**, BUKAN
terhadap sosok Jokowi secara umum. Ini penting — jangan terbalik.

## Definisi kelas

### Positif  (mendukung / percaya narasi tuduhan)
Komentar yang **mendukung narasi isu**, **percaya tuduhan ijazah palsu**, atau
**menguatkan tuduhan**. Termasuk: menghina/menyerang Jokowi terkait ijazah,
menyebut ijazah palsu/bodong, menuntut Jokowi membuktikan/mengakui, memuji pihak
penuduh (Roy Suryo, dll), curiga ada yang ditutupi.
Contoh: "Semakin terlihat PALSU.", "Tunjukkan yang asli kalau berani!",
"UGM jelas melindungi kebohongan."

### Negatif  (menolak / membantah narasi tuduhan)
Komentar yang **menolak isu**, **menganggap tuduhan tidak benar**, atau
**mengkritik narasi/penuduh**. Termasuk: membela Jokowi soal ijazah, menyebut
ijazah asli/sudah terbukti, menyebut tuduhan fitnah/hoaks/cari sensasi,
mengecam penuduh (Roy Suryo, channel provokator) karena memprovokasi.
Contoh: "Ijazahnya jelas asli, sudah ditunjukkan UGM.", "Ini cuma fitnah cari
panggung.", "Channel provokator pemecah belah."

### Netral  (tidak jelas sikap / bertanya / informasi saja / tidak condong)
- Sikap tidak jelas terhadap tuduhan.
- Hanya **bertanya** tanpa menyiratkan keberpihakan ("Kapan sidangnya?").
- **Informasi saja** / pernyataan faktual netral / prosedural ("Ijazah hilang
  bisa minta salinan dari UGM.").
- Tidak condong ke salah satu pihak, komentar OOT, candaan ringan, sapaan,
  emoji saja, doa umum, spam.

## Aturan tie-break
- Kalau ada sindiran/sarkasme, nilai maksud sebenarnya (sarkasme membela Jokowi
  yang menyerang penuduh = Negatif).
- Kalau menyerang Jokowi TAPI bukan soal isu ijazah (mis. soal politik lain),
  dan tidak menyentuh tuduhan ijazah → cenderung Netral kecuali jelas menguatkan
  framing tuduhan.
- Kalau ragu antara berpihak vs tidak, dan tidak ada sinyal sikap yang jelas →
  Netral.
- Bahasa campur/typo/alay tetap dinilai dari maksudnya.

## Output (WAJIB)
Tulis file JSONL ke path yang diberikan, **satu objek per baris**, untuk SETIAP
baris input (jumlah baris output = jumlah baris input), format:
```
{"idx": <int sama persis dgn input>, "label": "Positif|Negatif|Netral", "confidence": <0.0-1.0>, "notes": "<alasan singkat <=8 kata>"}
```
- `confidence`: 0.9–1.0 sangat yakin; 0.6–0.8 cukup yakin; <0.6 ragu.
- `notes` singkat, mis. "menguatkan tuduhan palsu" / "membela, sebut fitnah" / "cuma bertanya".
- JANGAN lewatkan baris. JANGAN ubah idx.

# Rubrik Pelabelan Sentimen — Isu Ijazah Jokowi

Komentar diambil dari video YouTube seputar **dugaan ijazah Jojowi palsu**.
**ACUAN POLARITAS = SIKAP TERHADAP ISU/NARASI TUDUHAN (ijazah palsu)**, BUKAN
terhadap sosok Jokowi secara umum. Ini penting — jangan terbalik.

## Definisi kelas

### Positif  (mendukung / percaya narasi tuduhan)
Komentar yang **mendukung narasi isu**, **percaya tuduhan ijazah palsu**, atau
**menguatkan tuduhan**. Termasuk: menghina/menyerang Jokowi terkait ijazah,
menyebut ijazah palsu/bodong, menuntut Jokowi membuktikan/mengakui, mendukung
penuduh **terkait klaimnya** (mis. "Roy Suryo benar, ijazah harus dibongkar"),
curiga ada yang ditutupi. (Pujian telanjang ke penuduh tanpa menyentuh klaim
bukan Positif — lihat "Aturan keputusan: nilai DARI TEKS saja".)
Contoh: "Semakin terlihat PALSU.", "Tunjukkan yang asli kalau berani!",
"UGM jelas melindungi kebohongan."

### Negatif  (menolak / membantah narasi tuduhan)
Komentar yang **menolak isu**, **menganggap tuduhan tidak benar**, atau
**mengkritik narasi/penuduh**. Termasuk: membela Jokowi soal ijazah, menyebut
ijazah asli/sudah terbukti, menyebut tuduhan fitnah/hoaks/cari sensasi,
mengecam penuduh **karena menyebar tuduhan** (mis. "Roy Suryo provokator/cari
sensasi soal ijazah"). (Hinaan telanjang ke penuduh tanpa menyentuh klaim —
mis. "Roy Suryo stres" — bukan Negatif tapi Netral; lihat aturan text-only.)
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

## Aturan keputusan: nilai DARI TEKS saja (text-only stance)
Sikap dinilai **hanya dari sinyal yang ada di teks komentar**, bukan dari konteks
luar yang tidak tampak di teks (judul video, siapa "ibu/bapak" yang dimaksud,
arah sindiran yang tak terlihat). Alasan: model dilatih dari teks komentar, jadi
sikap yang mustahil dipelajari dari teksnya sendiri = noise, bukan signal — dan
labeler tak boleh dituntut tahu konteks video agar antar-labeler reprodusibel.

Sebuah komentar baru boleh dilabeli **Positif/Negatif** HANYA kalau teksnya
**menyentuh KLAIM/isu ijazah itu sendiri**. "Menyentuh klaim" TIDAK harus
menyebut kata "ijazah" — yang dihitung adalah:
- **predikat tentang kebenaran tuduhan**: palsu/bodong/asli/terbukti/fitnah/hoaks/
  bohong/tipu, termasuk **menyatakan ijazah direkayasa** (diedit/dipalsukan/
  rekayasa = ijazah palsu → Positif), ATAU
- **tuntutan konsekuensi yang mengandaikan Jokowi bersalah**: adili/penjarakan/
  tangkap/buktikan/akui/mundur (→ Positif), karena menuntut hukuman = menganggap
  tuduhan benar. (Sebaliknya: membela/menyebut fitnah → Negatif.)

Catatan — verba konsekuensi (adili/tangkap/laporin/penjara) TIDAK otomatis
berpolaritas; bergantung **siapa targetnya** dan **apakah basisnya disebut**:
- Target = **Jokowi (tertuduh)**: tuntut hukum → **Positif**. Di korpus ini Jokowi
  hanya relevan sebagai subjek tuduhan, jadi menuntut hukumannya = menganggap
  tuduhan benar (mis. "adili pria solo itu", "adili Mulyono"). **Alias Jokowi yang
  baku se-korpus: "pria solo", "Mulyono" (nama kecil) → = Jokowi.**
  Tapi harus **tuntutan akuntabilitas hukum spesifik** (adili/penjara/tangkap/
  buktikan). **Kutukan/umpatan generik** ke Jokowi ("Mulyono akan sengsara/celaka/
  mampus") = serang **sosok**, tak terikat klaim → **Netral** (bisa benci karena
  apa saja). Bandingkan: "adili Mulyono" (Positif) vs "Mulyono akan sengsara" (Netral).
- Target = **penuduh/tokoh lain** (Roy Suryo, dll): tuntut hukum → **Negatif hanya
  kalau basis klaim disebut**, yakni ada **kata isu di teks** (ijazah/palsu/fitnah/
  tuduhan) — mis. "tangkap Roy Suryo, fitnah dia". Tuntutan **telanjang tanpa kata
  isu** → **Netral**: "kabarin kalau Roy Suryo ditangkap", "Buktikan Roy Panci
  segera tangkap" (verba generik "buktikan" + olok nama, nol kata isu → butuh
  inferensi peran dari luar teks). Olok nama (Roy Panci) = personal, bukan sikap.
- Target **tak teridentifikasi** atau basis soal **perilaku/cara bicara** →
  **Netral** (mis. "laporin aja dia, pengacara siapa itu yg bicaranya meluap").

**Menilai KARAKTER/kualitas pribadi saja tidak cukup.** Pujian atau hinaan ke
seorang tokoh ("stres", "gila", "hebat", "mantap", "keren") TANPA menyentuh klaim
→ **Netral**, **meskipun tokohnya penuduh/pembela yang dikenal dan disebut di
teks**. Alasan: tanpa konten soal klaim, tidak ada sinyal yang bisa dipelajari
model → model bingung saat testing. Menyebut nama tokoh saja BUKAN sinyal sikap.

Bedanya: "adili pria solo itu" = tuntutan konsekuensi (mengandaikan bersalah) →
**Positif**; "Roy Suryo stres" = nilai kondisi mental orang → **Netral**.

**Keluhan meta soal isunya** (muak/capek/bosan membahas, buang waktu/anggaran,
"harusnya sepele") TANPA predikat kebenaran klaim → **Netral**. Justru sentimen
"muak membahas / hentikan" condong MENJAUH dari Positif (yang percaya tuduhan
biasanya ingin isu terus dikejar), tapi arahnya tetap tak determinate → Netral.

**Curiga ditutupi harus spesifik.** Marker Positif "curiga ada yang ditutupi/
dilindungi" berlaku kalau ditautkan ke **kasus/cover-up spesifik** (aparat
melindungi, ungkap kasus, transparansi, "polisi titipan Jokowi" #46). Kalau cuma
sentimen **umum** ("semua melindungi Jokowi, hancur negara" #89) tanpa tautan
kasus/cover-up → **Netral** (keluhan politik luas, sekelas #34).

**Serangan karakter + verdict klaim = ikut verdict.** Kalau satu komentar memuat
olok/serangan karakter (Netral sendirian) DAN penegasan klaim ("ijazah palsu/asli")
sekaligus, polaritasnya **ikut verdict klaimnya**, bukan dinetralkan oleh hinaannya
(mis. "otak di dengkul... ijazah Jokowi PALSU" → Positif).

**Objek predikat: klaim vs orang.** Predikat kebenaran (palsu/bohong/diedit/dst.)
berpolaritas kalau objeknya = **klaim/ijazah** ("**ijazah** diedit", "**ijazahnya**
palsu" → arah pasti). Tapi kalau objeknya = **orang** yang **tak teridentifikasi**
("si **pembohong** itu...", "**kebohongan** [siapa] dibuat-buat") → subjek/arah
menggantung → **Netral**. Yang menentukan: predikat menempel ke klaim atau ke
orang-yang-tak-jelas.
Asimetri (seperti "adili"): predikat "bohong/pembohong" yang menempel ke **JOKOWI
(tertuduh)** → **Positif** ("Jokowi bohong" = bohong soal ijazahnya = palsu; mis.
"Jokowi sama bohongnya"). Ke **penuduh/pihak ketiga (Roy/Andi)** tanpa konteks
isi-bohongnya → **Netral** (mis. "Andi ketauan bohongnya", #55).

**Kondisional ≠ assertion.** Kata "palsu/asli" yang muncul sebagai **andaian/
hipotesis** ("**kalo** mungkin itu palsu...", "**andai** asli ya...") BUKAN sikap —
komentator tak commit ke kebenaran klaim → **Netral**. Yang berpolaritas hanya
**penegasan** ("ijazahnya palsu/asli"), bukan pengandaiannya. Jangan otomatis
nge-Positif/Negatif-kan tiap kemunculan kata "palsu/asli".
TAPI bedakan **kondisional RETORIS menggiring** dari hipotetis terbuka: "kalau
ijazah asli, ngapain cari pengacara mahal?" menggiring ke kesimpulan curiga (=
palsu) → **ikut arah kesimpulannya** (Positif). Hanya hipotetis **genuinely
terbuka** (mis. "kalo mungkin palsu, UGM dirugikan", #11) yang → Netral.

Aturan ini **dua arah** dan berlaku walau tokohnya jelas: hinaan/pujian telanjang
ke pihak mana pun (pro maupun kontra) sama-sama Netral selama klaim tak disinggung.

Contoh:
- "Roy soryo orang stres" (hina orang, tak sentuh klaim) → **Netral**.
- "Mantap ibu,,❤❤❤" (tak sebut isu, "ibu" tak teridentifikasi) → **Netral**.
- "Mantap pak Roy 👍" (puji penuduh tapi tak sentuh klaim) → **Netral**.
- "Roy Suryo cuma cari sensasi soal ijazah" (nilai tuduhan bad-faith) → **Negatif**.
- "Mantap bu, fitnah ijazah ini keterlaluan" (sebut fitnah = verdict klaim) → **Negatif**.
- "Adili segera pria solo itu" (tuntut hukum tertuduh/Jokowi) → **Positif**.
- "Kabarin kalau Roy Suryo ditangkap" (tuntut hukum penuduh, basis tak disebut) → **Netral**.
- "Tangkap Roy Suryo, fitnah dia soal ijazah" (penuduh + basis klaim) → **Negatif**.
- "Buktikan Roy Panci segera tangkap" (penuduh, "buktikan" generik + olok nama, nol kata isu) → **Netral**.
- "Geng Roy, kalian yg harus MEMBUKTIKAN ijazah itu memang palsu!" (beban bukti ke penuduh + kata isu "palsu") → **Negatif**. (Bandingkan: "buktikan" ke Jokowi/"tunjukkan ijazahmu" → Positif; "buktikan" ke penuduh + kata isu → Negatif; bare "buktikan" + olok nama → Netral.)
- "Mantan napi, koruptor, partai kalah pilpres, rame2 dukung Suryo 😂" (ad-hominem ke kubu penuduh, nol kata isu) → **Netral**.
- "Roy Suryo cs punya dendam pribadi, balas dendam ke Jokowi" (spekulasi motif penuduh, nol kata isu, tanpa verdict) → **Netral**. Beda dgn "tuduhan ini fitnah/cari sensasi **soal ijazah**" (motif ditautkan ke isu + verdict) → **Negatif**.
- "Laporin aja dia, pengacara siapa itu yg bicaranya meluap" (target/basis tak jelas) → **Netral**.
- "Muak membahas ijazah Jokowi, buang anggaran, selesai 15 menit" (keluhan meta, tak ada predikat klaim) → **Netral**.
- "Kalo mungkin itu palsu, UGM yg dirugikan, harusnya UGM menuntut" (hipotetis + argumen prosedural) → **Netral**.
- "Pak Refli, tolong berkasnya dilengkapi, jangan tergesa, ini masalah hukum" (nasihat prosedural ke pihak perkara, nol kata isu, tanpa verdict) → **Netral**.
- "Berarti next time boleh pakai ijazah palsu, negara tipu-tipu dong" (sarkasme MENEGASKAN palsu + nutupi) → **Positif**.
- "Ketahuan yang edit ijazah itu si X, harus kena ITE" (edit ijazah = ijazah palsu) → **Positif**.
- "Ijazah harus dibongkar, jangan ditutupi!" (kuatkan tuduhan) → **Positif**.

## Output (WAJIB)
Tulis file JSONL ke path yang diberikan, **satu objek per baris**, untuk SETIAP
baris input (jumlah baris output = jumlah baris input), format:
```
{"idx": <int sama persis dgn input>, "label": "Positif|Negatif|Netral", "confidence": <0.0-1.0>, "notes": "<alasan singkat <=8 kata>"}
```
- `confidence`: 0.9–1.0 sangat yakin; 0.6–0.8 cukup yakin; <0.6 ragu.
- `notes` singkat, mis. "menguatkan tuduhan palsu" / "membela, sebut fitnah" / "cuma bertanya".
- JANGAN lewatkan baris. JANGAN ubah idx.

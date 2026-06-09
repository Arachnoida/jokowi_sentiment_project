# 🎤 Cheat Sheet — Presentasi Progress (pegang saat tampil)

**Pembuka (hook):** *"Saya bandingkan SVM klasik vs IndoBERT untuk sentimen komentar
YouTube isu ijazah Jokowi — dan menemukan model terbaik tergantung ukuran & keseimbangan
data, bukan sekadar yang paling canggih."*

---

## Angka wajib hafal
- Data: **14.107** scraped → **10.000 dilabeli** (LLM-assisted, 3 kelas).
- **4 versi:** v1 6k-imbal · v2 3k-bal · v3 10k-imbal · v4 5,8k-bal.
- **SVM terbaik = v2 → macro-F1 0,694** | **IndoBERT terbaik = v4 → 0,666**.
- Cross-validation (jujur): **~0,59–0,62**.
- Ensemble SVM: **+2–3%**.

## 3 temuan ( INTI presentasi)
1. **CROSSOVER:** data kecil → **SVM menang**; data 10k → **IndoBERT menyalip**.
2. **Balance > imbalance** → imbalanced bikin **Netral** (minoritas) anjlok.
3. **Kualitas > kuantitas** → v2 (3k bersih) > v4 (10k noisy).

## Istilah penting
- **Issue-anchored:** sikap thd TUDUHAN, bukan thd Jokowi. Positif=dukung tuduhan.
- **macro-F1:** rata-rata F1 per kelas (adil utk 3 kelas; accuracy bias ke mayoritas).
- **Netral = kelas tersulit** (paling ambigu & paling sedikit).

## Pipeline (kalau ditanya alur)
YouTube API → `raw_comments` → pelabelan LLM → preprocessing (SVM agresif / IndoBERT minimal)
→ training → evaluasi. **Semua di MongoDB Atlas.**

---

## ❓ Jawaban cepat pertanyaan sulit
| Pertanyaan | Jawaban singkat |
|---|---|
| Kenapa label LLM? | Skala + konsisten; jujur *LLM-assisted*; diuji konsensus 3-pass |
| Kenapa macro-F1? | 3 kelas; accuracy bias mayoritas (v3: acc 0,66 tapi macro-F1 0,63) |
| Kenapa akurasi ~0,6–0,7? | Tugas ambigu (Positif↔Netral, sarkasme/alay); relabel tak menolong → batas tugas |
| Kenapa IndoBERT kalah? | Data kecil + domain mismatch (Wiki/berita) → rencana IndoBERTweet |
| Kenapa Netral susah? | Kelas "tak bersikap" intrinsik ambigu & paling langka |
| Relabel kok gagal? | Bukti bahwa plafon = ambiguitas tugas, bukan noise label (negative result yang sah) |

## Penutup
*"Tidak ada pemenang mutlak; plafon ditentukan ambiguitas tugas. Langkah berikutnya:
IndoBERTweet (model domain medsos) + sampel gold-standard manusia."*

> **Tonjolkan kematangan:** evaluasi pakai **cross-validation** + eksperimen yang **gagal
> pun didokumentasikan**.

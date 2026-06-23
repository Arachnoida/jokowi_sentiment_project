"""
src/text_normalizer.py
Normalisasi teks komentar Indonesia: cleaning, slang normalization, stopwords.
Digunakan sebagai Python helper yang dipanggil dari Spark UDF.
"""

import re
from typing import List

SLANG_DICT = {
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lo": "kamu",
    "lu": "kamu",
    "elu": "kamu",
    "ga": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "g": "tidak",
    "udah": "sudah",
    "dah": "sudah",
    "blm": "belum",
    "blom": "belum",
    "bgt": "sangat",
    "banget": "sangat",
    "bngt": "sangat",
    "bener": "benar",
    "lg": "lagi",
    "lgi": "lagi",
    "jg": "juga",
    "jga": "juga",
    "krn": "karena",
    "karna": "karena",
    "tp": "tapi",
    "tpi": "tapi",
    "yg": "yang",
    "yng": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "trs": "terus",
    "trus": "terus",
    "utk": "untuk",
    "buat": "untuk",
    "bwt": "untuk",
    "dri": "dari",
    "dr": "dari",
    "kl": "kalau",
    "klo": "kalau",
    "kalu": "kalau",
    "kalo": "kalau",
    "klu": "kalau",
    "sy": "saya",
    "q": "saya",
    "ntar": "nanti",
    "tar": "nanti",
    "aja": "saja",
    "gk": "tidak",
    "tak": "tidak",
    "gakk": "tidak",
    "sdh": "sudah",
    "org": "orang",
    "tau": "tahu",
    "cuma": "hanya",
    "kak": "",
    "bang": "",
    "kok": "",
    "kan": "",
    "jgn": "jangan",  # pertahankan negasi
    "jkw": "jokowi",
    "roi": "roy",  # konsolidasi entitas (typo umum)
    "lah": "",
    "dong": "",
    "deh": "",
    "sih": "",
    "nih": "",
    "wkwk": "",
    "wkwkwk": "",
    "haha": "",
    "hehe": "",
    "hihi": "",
    "kwkw": "",
    "xD": "",
    "xd": "",
    "ok": "oke",
    "mantap": "mantap",
    "mantep": "mantap",
    "keren": "keren",
    "kerennn": "keren",
    "bagus": "bagus",
    "baguss": "bagus",
    "jelek": "jelek",
    "jelekk": "jelek",
    "gila": "gila",
    "gilaa": "gila",
    "bro": "",
    "gan": "",
    "min": "",
    "iya": "ya",
    "iyaa": "ya",
    "makasih": "terima kasih",
    "makasi": "terima kasih",
    "mksh": "terima kasih",
    "tks": "terima kasih",
    "smgt": "semangat",
    "subscribe": "berlangganan",
    "sub": "berlangganan",
    "like": "suka",
    "komen": "komentar",
    "share": "bagikan",
    "upload": "unggah",
    "konten": "konten",
    "channel": "saluran",
}

STOPWORDS_ID = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "pada",
    "adalah",
    "ini",
    "itu",
    "atau",
    "juga",
    "sudah",
    "akan",
    "ada",
    "bisa",
    "lebih",
    "saat",
    "kami",
    "mereka",
    "kita",
    "anda",
    "ia",
    "dia",
    "kamu",
    "saya",
    "apa",
    "bagaimana",
    "mengapa",
    "ketika",
    "karena",
    "jika",
    "maka",
    "namun",
    "tetapi",
    "tapi",
    "saja",
    "lagi",
    "pun",
    "agar",
    "supaya",
    "ya",
    "telah",
    "sedang",
    "masih",
    "sangat",
    "sekali",
    "amat",
    "oleh",
    "dalam",
    "luar",
    "bawah",
    "atas",
    "antara",
    "setelah",
    "sebelum",
    "sejak",
    "selama",
    "sering",
    "kadang",
    "jarang",
    "selalu",
    "pernah",
    "satu",
    "dua",
    "tiga",
    "beberapa",
    "setiap",
    "semua",
    "lain",
    "sama",
    "baru",
    "lama",
    "besar",
    "kecil",
    "nya",
    "si",  # klitik/partikel berdiri sendiri — noise utk TF-IDF
}

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#\w+")
_RE_EMOJI = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f926-\U0001f937"
    "\u200d"
    "\u2640-\u2642"
    "\u2600-\u2b55"
    "\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    re.UNICODE,
)
_RE_NUMBER = re.compile(r"\b\d+\b")
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")


# [SVM-1] cleaning agresif: lowercase + buang url/mention/hashtag/emoji/angka/non-alfabet
def clean_for_svm(text: str) -> str:
    """Cleaning agresif untuk jalur SVM + TF-IDF."""
    if not text or not isinstance(text, str):
        return ""
    t = text.lower()
    t = _RE_URL.sub(" ", t)
    t = _RE_MENTION.sub(" ", t)
    t = _RE_HASHTAG.sub(" ", t)
    t = _RE_EMOJI.sub(" ", t)
    t = _RE_NUMBER.sub(" ", t)
    t = _RE_NON_ALPHA.sub(" ", t)
    t = _RE_MULTI_SPACE.sub(" ", t)
    return t.strip()


# [BERT-1] cleaning minimal: buang url/mention/emoji, TAPI pertahankan tanda baca & morfologi
def clean_for_bert(text: str) -> str:
    """Cleaning minimal untuk jalur IndoBERT (morfologi terjaga)."""
    if not text or not isinstance(text, str):
        return ""
    t = text.lower()
    t = _RE_URL.sub(" ", t)
    t = _RE_MENTION.sub(" ", t)
    t = _RE_EMOJI.sub(" ", t)
    t = re.sub(r"[^\w\s.,!?;:'\"()-]", " ", t)
    t = _RE_MULTI_SPACE.sub(" ", t)
    return t.strip()


# [SVM-2] ganti slang/informal -> bentuk baku (dipakai di dalam preprocess_svm_python)
def normalize_slang(text: str, slang_dict: dict = SLANG_DICT) -> str:
    """Ganti kata slang/informal dengan bentuk baku."""
    if not text:
        return ""
    tokens = text.split()
    normalized = [slang_dict.get(tok, tok) for tok in tokens]
    result = " ".join(tok for tok in normalized if tok)
    return _RE_MULTI_SPACE.sub(" ", result).strip()


# [SVM-3a] pecah teks jadi token per-spasi
def tokenize(text: str) -> List[str]:
    """Tokenisasi sederhana berdasarkan whitespace."""
    return text.split() if text else []


# [SVM-3b] buang stopword + token 1 huruf (negasi spt 'tidak'/'jangan' sengaja dipertahankan)
def remove_stopwords(tokens: List[str], stopwords: set = STOPWORDS_ID) -> List[str]:
    """Hapus stopword dari daftar token."""
    return [tok for tok in tokens if tok not in stopwords and len(tok) > 1]


# [SVM-gabungan] urutan penuh: clean_for_svm -> normalize_slang -> tokenize -> remove_stopwords (BELUM di-stem)
def preprocess_svm_python(text: str) -> str:
    """Pipeline SVM tanpa stemming (stemming dilakukan terpisah via Sastrawi UDF)."""
    cleaned = clean_for_svm(text)
    normalized = normalize_slang(cleaned)
    tokens = tokenize(normalized)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


# [BERT-gabungan] alias clean_for_bert (tanpa stem/stopword — IndoBERT butuh kata utuh)
def preprocess_bert_python(text: str) -> str:
    """Pipeline IndoBERT: cleaning minimal, tanpa stemming/stopword removal."""
    return clean_for_bert(text)


# helper stemming per-token (jalur produksi men-stem STRING PENUH di udf.py; ini cadangan/uji)
def try_stem_sastrawi(text: str) -> str:
    """Stemming dengan PySastrawi. Fallback ke input jika tidak tersedia."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        tokens = text.split()
        return " ".join(stemmer.stem(tok) for tok in tokens)
    except ImportError:
        return text

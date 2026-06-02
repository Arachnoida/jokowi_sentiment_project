"""
src/text_normalizer.py
Normalisasi teks komentar Indonesia: cleaning, slang normalization, stopwords.
Digunakan sebagai Python helper yang dipanggil dari Spark UDF.
"""

import re
from typing import List

SLANG_DICT = {
    "gw": "saya", "gue": "saya", "gua": "saya",
    "lo": "kamu", "lu": "kamu", "elu": "kamu",
    "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak",
    "enggak": "tidak", "tdk": "tidak", "g": "tidak",
    "udah": "sudah", "dah": "sudah",
    "blm": "belum", "blom": "belum",
    "bgt": "sangat", "banget": "sangat", "bngt": "sangat",
    "bener": "benar",
    "lg": "lagi", "lgi": "lagi",
    "jg": "juga", "jga": "juga",
    "krn": "karena", "karna": "karena",
    "tp": "tapi", "tpi": "tapi",
    "yg": "yang", "yng": "yang",
    "dgn": "dengan", "dg": "dengan",
    "trs": "terus", "trus": "terus",
    "utk": "untuk", "buat": "untuk", "bwt": "untuk",
    "dri": "dari", "dr": "dari",
    "kl": "kalau", "klo": "kalau", "kalu": "kalau", "kalo": "kalau", "klu": "kalau",
    "sy": "saya", "q": "saya",
    "ntar": "nanti", "tar": "nanti",
    # Tambahan berbasis frekuensi korpus (sering muncul, sebelumnya tak tertangani):
    "aja": "saja", "gk": "tidak", "tak": "tidak", "gakk": "tidak",
    "sdh": "sudah", "org": "orang", "tau": "tahu", "cuma": "hanya",
    "kak": "", "bang": "", "kok": "", "kan": "",
    "lah": "", "dong": "", "deh": "", "sih": "", "nih": "",
    "wkwk": "", "wkwkwk": "", "haha": "", "hehe": "", "hihi": "",
    "kwkw": "", "xD": "", "xd": "",
    "ok": "oke",
    "mantap": "mantap", "mantep": "mantap",
    "keren": "keren", "kerennn": "keren",
    "bagus": "bagus", "baguss": "bagus",
    "jelek": "jelek", "jelekk": "jelek",
    "gila": "gila", "gilaa": "gila",
    "bro": "", "gan": "", "min": "",
    "iya": "ya", "iyaa": "ya",
    "makasih": "terima kasih", "makasi": "terima kasih",
    "mksh": "terima kasih", "tks": "terima kasih",
    "smgt": "semangat",
    "subscribe": "berlangganan", "sub": "berlangganan",
    "like": "suka", "komen": "komentar",
    "share": "bagikan", "upload": "unggah",
    "konten": "konten", "channel": "saluran",
}

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada",
    "adalah", "ini", "itu", "atau", "tidak", "juga", "sudah",
    "akan", "ada", "bisa", "lebih", "saat", "kami",
    "mereka", "kita", "anda", "ia", "dia",
    "kamu", "saya", "apa", "bagaimana", "mengapa",
    "ketika", "karena", "jika", "maka", "namun", "tetapi", "tapi",
    "saja", "lagi", "pun", "agar", "supaya", "ya",
    "telah", "sedang", "masih", "sangat", "sekali", "amat",
    "oleh", "dalam", "luar", "bawah", "atas", "antara", "setelah",
    "sebelum", "sejak", "selama", "sering", "kadang", "jarang",
    "selalu", "bukan", "belum", "pernah",
    "satu", "dua", "tiga", "beberapa", "setiap", "semua",
    "lain", "sama", "baru", "lama", "besar", "kecil",
}

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#\w+")
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\u200d"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    re.UNICODE,
)
_RE_NUMBER = re.compile(r"\b\d+\b")
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")


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


def normalize_slang(text: str, slang_dict: dict = SLANG_DICT) -> str:
    """Ganti kata slang/informal dengan bentuk baku."""
    if not text:
        return ""
    tokens = text.split()
    normalized = [slang_dict.get(tok, tok) for tok in tokens]
    result = " ".join(tok for tok in normalized if tok)
    return _RE_MULTI_SPACE.sub(" ", result).strip()


def tokenize(text: str) -> List[str]:
    """Tokenisasi sederhana berdasarkan whitespace."""
    return text.split() if text else []


def remove_stopwords(tokens: List[str], stopwords: set = STOPWORDS_ID) -> List[str]:
    """Hapus stopword dari daftar token."""
    return [tok for tok in tokens if tok not in stopwords and len(tok) > 1]


def preprocess_svm_python(text: str) -> str:
    """Pipeline SVM tanpa stemming (stemming dilakukan terpisah via Sastrawi UDF)."""
    cleaned = clean_for_svm(text)
    normalized = normalize_slang(cleaned)
    tokens = tokenize(normalized)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def preprocess_bert_python(text: str) -> str:
    """Pipeline IndoBERT: cleaning minimal, tanpa stemming/stopword removal."""
    return clean_for_bert(text)


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

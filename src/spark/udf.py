"""UDF Spark untuk preprocessing teks Indonesia.

Membungkus fungsi murni di ``src.text_normalizer`` agar bisa dipakai sebagai
kolom Spark terdistribusi. Konsisten dengan jalur sklearn/Mongo:

  - Jalur SVM : preprocess_svm_python (clean -> slang -> stopword) lalu STEM
    Sastrawi pada STRING PENUH (bukan per-token), satu instance stemmer
    di-inisialisasi sekali per worker (lazy singleton) — identik dengan
    ``src/modeling/_backfill_processed_svm.py``.
  - Jalur BERT: clean_for_bert (cleaning minimal, morfologi terjaga).
"""
from __future__ import annotations

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from src.text_normalizer import clean_for_bert, preprocess_svm_python

# Lazy singleton: dibuat sekali per proses worker Python, bukan per baris.
_STEMMER = None

# Kata yang DILINDUNGI dari over-stemming Sastrawi. Prefiks "se-" untuk kata
# bermuatan sikap ("se-" = setuju/sependapat) di-stem jadi root yang kehilangan
# makna: setuju->tuju, sependapat->dapat. Didaftarkan sebagai root word ke kamus
# stemmer agar dikenali utuh. WAJIB ditambahkan SEBELUM stem pertama (CachedStemmer
# meng-cache hasil per kata, jadi add() setelah stem tidak berefek).
# Catatan: pendekatan kamus hanya memperbaiki sebagian over-stem (yg hasil strip-nya
# bukan root valid). Kasus rule-based spt "dibantah->ban"/"seandainya->anda" TIDAK
# bisa diperbaiki via dictionary.add → keterbatasan Sastrawi yg didokumentasikan.
_PROTECTED_ROOTS = (
    "setuju", "sependapat", "sepaham", "sepakat",  # sikap (se-)
    "seting", "setting", "setel", "mentri",        # over-stem: diseting->ting, mentri->tri
)


# bikin stemmer Sastrawi SEKALI per proses worker (lazy singleton), bukan tiap baris
def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        stemmer = StemmerFactory().create_stemmer()
        # create_stemmer() -> CachedStemmer; kamus ada di .delegatedStemmer
        inner = getattr(stemmer, "delegatedStemmer", stemmer)
        for word in _PROTECTED_ROOTS:
            inner.dictionary.add(word)
        _STEMMER = stemmer
    return _STEMMER


# [JALUR A - SVM] preprocess_svm_python lalu STEM string penuh -> isi kolom 'svm'
def make_svm_text(text: str | None) -> str:
    """Teks fitur jalur SVM: preprocess + stem Sastrawi pada string penuh."""
    pre = preprocess_svm_python(text or "")
    return _stemmer().stem(pre) if pre else pre


# [JALUR A - BERT] clean_for_bert saja -> isi kolom 'bert'
def make_bert_text(text: str | None) -> str:
    """Teks fitur jalur IndoBERT: cleaning minimal."""
    return clean_for_bert(text or "")


# bungkus jadi Spark UDF -> dipakai sbg kolom di regenerate_processed_mongo.py / preprocess_spark.py
make_svm_text_udf = udf(make_svm_text, StringType())
make_bert_text_udf = udf(make_bert_text, StringType())

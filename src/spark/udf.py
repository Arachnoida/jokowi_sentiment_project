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


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        _STEMMER = StemmerFactory().create_stemmer()
    return _STEMMER


def make_svm_text(text: str | None) -> str:
    """Teks fitur jalur SVM: preprocess + stem Sastrawi pada string penuh."""
    pre = preprocess_svm_python(text or "")
    return _stemmer().stem(pre) if pre else pre


def make_bert_text(text: str | None) -> str:
    """Teks fitur jalur IndoBERT: cleaning minimal."""
    return clean_for_bert(text or "")


make_svm_text_udf = udf(make_svm_text, StringType())
make_bert_text_udf = udf(make_bert_text, StringType())

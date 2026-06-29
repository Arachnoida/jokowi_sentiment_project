"""Helper subset eksperimen: baca allowlist comment_id dari CSV.

Dipakai trainer (SVM sklearn/Spark, IndoBERT) untuk membatasi data ke subset
tertentu (mis. balanced 3000) TANPA mengubah koleksi/parquet produksi —
cukup filter baris setelah load. Default trainer (tanpa --subset) tak terpengaruh.
"""
from __future__ import annotations

import pandas as pd


def load_subset_ids(path: str) -> set[str]:
    """Set comment_id (string) dari kolom `comment_id` sebuah CSV allowlist."""
    ids = pd.read_csv(path, usecols=["comment_id"])["comment_id"].astype(str)
    return set(ids)

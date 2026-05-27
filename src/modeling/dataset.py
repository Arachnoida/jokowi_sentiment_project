"""
src/modeling/dataset.py
Bangun dataset siap-model dari data berlabel:
  1. Terapkan dua jalur preprocessing (SVM agresif + stem, IndoBERT minimal).
  2. Encode label ke id.
  3. Split stratified train/val/test (70/20/10) dengan seed tetap.
  4. Simpan/muat split sebagai Parquet agar SVM & IndoBERT memakai data identik.

Sengaja TIDAK memakai Spark: data berlabel kecil (~3.000 baris) sehingga pandas
jauh lebih cepat dan sederhana.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from configs.config import Config
from src.utils import setup_logger
from src.text_normalizer import preprocess_svm_python, preprocess_bert_python
from src.modeling.labels import LABEL2ID, LABELS, class_distribution

logger = setup_logger("modeling.dataset")

# Stemmer Sastrawi mahal untuk dibuat — buat SEKALI lalu dipakai ulang.
_STEMMER = None


def _get_stemmer():
    """Kembalikan stemmer Sastrawi tunggal (lazy, di-cache). None bila tak ada."""
    global _STEMMER
    if _STEMMER is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _STEMMER = StemmerFactory().create_stemmer()
            logger.info("Stemmer Sastrawi diinisialisasi.")
        except ImportError:
            logger.warning(
                "PySastrawi tidak terpasang — jalur SVM berjalan TANPA stemming."
            )
            _STEMMER = False  # penanda 'tidak tersedia' agar tak coba ulang
    return _STEMMER or None


def make_text_svm(text: str) -> str:
    """Jalur A: clean agresif + slang + buang stopword (text_normalizer) lalu stem."""
    pre = preprocess_svm_python(text or "")
    stemmer = _get_stemmer()
    if stemmer is not None and pre:
        return stemmer.stem(pre)
    return pre


def make_text_bert(text: str) -> str:
    """Jalur B: cleaning minimal (morfologi terjaga)."""
    return preprocess_bert_python(text or "")


def build_modeling_frame(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Dari DataFrame berlabel (kolom: comment_id, text, label[, label_id]),
    hasilkan kolom text_svm, text_bert, label_id; buang baris yang kosong
    setelah preprocessing atau tanpa label valid.
    """
    df = df_labeled.copy()
    if "label_id" not in df.columns or df["label_id"].isna().any():
        df["label_id"] = df["label"].map(LABEL2ID)
    df = df[df["label_id"].notna()].copy()
    df["label_id"] = df["label_id"].astype(int)

    logger.info("Preprocessing jalur SVM (clean+slang+stopword+stem)...")
    df["text_svm"] = df["text"].astype(str).map(make_text_svm)
    logger.info("Preprocessing jalur IndoBERT (minimal)...")
    df["text_bert"] = df["text"].astype(str).map(make_text_bert)

    before = len(df)
    df = df[(df["text_svm"].str.len() > 0) & (df["text_bert"].str.len() > 0)]
    dropped = before - len(df)
    if dropped:
        logger.warning("%d baris dibuang karena kosong setelah preprocessing.", dropped)

    cols = ["comment_id", "video_id", "text", "text_svm", "text_bert",
            "label", "label_id"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    seed: Optional[int] = None,
    label_col: str = "label_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split stratified 70/20/10 (train/val/test) menjaga proporsi kelas di tiap split.
    """
    from sklearn.model_selection import train_test_split

    seed = seed if seed is not None else Config.modeling.RANDOM_SEED
    train_r = Config.modeling.TRAIN_RATIO
    val_r = Config.modeling.VAL_RATIO
    test_r = Config.modeling.TEST_RATIO

    # 1) pisahkan test
    df_temp, df_test = train_test_split(
        df, test_size=test_r, stratify=df[label_col], random_state=seed,
    )
    # 2) dari sisa, pisahkan val (proporsional terhadap sisa)
    val_relative = val_r / (train_r + val_r)
    df_train, df_val = train_test_split(
        df_temp, test_size=val_relative, stratify=df_temp[label_col],
        random_state=seed,
    )
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        logger.info("Split %-5s: %d baris", name, len(part))
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> Path:
    """Simpan ketiga split sebagai Parquet ke data/processed/splits/."""
    out_dir = out_dir or Config.paths.SPLITS
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
    logger.info("Split disimpan ke %s", out_dir)
    return out_dir


def load_splits(
    in_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Muat train/val/test dari Parquet."""
    in_dir = in_dir or Config.paths.SPLITS
    parts = []
    for name in ("train", "val", "test"):
        fp = in_dir / f"{name}.parquet"
        if not fp.exists():
            raise FileNotFoundError(
                f"Split '{name}' tidak ditemukan: {fp}. "
                "Jalankan build_and_save() / notebook 05 lebih dulu."
            )
        parts.append(pd.read_parquet(fp))
    return tuple(parts)  # type: ignore[return-value]


def build_and_save(
    df_labeled: pd.DataFrame,
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orkestrasi penuh: preprocessing dua jalur -> split stratified -> simpan.
    Mengembalikan (train, val, test) dan mencetak ringkasan distribusi kelas.
    """
    df_model = build_modeling_frame(df_labeled)

    dist = class_distribution(df_model)
    logger.info("Distribusi kelas dataset model:\n%s", dist.to_string())
    if dist.min() == 0:
        logger.warning(
            "Ada kelas tanpa contoh: %s. Periksa hasil labeling.",
            [LABELS[i] for i, v in enumerate(dist.values) if v == 0],
        )

    df_train, df_val, df_test = stratified_split(df_model, seed=seed)
    save_splits(df_train, df_val, df_test, out_dir=out_dir)
    return df_train, df_val, df_test

"""
src/modeling/labels.py
Kosakata label sentimen + parsing hasil ekspor Label Studio menjadi DataFrame.

Jalur utama pengambilan label: tombol "Export" (format JSON) di UI Label Studio,
disimpan ke outputs/labeling/label_studio_export.json. Parser di sini robust
terhadap format ekspor LS standar (list of task dengan field `data` + `annotations`).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from configs.config import Config
from src.utils import setup_logger

logger = setup_logger("modeling.labels")

# Urutan menentukan id kelas: Negatif=0, Netral=1, Positif=2.
LABELS: List[str] = list(Config.modeling.LABELS)
LABEL2ID: Dict[str, int] = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: lab for lab, i in LABEL2ID.items()}

# Pemetaan toleran terhadap variasi penulisan label dari anotator.
_LABEL_ALIASES = {
    "positif": "Positif", "positive": "Positif", "pos": "Positif",
    "negatif": "Negatif", "negative": "Negatif", "neg": "Negatif",
    "netral": "Netral", "neutral": "Netral", "net": "Netral",
}


def normalize_label(value: Optional[str]) -> Optional[str]:
    """Petakan string label mentah ke salah satu kelas kanonik, atau None."""
    if not value or not isinstance(value, str):
        return None
    key = value.strip().lower()
    return _LABEL_ALIASES.get(key)


def _extract_choice(annotation: Dict[str, Any], from_name: str) -> Optional[str]:
    """Ambil nilai Choices dari satu annotation Label Studio."""
    for res in annotation.get("result", []):
        if res.get("from_name") and res.get("from_name") != from_name:
            continue
        value = res.get("value", {})
        choices = value.get("choices")
        if choices:
            return normalize_label(choices[0])
    return None


def parse_label_studio_export(
    path: Union[str, Path],
    from_name: Optional[str] = None,
    skip_unlabeled: bool = True,
) -> pd.DataFrame:
    """
    Baca file JSON hasil ekspor Label Studio menjadi DataFrame berlabel.

    Args:
        path: Lokasi file JSON ekspor (list of task).
        from_name: Nama Choices di config XML (default dari Config).
        skip_unlabeled: Jika True, task tanpa anotasi/label valid dibuang.

    Returns:
        DataFrame kolom: comment_id, video_id, text, label, label_id.

    Raises:
        FileNotFoundError: jika file ekspor tidak ada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File ekspor Label Studio tidak ditemukan: {path}\n"
            "Di UI Label Studio: project > Export > pilih format JSON, "
            "lalu simpan ke lokasi tersebut."
        )

    from_name = from_name or Config.label_studio.FROM_NAME
    tasks = json.loads(path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    n_no_annot = 0
    for task in tasks:
        data = task.get("data", {})
        annotations = task.get("annotations", []) or []

        label: Optional[str] = None
        for annot in annotations:
            if annot.get("was_cancelled"):
                continue
            label = _extract_choice(annot, from_name)
            if label:
                break

        if label is None:
            n_no_annot += 1
            if skip_unlabeled:
                continue

        rows.append({
            "comment_id": data.get("comment_id"),
            "video_id": data.get("video_id"),
            "text": data.get("text", ""),
            "label": label,
            "label_id": LABEL2ID.get(label) if label else None,
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Parsing ekspor LS: %d task total, %d berlabel, %d tanpa label%s.",
        len(tasks), len(tasks) - n_no_annot, n_no_annot,
        " (dibuang)" if skip_unlabeled else "",
    )
    if not df.empty:
        df = df[df["text"].astype(str).str.strip().astype(bool)]
        df = df.drop_duplicates(subset="comment_id")
    return df.reset_index(drop=True)


def class_distribution(df: pd.DataFrame, col: str = "label") -> pd.Series:
    """Hitung jumlah per kelas (untuk cek keseimbangan dataset)."""
    return df[col].value_counts().reindex(LABELS, fill_value=0)

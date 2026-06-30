from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


TABLEAU_FILES = {
    "class_distribution": "tableau_class_distribution.csv",
    "model_comparison": "tableau_model_comparison.csv",
    "per_class_metrics": "tableau_per_class_metrics.csv",
    "confusion_matrix": "tableau_confusion_matrix.csv",
    "predictions": "tableau_predictions_balanced3k.csv",
    "top_features": "tableau_top_features.csv",
    "top_terms": "tableau_top_terms.csv",
    "eda_summary": "tableau_eda_summary.csv",
    "manifest": "tableau_manifest.csv",
}

LABEL_ORDER = ["Negatif", "Netral", "Positif"]
MODEL_ORDER = ["IndoBERT", "SVM sklearn", "SVM Spark MLlib"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "f1_negatif": "F1 Negatif",
    "f1_netral": "F1 Netral",
    "f1_positif": "F1 Positif",
    "f1_score": "F1-score",
    "precision": "Precision",
    "recall": "Recall",
    "weighted_f1": "Weighted F1",
}

GENERIC_TERMS = {
    "tidak", "orang", "kalau", "yang", "dan", "atau", "ini", "itu",
    "saya", "kamu", "dia", "mereka", "kita", "kami", "aja", "saja",
    "pak", "bu", "para", "dengan", "untuk", "dari", "ke", "di",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_tableau_dir() -> Path:
    env_dir = os.getenv("SENTIMENT_TABLEAU_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    root_candidate = get_project_root() / "outputs" / "tableau"
    if root_candidate.exists():
        return root_candidate

    return (Path.cwd() / "outputs" / "tableau").resolve()


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series

    if series.dtype == bool:
        return series

    mapping = {
        "true": True,
        "false": False,
        "benar": True,
        "salah": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def standardize_top_terms(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source_path", "rank", "term", "count"])

    df = df.copy()

    merged_col = None
    for col in df.columns:
        if "," in col and all(token in col for token in ["rank", "term", "count"]):
            merged_col = col
            break

    if merged_col is not None:
        split_df = df[merged_col].astype(str).str.split(",", expand=True)
        if split_df.shape[1] >= 4:
            split_df = split_df.iloc[:, :4]
            split_df.columns = ["path", "rank", "term", "count"]
            source_col = df["source"] if "source" in df.columns else "vocab.top_terms"
            df = pd.DataFrame(
                {
                    "source_path": source_col.astype(str),
                    "rank": split_df["rank"],
                    "term": split_df["term"],
                    "count": split_df["count"],
                }
            )

    if "source_path" not in df.columns:
        if {"source", "path"}.issubset(df.columns):
            df["source_path"] = df["source"].astype(str) + "." + df["path"].astype(str)
        elif "source" in df.columns:
            df["source_path"] = df["source"].astype(str)
        else:
            df["source_path"] = "vocab.top_terms"

    for col in ["rank", "count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["source_path", "rank", "term", "count"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    df = df[required].dropna(subset=["term"]).copy()
    fallback_rank = pd.Series(range(1, len(df) + 1), index=df.index)
    df["rank"] = df["rank"].fillna(fallback_rank).astype(int)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["term"] = df["term"].astype(str)
    return df.sort_values("rank")


def get_domain_terms(top_terms: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if top_terms.empty:
        return top_terms

    df = top_terms.copy()
    df["term_norm"] = df["term"].astype(str).str.lower().str.strip()
    df = df[~df["term_norm"].isin(GENERIC_TERMS)].copy()
    df = df.drop(columns=["term_norm"], errors="ignore")
    return df.sort_values("count", ascending=False).head(top_n)


def standardize_dataframes(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    class_df = data.get("class_distribution", pd.DataFrame()).copy()
    if not class_df.empty and "count" in class_df.columns:
        class_df["count"] = pd.to_numeric(class_df["count"], errors="coerce").fillna(0).astype(int)
        class_df["dataset"] = class_df["dataset"].astype(str)
        class_df["label"] = pd.Categorical(class_df["label"], LABEL_ORDER, ordered=True)
    data["class_distribution"] = class_df

    comp_df = data.get("model_comparison", pd.DataFrame()).copy()
    for col in ["accuracy", "macro_f1", "f1_negatif", "f1_netral", "f1_positif"]:
        if col in comp_df.columns:
            comp_df[col] = pd.to_numeric(comp_df[col], errors="coerce")
    data["model_comparison"] = comp_df

    per_class_df = data.get("per_class_metrics", pd.DataFrame()).copy()
    for col in ["precision", "recall", "f1_score", "support", "accuracy", "macro_f1", "weighted_f1"]:
        if col in per_class_df.columns:
            per_class_df[col] = pd.to_numeric(per_class_df[col], errors="coerce")
    if not per_class_df.empty and "label" in per_class_df.columns:
        per_class_df["label"] = pd.Categorical(per_class_df["label"], LABEL_ORDER, ordered=True)
    data["per_class_metrics"] = per_class_df

    cm_df = data.get("confusion_matrix", pd.DataFrame()).copy()
    if not cm_df.empty and "count" in cm_df.columns:
        cm_df["count"] = pd.to_numeric(cm_df["count"], errors="coerce").fillna(0).astype(int)
        for col in ["actual_label", "predicted_label"]:
            if col in cm_df.columns:
                cm_df[col] = pd.Categorical(cm_df[col], LABEL_ORDER, ordered=True)
    data["confusion_matrix"] = cm_df

    pred_df = data.get("predictions", pd.DataFrame()).copy()
    if not pred_df.empty:
        if "is_correct" in pred_df.columns:
            normalized = normalize_bool_series(pred_df["is_correct"])
            pred_df["is_correct"] = normalized.fillna(pred_df["actual_label"] == pred_df["predicted_label"])
        else:
            pred_df["is_correct"] = pred_df["actual_label"] == pred_df["predicted_label"]

        if "error_status" not in pred_df.columns:
            pred_df["error_status"] = pred_df["is_correct"].map({True: "Benar", False: "Salah"})

        if "error_pair" not in pred_df.columns:
            pred_df["error_pair"] = pred_df["actual_label"].astype(str) + " → " + pred_df["predicted_label"].astype(str)

        if "confidence" in pred_df.columns:
            pred_df["confidence"] = pd.to_numeric(pred_df["confidence"], errors="coerce")

        if "text" in pred_df.columns:
            pred_df["text"] = pred_df["text"].astype(str)
            pred_df["text_preview"] = pred_df["text"].str.slice(0, 170)
            pred_df.loc[pred_df["text"].str.len() > 170, "text_preview"] += "..."
    data["predictions"] = pred_df

    top_features = data.get("top_features", pd.DataFrame()).copy()
    for col in ["weight", "rank"]:
        if col in top_features.columns:
            top_features[col] = pd.to_numeric(top_features[col], errors="coerce")
    if not top_features.empty and "label" in top_features.columns:
        top_features["label"] = pd.Categorical(top_features["label"], LABEL_ORDER, ordered=True)
    data["top_features"] = top_features

    data["top_terms"] = standardize_top_terms(data.get("top_terms", pd.DataFrame()))
    return data


@st.cache_data(show_spinner=False)
def load_all_data(tableau_dir_str: str | None = None) -> dict[str, pd.DataFrame]:
    tableau_dir = Path(tableau_dir_str).resolve() if tableau_dir_str else get_tableau_dir()

    data: dict[str, pd.DataFrame] = {}
    for key, filename in TABLEAU_FILES.items():
        data[key] = read_csv_safe(tableau_dir / filename)

    data = standardize_dataframes(data)
    return data


def get_available_datasets(data: dict[str, pd.DataFrame]) -> list[str]:
    comp = data.get("model_comparison", pd.DataFrame())
    if not comp.empty and "dataset" in comp.columns:
        values = comp["dataset"].dropna().astype(str).unique().tolist()
        ordered = [d for d in ["balanced3k", "full14k"] if d in values]
        return ordered + [d for d in values if d not in ordered]
    return ["balanced3k", "full14k"]


def get_available_models(data: dict[str, pd.DataFrame], include_spark: bool = True) -> list[str]:
    comp = data.get("model_comparison", pd.DataFrame())
    if not comp.empty and "model" in comp.columns:
        values = comp["model"].dropna().astype(str).unique().tolist()
        ordered = [m for m in MODEL_ORDER if m in values]
        models = ordered + [m for m in values if m not in ordered]
    else:
        models = MODEL_ORDER.copy()

    if not include_spark:
        models = [m for m in models if "Spark" not in m]

    return models


def get_metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def get_metric_from_summary(data: dict[str, pd.DataFrame], metric_path: str, default: Any = None) -> Any:
    summary = data.get("eda_summary", pd.DataFrame())
    if summary.empty or "metric_path" not in summary.columns:
        return default

    row = summary[summary["metric_path"].astype(str) == metric_path]
    if row.empty:
        return default

    return row.iloc[0].get("value", default)


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number.is_integer():
        return f"{int(number):,}".replace(",", ".")
    return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percent(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{decimals}f}%".replace(".", ",")


def format_compact(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(number) >= 1000:
        return f"{int(number):,}".replace(",", ".")
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".replace(".", ",")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def get_class_balance_status(class_df: pd.DataFrame, dataset: str) -> tuple[str, str]:
    if class_df.empty:
        return "-", "Data tidak tersedia"

    subset = class_df[class_df["dataset"].astype(str) == dataset].copy()
    if subset.empty:
        return "-", "Data tidak tersedia"

    counts = subset["count"].astype(int).tolist()
    if len(counts) > 1 and len(set(counts)) == 1:
        return "Balanced", f"{format_number(counts[0])} data per kelas"

    subset = subset.sort_values("count", ascending=False)
    top = subset.iloc[0]
    second = subset.iloc[1] if len(subset) > 1 else None

    if second is not None:
        diff = int(top["count"] - second["count"])
        return str(top["label"]), f"Selisih {format_number(diff)} dari kelas berikutnya"

    return str(top["label"]), "Kelas terbanyak"

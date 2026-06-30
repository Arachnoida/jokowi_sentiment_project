from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


LABEL_ORDER = ["Negatif", "Netral", "Positif"]

METRIC_FILE_SPECS = [
    {
        "dataset": "balanced3k",
        "model": "SVM sklearn",
        "path": "svm_balanced3k_metrics.json",
    },
    {
        "dataset": "balanced3k",
        "model": "SVM Spark MLlib",
        "path": "svm_spark_balanced3k_metrics.json",
    },
    {
        "dataset": "balanced3k",
        "model": "IndoBERT",
        "path": "indobert_balanced3k_metrics.json",
    },
    {
        "dataset": "full14k",
        "model": "SVM sklearn",
        "path": "svm_full14k_metrics.json",
    },
    {
        "dataset": "full14k",
        "model": "SVM Spark MLlib",
        "path": "svm_spark_metrics.json",
    },
    {
        "dataset": "full14k",
        "model": "IndoBERT",
        "path": "indobert_metrics.json",
    },
]

PREDICTION_FILE_SPECS = [
    {
        "dataset": "balanced3k",
        "model": "SVM sklearn",
        "path": "svm_balanced3k_predictions.csv",
    },
    {
        "dataset": "balanced3k",
        "model": "IndoBERT",
        "path": "indobert_balanced3k_predictions.csv",
    },
]


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    return logging.getLogger("export_tableau")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_if_exists(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        logger.warning("File tidak ditemukan: %s", path)
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        logger.warning("File kosong: %s", path)
        return pd.DataFrame()


def get_test_block(metrics: dict[str, Any]) -> dict[str, Any]:
    if isinstance(metrics.get("test"), dict):
        return metrics["test"]
    return metrics


def get_f1_value(class_metrics: dict[str, Any]) -> float | None:
    for key in ["f1", "f1-score", "f1_score"]:
        if key in class_metrics:
            return class_metrics[key]
    return None


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "benar"}:
        return True
    if text in {"false", "0", "no", "n", "salah"}:
        return False
    return None


def export_model_comparison(
    reports_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    file_specs = [
        ("balanced3k", reports_dir / "model_comparison_balanced3k.csv"),
        ("full14k", reports_dir / "model_comparison_full14k.csv"),
    ]

    for dataset_name, path in file_specs:
        df = read_csv_if_exists(path, logger)
        if df.empty:
            continue

        df.insert(0, "dataset", dataset_name)
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "accuracy",
                "macro_f1",
                "f1_negatif",
                "f1_netral",
                "f1_positif",
            ]
        )

    write_csv(combined, output_dir / "tableau_model_comparison.csv")
    logger.info("Exported tableau_model_comparison.csv (%s rows)", len(combined))
    return combined


def export_per_class_metrics(
    reports_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for spec in METRIC_FILE_SPECS:
        path = reports_dir / spec["path"]
        if not path.exists():
            logger.warning("Metric file dilewati karena tidak ditemukan: %s", path.name)
            continue

        metrics = read_json(path)
        test = get_test_block(metrics)
        per_class = test.get("per_class", {})

        for label in LABEL_ORDER:
            class_metrics = per_class.get(label, {})
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "model": spec["model"],
                    "label": label,
                    "precision": class_metrics.get("precision"),
                    "recall": class_metrics.get("recall"),
                    "f1_score": get_f1_value(class_metrics),
                    "support": class_metrics.get("support"),
                    "accuracy": test.get("accuracy"),
                    "macro_f1": test.get("macro_f1"),
                    "weighted_f1": test.get("weighted_f1"),
                    "n_train": test.get("n_train"),
                    "n_test": test.get("n_test"),
                }
            )

    df = pd.DataFrame(rows)
    write_csv(df, output_dir / "tableau_per_class_metrics.csv")
    logger.info("Exported tableau_per_class_metrics.csv (%s rows)", len(df))
    return df


def export_confusion_matrix(
    reports_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for spec in METRIC_FILE_SPECS:
        path = reports_dir / spec["path"]
        if not path.exists():
            logger.warning("Confusion matrix dilewati karena metric file tidak ditemukan: %s", path.name)
            continue

        metrics = read_json(path)
        test = get_test_block(metrics)
        matrix = test.get("confusion_matrix")

        if matrix is None:
            logger.warning("Tidak ada confusion_matrix pada: %s", path.name)
            continue

        labels = test.get("labels") or metrics.get("labels") or LABEL_ORDER

        for actual_idx, row in enumerate(matrix):
            for predicted_idx, count in enumerate(row):
                actual_label = labels[actual_idx] if actual_idx < len(labels) else str(actual_idx)
                predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else str(predicted_idx)

                rows.append(
                    {
                        "dataset": spec["dataset"],
                        "model": spec["model"],
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "count": int(count),
                        "is_diagonal": actual_label == predicted_label,
                    }
                )

    df = pd.DataFrame(rows)
    write_csv(df, output_dir / "tableau_confusion_matrix.csv")
    logger.info("Exported tableau_confusion_matrix.csv (%s rows)", len(df))
    return df


def standardize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "label_asli": "actual_label",
        "label": "actual_label",
        "prediksi": "predicted_label",
        "prediction": "predicted_label",
        "benar": "is_correct",
        "correct": "is_correct",
        "keyakinan": "confidence",
    }

    out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    required_columns = ["comment_id", "text", "actual_label", "predicted_label"]
    for column in required_columns:
        if column not in out.columns:
            out[column] = None

    if "is_correct" not in out.columns:
        out["is_correct"] = out["actual_label"] == out["predicted_label"]
    else:
        out["is_correct"] = out["is_correct"].map(normalize_bool)

    if "confidence" not in out.columns:
        out["confidence"] = None

    out["error_status"] = out["is_correct"].map(
        lambda value: "Benar" if value is True else "Salah" if value is False else "Tidak diketahui"
    )

    out["error_pair"] = out["actual_label"].astype(str) + " → " + out["predicted_label"].astype(str)

    selected = [
        "comment_id",
        "text",
        "actual_label",
        "predicted_label",
        "is_correct",
        "error_status",
        "error_pair",
        "confidence",
    ]

    return out[selected]


def export_predictions(
    reports_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for spec in PREDICTION_FILE_SPECS:
        path = reports_dir / spec["path"]
        df = read_csv_if_exists(path, logger)

        if df.empty:
            continue

        standardized = standardize_prediction_columns(df)
        standardized.insert(0, "dataset", spec["dataset"])
        standardized.insert(1, "model", spec["model"])
        frames.append(standardized)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "comment_id",
                "text",
                "actual_label",
                "predicted_label",
                "is_correct",
                "error_status",
                "error_pair",
                "confidence",
            ]
        )

    write_csv(combined, output_dir / "tableau_predictions_balanced3k.csv")
    logger.info("Exported tableau_predictions_balanced3k.csv (%s rows)", len(combined))
    return combined


def export_top_features(
    reports_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    path = reports_dir / "svm_balanced3k_top_features.csv"
    df = read_csv_if_exists(path, logger)

    if df.empty:
        df = pd.DataFrame(columns=["model", "dataset", "label", "feature", "weight", "rank"])
        write_csv(df, output_dir / "tableau_top_features.csv")
        logger.info("Exported empty tableau_top_features.csv")
        return df

    out = df.copy()

    column_aliases = {
        "class": "label",
        "kelas": "label",
        "term": "feature",
        "word": "feature",
        "coef": "weight",
        "score": "weight",
        "importance": "weight",
    }

    out = out.rename(columns={k: v for k, v in column_aliases.items() if k in out.columns})

    if "label" not in out.columns:
        out["label"] = None

    if "feature" not in out.columns:
        candidate_cols = [c for c in out.columns if c.lower() in {"token", "ngram", "kata"}]
        out["feature"] = out[candidate_cols[0]] if candidate_cols else None

    if "weight" not in out.columns:
        out["weight"] = None

    if "rank" not in out.columns:
        out["rank"] = out.groupby("label").cumcount() + 1

    out.insert(0, "dataset", "balanced3k")
    out.insert(1, "model", "SVM sklearn")

    selected = ["dataset", "model", "label", "feature", "weight", "rank"]
    out = out[selected]

    write_csv(out, output_dir / "tableau_top_features.csv")
    logger.info("Exported tableau_top_features.csv (%s rows)", len(out))
    return out


def flatten_scalars(obj: Any, prefix: str = "") -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_scalars(value, next_prefix)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            next_prefix = f"{prefix}[{index}]"
            yield from flatten_scalars(value, next_prefix)
    else:
        yield {
            "metric_path": prefix,
            "value": obj,
            "value_type": type(obj).__name__,
        }


def maybe_extract_label_distribution(obj: Any, source_path: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        keys = set(obj.keys())

        if set(LABEL_ORDER).issubset(keys):
            for label in LABEL_ORDER:
                value = obj.get(label)
                if isinstance(value, (int, float)):
                    rows.append(
                        {
                            "dataset": "full14k",
                            "source_path": source_path,
                            "label": label,
                            "count": int(value),
                        }
                    )

        for key, value in obj.items():
            child_path = f"{source_path}.{key}" if source_path else str(key)
            rows.extend(maybe_extract_label_distribution(value, child_path))

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child_path = f"{source_path}[{index}]"
            rows.extend(maybe_extract_label_distribution(value, child_path))

    return rows


def maybe_extract_terms(obj: Any, source_path: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{source_path}.{key}" if source_path else str(key)

            if isinstance(value, list) and any(token in key.lower() for token in ["term", "word", "token"]):
                for rank, item in enumerate(value, start=1):
                    term = None
                    count = None

                    if isinstance(item, dict):
                        term = item.get("term") or item.get("word") or item.get("token")
                        count = item.get("count") or item.get("freq") or item.get("frequency")
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        term = item[0]
                        count = item[1]

                    if term is not None:
                        rows.append(
                            {
                                "source_path": child_path,
                                "rank": rank,
                                "term": term,
                                "count": count,
                            }
                        )

            rows.extend(maybe_extract_terms(value, child_path))

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child_path = f"{source_path}[{index}]"
            rows.extend(maybe_extract_terms(value, child_path))

    return rows


def export_eda_tables(
    reports_dir: Path,
    labeling_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = reports_dir / "eda_spark.json"

    if path.exists():
        eda = read_json(path)
    else:
        logger.warning("eda_spark.json tidak ditemukan.")
        eda = {}

    summary_df = pd.DataFrame(list(flatten_scalars(eda)))
    write_csv(summary_df, output_dir / "tableau_eda_summary.csv")
    logger.info("Exported tableau_eda_summary.csv (%s rows)", len(summary_df))

    distribution_rows = maybe_extract_label_distribution(eda)

    balanced_path = labeling_dir / "balanced_3000.csv"
    balanced_df = read_csv_if_exists(balanced_path, logger)

    if not balanced_df.empty:
        label_col = None
        for candidate in ["label", "label_name", "sentiment"]:
            if candidate in balanced_df.columns:
                label_col = candidate
                break

        if label_col is not None:
            counts = balanced_df[label_col].value_counts(dropna=False)
            for label, count in counts.items():
                distribution_rows.append(
                    {
                        "dataset": "balanced3k",
                        "source_path": "outputs/labeling/balanced_3000.csv",
                        "label": label,
                        "count": int(count),
                    }
                )
        else:
            logger.warning("Kolom label tidak ditemukan pada balanced_3000.csv")

    class_distribution_df = pd.DataFrame(distribution_rows)
    write_csv(class_distribution_df, output_dir / "tableau_class_distribution.csv")
    logger.info("Exported tableau_class_distribution.csv (%s rows)", len(class_distribution_df))

    top_terms_df = pd.DataFrame(maybe_extract_terms(eda))
    write_csv(top_terms_df, output_dir / "tableau_top_terms.csv")
    logger.info("Exported tableau_top_terms.csv (%s rows)", len(top_terms_df))

    return summary_df, class_distribution_df, top_terms_df


def export_manifest(output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for csv_path in sorted(output_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            rows.append(
                {
                    "file_name": csv_path.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": ", ".join(df.columns),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "file_name": csv_path.name,
                    "rows": None,
                    "columns": None,
                    "column_names": f"ERROR: {error}",
                }
            )

    manifest = pd.DataFrame(rows)
    write_csv(manifest, output_dir / "tableau_manifest.csv")
    logger.info("Exported tableau_manifest.csv (%s rows)", len(manifest))
    return manifest


def run_export(project_dir: Path) -> None:
    logger = setup_logging()

    reports_dir = project_dir / "outputs" / "reports"
    labeling_dir = project_dir / "outputs" / "labeling"
    output_dir = project_dir / "outputs" / "tableau"

    ensure_dir(output_dir)

    logger.info("Project dir : %s", project_dir)
    logger.info("Reports dir : %s", reports_dir)
    logger.info("Labeling dir: %s", labeling_dir)
    logger.info("Output dir  : %s", output_dir)

    export_model_comparison(reports_dir, output_dir, logger)
    export_per_class_metrics(reports_dir, output_dir, logger)
    export_confusion_matrix(reports_dir, output_dir, logger)
    export_predictions(reports_dir, output_dir, logger)
    export_top_features(reports_dir, output_dir, logger)
    export_eda_tables(reports_dir, labeling_dir, output_dir, logger)
    export_manifest(output_dir, logger)

    logger.info("Selesai. File Tableau-ready tersedia di: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Tableau-ready CSV files from modeling artifacts."
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Path root project. Jika kosong, otomatis memakai root repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.project_dir:
        root = Path(args.project_dir).resolve()
    else:
        root = project_root()

    run_export(root)


if __name__ == "__main__":
    main()
"""
src/modeling/train_svm.py
Jalur A: SVM + TF-IDF untuk klasifikasi sentimen 3-kelas.

Memakai kolom `text_svm` (sudah di-clean agresif + slang + stopword + stem).
Model: TF-IDF (n-gram 1-2) -> LinearSVC. Cepat & kuat sebagai baseline teks.
Walau dataset seimbang, class_weight='balanced' dipakai sebagai jaring pengaman.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from configs.config import Config
from src.utils import setup_logger
from src.modeling.dataset import load_splits
from src.modeling import evaluate as ev

logger = setup_logger("modeling.train_svm")

TEXT_COL = "text_svm"
LABEL_COL = "label_id"


def build_pipeline(C: float = 1.0, ngram_max: int = 2, min_df: int = 2):
    """Bangun pipeline TF-IDF + LinearSVC."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, ngram_max),
            min_df=min_df,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(
            C=C,
            class_weight="balanced",
            random_state=Config.modeling.RANDOM_SEED,
        )),
    ])


def grid_search(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    Cari hiperparameter terbaik memakai split train+val sebagai validasi tetap
    (PredefinedSplit) — bukan CV acak — agar konsisten dengan pemakaian val.
    """
    import numpy as np
    from sklearn.model_selection import GridSearchCV, PredefinedSplit

    X = pd.concat([df_train[TEXT_COL], df_val[TEXT_COL]], ignore_index=True)
    y = pd.concat([df_train[LABEL_COL], df_val[LABEL_COL]], ignore_index=True)
    # -1 = selalu di train, 0 = fold validasi
    test_fold = np.r_[np.full(len(df_train), -1), np.zeros(len(df_val))]
    ps = PredefinedSplit(test_fold)

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "clf__C": [0.1, 0.5, 1.0, 2.0],
    }
    gs = GridSearchCV(
        build_pipeline(), param_grid, scoring="f1_macro", cv=ps, n_jobs=-1, verbose=1,
    )
    gs.fit(X, y)
    logger.info("Param terbaik: %s (val macro-F1=%.4f)", gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_


def train_and_evaluate(
    tune: bool = True,
    save: bool = True,
) -> Tuple[object, Dict[str, Dict]]:
    """
    Latih SVM lalu evaluasi pada val & test memakai split tersimpan.

    Args:
        tune: jika True, grid search hiperparameter di val; jika False pakai default.
        save: jika True, simpan model (joblib) + metrik + confusion matrix.

    Returns:
        (model terlatih, {"val": metrics_val, "test": metrics_test}).
    """
    df_train, df_val, df_test = load_splits()
    logger.info("Train=%d Val=%d Test=%d", len(df_train), len(df_val), len(df_test))

    if tune:
        model, best_params = grid_search(df_train, df_val)
        # Latih ulang pada train (val dipakai untuk seleksi, bukan dilatih).
        model.fit(df_train[TEXT_COL], df_train[LABEL_COL])
    else:
        model = build_pipeline()
        model.fit(df_train[TEXT_COL], df_train[LABEL_COL])

    results: Dict[str, Dict] = {}
    for name, part in [("val", df_val), ("test", df_test)]:
        y_pred = model.predict(part[TEXT_COL])
        metrics = ev.evaluate_predictions(part[LABEL_COL].tolist(), list(y_pred))
        ev.print_report(metrics, title=f"SVM — {name.upper()}")
        results[name] = metrics

    if save:
        _save_artifacts(model, results)
    return model, results


def _save_artifacts(model, results: Dict[str, Dict]) -> None:
    import joblib

    Config.paths.ensure_dirs()
    model_path = Config.paths.MODELS / "svm_tfidf.joblib"
    joblib.dump(model, model_path)
    logger.info("Model SVM disimpan ke %s", model_path)

    ev.save_metrics(results["test"], Config.paths.REPORTS / "svm_test_metrics.json",
                    model_name="SVM+TF-IDF")
    ev.save_metrics(results["val"], Config.paths.REPORTS / "svm_val_metrics.json",
                    model_name="SVM+TF-IDF")
    ev.plot_confusion_matrix(
        results["test"], Config.paths.REPORTS / "svm_test_confusion.png",
        title="SVM + TF-IDF — Test",
    )


def main(tune: bool = True) -> None:
    train_and_evaluate(tune=tune, save=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd

from .data_loader import format_percent, get_metric_label


def get_best_model(model_df: pd.DataFrame, dataset: str, metric: str = "macro_f1") -> tuple[str, float]:
    if model_df.empty or metric not in model_df.columns:
        return "-", 0.0

    subset = model_df[model_df["dataset"].astype(str) == dataset].copy()
    if subset.empty:
        return "-", 0.0

    best = subset.sort_values(metric, ascending=False).iloc[0]
    return str(best["model"]), float(best[metric])


def executive_insight(model_df: pd.DataFrame, dataset: str = "balanced3k") -> str:
    best_model, best_macro = get_best_model(model_df, dataset, "macro_f1")

    svm_row = model_df[
        (model_df["dataset"].astype(str) == dataset)
        & (model_df["model"].astype(str) == "SVM sklearn")
    ]

    svm_text = ""
    if not svm_row.empty:
        svm_macro = float(svm_row.iloc[0]["macro_f1"])
        delta = best_macro - svm_macro
        svm_text = (
            f" SVM sklearn tetap kompetitif dengan Macro-F1 {format_percent(svm_macro)}, "
            f"selisih sekitar {delta * 100:.2f} poin persentase dari model terbaik."
        ).replace(".", ",")

    return (
        f"Dataset awal berisi 14.107 komentar YouTube. Untuk evaluasi model, digunakan balanced3k "
        f"dengan 1.000 komentar pada setiap kelas sentimen. Pada dataset {dataset}, "
        f"{best_model} menjadi model terbaik berdasarkan Macro-F1 sebesar {format_percent(best_macro)}."
        f"{svm_text}"
    )


def model_performance_insight(model_df: pd.DataFrame, dataset: str, metric: str) -> str:
    metric_label = get_metric_label(metric)
    best_model, best_value = get_best_model(model_df, dataset, metric)
    best_macro_model, best_macro = get_best_model(model_df, dataset, "macro_f1")

    return (
        f"Pada dataset {dataset}, model dengan {metric_label} tertinggi adalah {best_model} "
        f"dengan skor {format_percent(best_value)}. Jika Macro-F1 digunakan sebagai metrik utama, "
        f"model terbaik adalah {best_macro_model} dengan Macro-F1 {format_percent(best_macro)}. "
        f"Macro-F1 diprioritaskan karena lebih adil untuk klasifikasi multi-kelas."
    )


def error_analysis_insight(filtered_pred: pd.DataFrame, model: str) -> str:
    if filtered_pred.empty:
        return f"Tidak ada data prediksi yang sesuai dengan filter untuk model {model}."

    total = len(filtered_pred)
    wrong = int((filtered_pred["error_status"].astype(str) == "Salah").sum())
    correct = total - wrong
    accuracy = correct / total if total else 0

    wrong_df = filtered_pred[filtered_pred["error_status"].astype(str) == "Salah"]
    if not wrong_df.empty:
        top_pair = wrong_df["error_pair"].value_counts().idxmax()
        top_count = int(wrong_df["error_pair"].value_counts().max())
        error_text = f" Error pair paling dominan adalah {top_pair} sebanyak {top_count} komentar."
    else:
        error_text = " Tidak ditemukan prediksi salah pada filter ini."

    return (
        f"Pada data terfilter, {model} menghasilkan {correct} prediksi benar dari {total} data "
        f"dengan akurasi sekitar {format_percent(accuracy)}.{error_text} "
        f"Pola ini dapat digunakan untuk meninjau komentar ambigu, sarkasme, dukungan implisit, "
        f"atau konteks politis yang sulit dibedakan antar kelas."
    )


def public_opinion_insight(top_terms_df: pd.DataFrame) -> str:
    terms = []
    if not top_terms_df.empty and "term" in top_terms_df.columns:
        terms = top_terms_df.sort_values("rank").head(8)["term"].astype(str).tolist()

    term_text = ", ".join(terms) if terms else "jokowi, roy, ijazah, palsu, benar, dan asli"

    return (
        f"Percakapan publik pada komentar YouTube didominasi oleh term seperti {term_text}. "
        f"Secara substantif, diskursus publik berpusat pada aktor yang terlibat, legitimasi isu, "
        f"serta klaim terkait keaslian atau kepalsuan ijazah. Analisis ini tidak bertujuan membuktikan "
        f"benar atau salahnya isu, tetapi memetakan pola opini publik di ruang komentar digital."
    )

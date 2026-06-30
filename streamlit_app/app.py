from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.charts import (
    PLOTLY_CONFIG,
    confusion_matrix_heatmap,
    correct_wrong_chart,
    error_pair_chart,
    full14k_distribution_chart,
    model_comparison_dual_chart,
    model_metric_chart,
    per_class_metric_chart,
    selected_dataset_distribution_chart,
    sentiment_distribution_chart,
    top_features_chart,
    top_terms_chart,
)
from utils.components import (
    download_dataframe_button,
    insight_box,
    label_definition_cards,
    load_css,
    metric_card,
    mini_card_grid,
    model_explanation_cards,
    neutral_box,
    page_header,
    pipeline_step_grid,
    render_dataframe,
    research_scope_card,
    section_header,
    sidebar_brand,
    sidebar_section,
    warning_box,
)
from utils.data_loader import (
    LABEL_ORDER,
    format_number,
    format_percent,
    get_available_datasets,
    get_available_models,
    get_class_balance_status,
    get_domain_terms,
    get_metric_label,
    get_tableau_dir,
    load_all_data,
)
from utils.insights import (
    error_analysis_insight,
    executive_insight,
    get_best_model,
    model_performance_insight,
    public_opinion_insight,
)


st.set_page_config(
    page_title="YPO Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()


def init_session_state(data: dict[str, pd.DataFrame]) -> None:
    datasets = get_available_datasets(data)
    models = get_available_models(data)

    defaults = {
        "selected_dataset": "balanced3k" if "balanced3k" in datasets else datasets[0],
        "selected_model": "IndoBERT" if "IndoBERT" in models else models[0],
        "selected_metric": "macro_f1",
        "selected_sentiments": LABEL_ORDER.copy(),
        "top_n": 10,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    sanitize_session_state(data)


def sanitize_session_state(data: dict[str, pd.DataFrame]) -> None:
    """Keep widget state valid before Streamlit instantiates sidebar widgets."""
    datasets = get_available_datasets(data)
    models = get_available_models(data)
    metrics = ["accuracy", "macro_f1", "f1_negatif", "f1_netral", "f1_positif"]

    if st.session_state.get("selected_dataset") not in datasets:
        st.session_state.selected_dataset = "balanced3k" if "balanced3k" in datasets else datasets[0]
    if st.session_state.get("selected_model") not in models:
        st.session_state.selected_model = "IndoBERT" if "IndoBERT" in models else models[0]
    if st.session_state.get("selected_metric") not in metrics:
        st.session_state.selected_metric = "macro_f1"

    sentiments = st.session_state.get("selected_sentiments", LABEL_ORDER.copy())
    st.session_state.selected_sentiments = [label for label in sentiments if label in LABEL_ORDER] or LABEL_ORDER.copy()

    try:
        top_n = int(st.session_state.get("top_n", 10))
    except (TypeError, ValueError):
        top_n = 10
    st.session_state.top_n = max(5, min(20, top_n))


def reset_filters(data: dict[str, pd.DataFrame]) -> None:
    datasets = get_available_datasets(data)
    models = get_available_models(data)
    st.session_state.selected_dataset = "balanced3k" if "balanced3k" in datasets else datasets[0]
    st.session_state.selected_model = "IndoBERT" if "IndoBERT" in models else models[0]
    st.session_state.selected_metric = "macro_f1"
    st.session_state.selected_sentiments = LABEL_ORDER.copy()
    st.session_state.top_n = 10


def render_sidebar(data: dict[str, pd.DataFrame]) -> str:
    sidebar_brand()

    sidebar_section("Navigation", "Pilih area analisis")
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Executive Overview",
            "Dataset Explorer",
            "Model Performance",
            "Error Analysis",
            "Public Opinion Insight",
            "Methodology",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    sidebar_section("Global Filters", "Berlaku pada visualisasi utama")

    datasets = get_available_datasets(data)
    models = get_available_models(data)

    st.sidebar.selectbox(
        "Dataset",
        options=datasets,
        key="selected_dataset",
        help="balanced3k digunakan sebagai basis utama evaluasi model.",
    )

    st.sidebar.selectbox(
        "Model",
        options=models,
        key="selected_model",
        help="Model yang digunakan untuk confusion matrix dan error analysis.",
    )

    st.sidebar.selectbox(
        "Metric Focus",
        options=["accuracy", "macro_f1", "f1_negatif", "f1_netral", "f1_positif"],
        key="selected_metric",
        format_func=get_metric_label,
    )

    st.sidebar.multiselect(
        "Sentiment Label",
        options=LABEL_ORDER,
        key="selected_sentiments",
    )

    st.sidebar.slider(
        "Top-N terms/features",
        min_value=5,
        max_value=20,
        step=1,
        key="top_n",
    )

    st.sidebar.button(
        "Reset Filters",
        use_container_width=True,
        on_click=reset_filters,
        args=(data,),
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Data mart: `{get_tableau_dir()}`")

    return page


def get_total_by_dataset(class_df: pd.DataFrame, dataset: str) -> int:
    if class_df.empty:
        return 0
    return int(class_df[class_df["dataset"].astype(str) == dataset]["count"].sum())


def get_dominant_label(class_df: pd.DataFrame, dataset: str) -> str:
    status, _ = get_class_balance_status(class_df, dataset)
    return status


def get_top_term(top_terms: pd.DataFrame) -> str:
    if top_terms.empty:
        return "-"
    return str(top_terms.sort_values("rank").iloc[0]["term"])


def get_issue_term(top_terms: pd.DataFrame) -> str:
    if top_terms.empty:
        return "-"
    priority = ["ijazah", "palsu", "asli", "benar", "ugm", "bukti", "roy", "jokowi"]
    terms = top_terms["term"].astype(str).str.lower().tolist()
    for term in priority:
        if term in terms:
            return term
    return str(top_terms.sort_values("rank").iloc[0]["term"])


def render_executive_overview(data: dict[str, pd.DataFrame]) -> None:
    page_header(
        "YouTube Public Opinion Analytics",
        "Interactive analytics platform for dataset exploration, model evaluation, error analysis, and public opinion insight.",
    )

    class_df = data["class_distribution"]
    comp_df = data["model_comparison"]
    top_terms = data["top_terms"]

    total_full = get_total_by_dataset(class_df, "full14k")
    total_balanced = get_total_by_dataset(class_df, "balanced3k")
    best_model, best_macro = get_best_model(comp_df, "balanced3k", "macro_f1")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Total Comments", format_number(total_full), "Full14k original dataset")
    with k2:
        metric_card("Balanced Dataset", format_number(total_balanced), "1.000 data per kelas", "rgba(167,139,250,0.18)")
    with k3:
        metric_card("Best Model", best_model, "Berdasarkan Macro-F1", "rgba(52,211,153,0.16)")
    with k4:
        metric_card("Best Macro-F1", format_percent(best_macro), "Balanced3k evaluation", "rgba(56,189,248,0.18)")

    insight_box("Executive Summary", executive_insight(comp_df, "balanced3k"))
    mini_card_grid(
        [
            ("Dataset", "Full14k untuk konteks", "Distribusi asli dipertahankan agar pola percakapan publik tidak kehilangan konteks."),
            ("Evaluation", "Balanced3k untuk model", "Evaluasi memakai kelas seimbang supaya perbandingan performa tidak bias terhadap kelas mayoritas."),
            ("Decision", "Macro-F1 sebagai fokus", "Macro-F1 dipakai sebagai metrik utama karena lebih adil untuk klasifikasi multi-kelas."),
        ],
        columns=3,
    )

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.plotly_chart(
            sentiment_distribution_chart(class_df, "Distribusi Sentimen: Full14k vs Balanced3k"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        st.plotly_chart(
            model_comparison_dual_chart(comp_df, "balanced3k"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    section_header("Top Discourse Terms", "Term dominan yang muncul dalam komentar YouTube.")
    overview_terms = data["top_terms"].sort_values("rank").head(min(st.session_state.top_n, 8))
    st.plotly_chart(
        top_terms_chart(overview_terms, min(st.session_state.top_n, 8), "Top Terms Keseluruhan"),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def render_dataset_explorer(data: dict[str, pd.DataFrame]) -> None:
    page_header(
        "Dataset Explorer",
        "Eksplorasi karakteristik dataset, distribusi kelas, dan term dominan.",
    )

    class_df = data["class_distribution"]
    top_terms = data["top_terms"]
    dataset = st.session_state.selected_dataset
    balance_status, balance_help = get_class_balance_status(class_df, dataset)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Selected Dataset", dataset, "Dataset aktif")
    with c2:
        metric_card("Total Data", format_number(get_total_by_dataset(class_df, dataset)), "Jumlah komentar")
    with c3:
        metric_card("Class Balance", balance_status, balance_help, "rgba(52,211,153,0.14)")
    with c4:
        metric_card("Top Term", get_top_term(top_terms), "Term paling sering muncul")

    insight_box(
        "Methodological Note",
        "Full14k digunakan untuk merepresentasikan distribusi komentar asli. Balanced3k digunakan sebagai basis evaluasi model agar setiap kelas sentimen memiliki representasi yang sama.",
    )
    mini_card_grid(
        [
            ("Profile", "Dataset aktif", f"Visualisasi berikut membaca distribusi kelas pada dataset {dataset}."),
            ("Balance", balance_status, balance_help),
            ("Vocabulary", "Top terms", "Term dominan membantu membaca isu yang paling sering muncul pada komentar."),
        ],
        columns=3,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(
            selected_dataset_distribution_chart(class_df, dataset),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        st.plotly_chart(
            top_terms_chart(top_terms, st.session_state.top_n, f"Top {st.session_state.top_n} Terms"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    section_header("Class Balance Comparison", "Perbandingan distribusi full14k dan balanced3k.")
    st.plotly_chart(
        sentiment_distribution_chart(class_df, "Perbandingan Distribusi Sentimen"),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def render_model_performance(data: dict[str, pd.DataFrame]) -> None:
    page_header(
        "Model Performance",
        "Perbandingan performa SVM sklearn, SVM Spark MLlib, dan IndoBERT.",
    )

    comp_df = data["model_comparison"]
    per_class = data["per_class_metrics"]
    cm_df = data["confusion_matrix"]

    dataset = st.session_state.selected_dataset
    selected_metric = st.session_state.selected_metric
    selected_model = st.session_state.selected_model
    metric_label = get_metric_label(selected_metric)

    best_model, best_metric_value = get_best_model(comp_df, dataset, selected_metric)

    selected_row = comp_df[
        (comp_df["dataset"].astype(str) == dataset)
        & (comp_df["model"].astype(str) == selected_model)
    ]

    selected_acc = selected_row.iloc[0]["accuracy"] if not selected_row.empty else None
    selected_macro = selected_row.iloc[0]["macro_f1"] if not selected_row.empty else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Best Model", best_model, f"Berdasarkan {metric_label}")
    with c2:
        metric_card("Best Selected Metric", format_percent(best_metric_value), dataset)
    with c3:
        metric_card("Selected Accuracy", format_percent(selected_acc), selected_model)
    with c4:
        metric_card("Selected Macro-F1", format_percent(selected_macro), selected_model)

    insight_box("Model Insight", model_performance_insight(comp_df, dataset, selected_metric))
    mini_card_grid(
        [
            ("Selected", selected_model, "Model aktif untuk confusion matrix dan analisis error."),
            ("Metric", metric_label, "Metrik fokus menentukan peringkat model pada halaman ini."),
            ("Comparison", "Three-model view", "IndoBERT, SVM sklearn, dan Spark MLlib dibandingkan dalam bahasa visual yang sama."),
        ],
        columns=3,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(
            model_metric_chart(comp_df, dataset, selected_metric),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        metric_for_class = st.selectbox(
            "Per-class metric",
            options=["f1_score", "precision", "recall"],
            index=0,
            format_func=get_metric_label,
            help="Pilih metrik per kelas yang ingin dibandingkan.",
        )
        st.plotly_chart(
            per_class_metric_chart(per_class, dataset, metric_for_class),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    section_header("Model Families", "Ringkasan pendek karakter model yang dibandingkan.")
    model_explanation_cards()

    section_header("Confusion Matrix", "Heatmap aktual vs prediksi berdasarkan dataset dan model terpilih.")
    st.plotly_chart(
        confusion_matrix_heatmap(cm_df, dataset, selected_model),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def get_active_prediction_model(data: dict[str, pd.DataFrame]) -> tuple[str, bool]:
    pred = data["predictions"]
    selected_model = st.session_state.selected_model
    if pred.empty or "model" not in pred.columns:
        return selected_model, True

    available_models = pred["model"].dropna().astype(str).unique().tolist()
    if selected_model in available_models:
        return selected_model, True

    fallback = "IndoBERT" if "IndoBERT" in available_models else available_models[0]
    return fallback, False


def filter_predictions(data: dict[str, pd.DataFrame], model: str | None = None) -> pd.DataFrame:
    pred = data["predictions"].copy()
    if pred.empty:
        return pred

    active_model = model or get_active_prediction_model(data)[0]
    if active_model in pred["model"].astype(str).unique():
        pred = pred[pred["model"].astype(str) == active_model]

    selected_sentiments = st.session_state.selected_sentiments
    if selected_sentiments:
        pred = pred[pred["actual_label"].astype(str).isin(selected_sentiments)]

    return pred


def render_error_analysis(data: dict[str, pd.DataFrame]) -> None:
    page_header(
        "Error Analysis",
        "Eksplorasi pola kesalahan klasifikasi pada test set balanced3k.",
    )

    active_model, model_is_available = get_active_prediction_model(data)
    pred = filter_predictions(data, active_model)

    if model_is_available:
        st.caption("Catatan: data prediksi detail tersedia untuk SVM sklearn dan IndoBERT pada balanced3k.")
    else:
        neutral_box(
            "Model Detail Fallback",
            f"Data prediksi detail untuk {st.session_state.selected_model} tidak tersedia. Halaman ini menampilkan detail {active_model} agar tabel, error pair, dan keyword explorer tetap dapat dianalisis.",
        )

    c_filter1, c_filter2, c_filter3 = st.columns(3)
    with c_filter1:
        actual_choice = st.selectbox("Actual label", options=["All"] + LABEL_ORDER)
    with c_filter2:
        predicted_choice = st.selectbox("Predicted label", options=["All"] + LABEL_ORDER)
    with c_filter3:
        status_choice = st.selectbox("Error status", options=["All", "Benar", "Salah"], index=2)

    if actual_choice != "All":
        pred = pred[pred["actual_label"].astype(str) == actual_choice]
    if predicted_choice != "All":
        pred = pred[pred["predicted_label"].astype(str) == predicted_choice]
    if status_choice != "All":
        pred = pred[pred["error_status"].astype(str) == status_choice]

    keyword = st.text_input("Search keyword in comments", placeholder="Contoh: ijazah, palsu, asli, roy, jokowi")
    if keyword:
        pred = pred[pred["text"].astype(str).str.contains(keyword, case=False, na=False)]

    total = len(pred)
    correct = int((pred["error_status"].astype(str) == "Benar").sum()) if not pred.empty else 0
    wrong = int((pred["error_status"].astype(str) == "Salah").sum()) if not pred.empty else 0
    acc = correct / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Filtered Test Data", format_number(total), "Sesuai filter aktif")
    with c2:
        metric_card("Correct", format_number(correct), "Prediksi benar", "rgba(52,211,153,0.14)")
    with c3:
        metric_card("Wrong", format_number(wrong), "Prediksi salah", "rgba(251,113,133,0.16)")
    with c4:
        metric_card("Accuracy", format_percent(acc), "Pada data terfilter")

    insight_box("Error Insight", error_analysis_insight(pred, active_model))

    c1, c2 = st.columns([0.85, 1.15])
    with c1:
        st.plotly_chart(
            correct_wrong_chart(pred),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        st.plotly_chart(
            error_pair_chart(pred, st.session_state.top_n),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    section_header("Filtered Comments Table", "Default menampilkan komentar salah prediksi. Gunakan filter dan search box untuk eksplorasi.")
    show_full_text = st.toggle("Show full comment text", value=False)

    text_col = "text" if show_full_text else "text_preview"
    display_cols = [
        col
        for col in ["model", "actual_label", "predicted_label", "error_status", "error_pair", "confidence", text_col]
        if col in pred.columns
    ]
    display_df = pred[display_cols].copy()
    if text_col == "text_preview":
        display_df = display_df.rename(columns={"text_preview": "text"})

    render_dataframe(display_df.head(220), height=360)
    download_dataframe_button(pred, "filtered_error_analysis.csv", "Download filtered error data")


def render_public_opinion_insight(data: dict[str, pd.DataFrame]) -> None:
    page_header(
        "Public Opinion Insight",
        "Interpretasi diskursus publik berdasarkan distribusi sentimen, term dominan, dan fitur model.",
    )

    class_df = data["class_distribution"]
    top_terms = data["top_terms"]
    top_features = data["top_features"]
    predictions = data["predictions"]
    domain_terms = get_domain_terms(top_terms, st.session_state.top_n)

    full_total = get_total_by_dataset(class_df, "full14k")
    dominant_sentiment, dominant_help = get_class_balance_status(class_df, "full14k")
    issue_term = get_issue_term(top_terms)
    top_domain_term = domain_terms.iloc[0]["term"] if not domain_terms.empty else get_top_term(top_terms)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Full14k Comments", format_number(full_total), "Distribusi asli")
    with c2:
        metric_card("Dominant Sentiment", dominant_sentiment, dominant_help)
    with c3:
        metric_card("Top Domain Term", str(top_domain_term), "Term domain non-generik")
    with c4:
        metric_card("Issue Term", issue_term, "Term isu dominan")

    insight_box("Research Insight", public_opinion_insight(top_terms))
    mini_card_grid(
        [
            ("Context", "Full14k discourse", "Distribusi asli dipakai untuk membaca kecenderungan opini publik secara makro."),
            ("Term", str(top_domain_term), "Term domain membantu mengurangi noise dari kata umum yang kurang informatif."),
            ("Ethics", "Non-verification", "Dashboard memetakan opini, bukan membuktikan benar atau salahnya substansi isu."),
        ],
        columns=3,
    )

    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        st.plotly_chart(
            full14k_distribution_chart(class_df),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        st.plotly_chart(
            top_terms_chart(domain_terms, min(st.session_state.top_n, len(domain_terms)), "Top Domain Terms"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    section_header("Top SVM Features by Sentiment", "Fitur dengan bobot terbesar pada model SVM sklearn.")
    insight_box(
        "Interpretation Note",
        "Top features merepresentasikan bobot model SVM, bukan kesimpulan normatif terhadap benar atau salahnya isu.",
    )
    sentiment_focus = st.selectbox("Sentiment focus", options=LABEL_ORDER, index=0)
    st.plotly_chart(
        top_features_chart(top_features, sentiment_focus, st.session_state.top_n),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    section_header("Keyword Explorer", "Cari komentar yang memuat kata tertentu pada test set balanced3k.")
    keyword = st.text_input("Keyword explorer", placeholder="Contoh: ijazah, palsu, asli, roy, jokowi", key="opinion_keyword")
    if keyword:
        subset = predictions[predictions["text"].astype(str).str.contains(keyword, case=False, na=False)].copy()
        st.caption(f"Ditemukan {len(subset)} komentar pada test set balanced3k yang memuat keyword `{keyword}`.")
        display_cols = [col for col in ["model", "actual_label", "predicted_label", "error_status", "text_preview"] if col in subset.columns]
        display_df = subset[display_cols].copy().rename(columns={"text_preview": "text"})
        render_dataframe(display_df.head(220), height=330)
        download_dataframe_button(subset, f"keyword_{keyword}_comments.csv", "Download keyword comments")
    else:
        st.info("Masukkan keyword untuk mengeksplorasi komentar, misalnya `ijazah`, `palsu`, `asli`, `roy`, atau `jokowi`.")


def render_methodology(data: dict[str, pd.DataFrame]) -> None:
    page_header(
        "Methodology",
        "Ringkasan metodologi, definisi label, pipeline, dan catatan etika penelitian.",
    )

    warning_box(
        "Ethical Note",
        "Analisis ini tidak bertujuan membuktikan benar atau salahnya isu, tetapi menganalisis pola opini publik pada komentar YouTube.",
    )

    section_header("Research Title")
    research_scope_card(
        "Analisis Sentimen Opini Publik terhadap Kontroversi Ijazah Jokowi di Platform YouTube",
        "Fokus penelitian adalah mengklasifikasikan dan menganalisis opini publik pada komentar YouTube terhadap narasi isu, bukan melakukan verifikasi fakta terhadap substansi isu.",
    )

    section_header("End-to-End Pipeline")
    pipeline_step_grid()

    section_header("Compared Models")
    model_explanation_cards()

    section_header("Label Definition")
    label_definition_cards()

    section_header("Reproducibility Note")
    insight_box(
        "Reproducible Data Mart",
        "Dashboard membaca file CSV dari folder outputs/tableau. Dengan demikian, visualisasi dapat direproduksi dengan menjalankan ulang pipeline export tanpa melakukan training ulang saat presentasi.",
    )

    manifest = data.get("manifest", pd.DataFrame())
    with st.expander("Show Data Mart Manifest", expanded=False):
        if not manifest.empty:
            render_dataframe(manifest, height=360)
        else:
            st.info("Manifest tidak tersedia.")


def main() -> None:
    data = load_all_data(str(get_tableau_dir()))
    init_session_state(data)
    page = render_sidebar(data)

    missing = [
        key for key, df in data.items()
        if key != "manifest" and isinstance(df, pd.DataFrame) and df.empty
    ]
    if missing:
        warning_box(
            "Data Warning",
            "Beberapa file data kosong atau tidak ditemukan: " + ", ".join(missing) + ". Periksa folder outputs/tableau.",
        )

    if page == "Executive Overview":
        render_executive_overview(data)
    elif page == "Dataset Explorer":
        render_dataset_explorer(data)
    elif page == "Model Performance":
        render_model_performance(data)
    elif page == "Error Analysis":
        render_error_analysis(data)
    elif page == "Public Opinion Insight":
        render_public_opinion_insight(data)
    elif page == "Methodology":
        render_methodology(data)


if __name__ == "__main__":
    main()

from __future__ import annotations

from html import escape
from pathlib import Path
import textwrap

import pandas as pd
import streamlit as st

from .data_loader import dataframe_to_csv_bytes


def _clean_html(html: str) -> str:
    """Return compact HTML so Streamlit never treats indented markup as a code block."""
    return " ".join(textwrap.dedent(html).strip().split())


def _safe(value: object) -> str:
    return escape(str(value), quote=True)


def load_css() -> None:
    css_path = Path(__file__).resolve().parents[1] / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        _clean_html(
            """
            <div class="sidebar-brand">
                <span class="brand-mark" aria-hidden="true"></span>
                <span class="brand-copy">
                    <span class="brand-title">YPO Analytics</span>
                    <span class="brand-caption">YouTube Public Opinion Intelligence</span>
                </span>
                <span class="brand-version">V5</span>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def sidebar_section(title: str, caption: str | None = None) -> None:
    caption_html = f'<span class="sidebar-section-caption">{_safe(caption)}</span>' if caption else ""
    st.sidebar.markdown(
        _clean_html(
            f"""
            <div class="sidebar-section-heading">
                <span class="sidebar-section-title">{_safe(title)}</span>
                {caption_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, eyebrow: str = "Sentiment Intelligence Platform") -> None:
    st.markdown(
        _clean_html(
            f"""
            <section class="hero-panel" aria-label="page header">
                <div class="hero-glow"></div>
                <div class="eyebrow">{_safe(eyebrow)}</div>
                <div class="page-title">{_safe(title)}</div>
                <div class="page-subtitle">{_safe(subtitle)}</div>
                <div class="status-row" aria-label="dashboard capability badges">
                    <span class="status-pill"><span class="pill-dot"></span>Data Mart Ready</span>
                    <span class="status-pill"><span class="pill-dot"></span>Interactive Analytics</span>
                    <span class="status-pill"><span class="pill-dot"></span>Reproducible Pipeline</span>
                </div>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str | None = None) -> None:
    st.markdown(
        _clean_html(
            f"""
            <div class="section-heading">
                <div class="section-kicker"></div>
                <div>
                    <div class="section-title">{_safe(title)}</div>
                    {f'<div class="section-caption">{_safe(caption)}</div>' if caption else ''}
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    help_text: str = "",
    accent: str = "rgba(56, 189, 248, 0.18)",
    icon: str | None = None,
) -> None:
    icon_html = f'<span class="metric-icon">{_safe(icon)}</span>' if icon else ""
    st.markdown(
        _clean_html(
            f"""
            <div class="metric-card" style="--accent: {accent};">
                <div class="metric-topline">
                    <div class="metric-label">{_safe(label)}</div>
                    {icon_html}
                </div>
                <div class="metric-value">{_safe(value)}</div>
                <div class="metric-help">{_safe(help_text)}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def insight_box(title: str, body: str) -> None:
    _callout_box("insight-box", title, body)


def warning_box(title: str, body: str) -> None:
    _callout_box("warning-box", title, body)


def neutral_box(title: str, body: str) -> None:
    _callout_box("neutral-box", title, body)


def _callout_box(class_name: str, title: str, body: str) -> None:
    st.markdown(
        _clean_html(
            f"""
            <div class="{class_name}">
                <span class="insight-title">{_safe(title)}</span>
                <span class="insight-body">{_safe(body)}</span>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def mini_card_grid(items: list[tuple[str, str, str]], columns: int = 3) -> None:
    """Render compact narrative cards. Each item is (label, title, body)."""
    cards = "".join(
        f"<div class='mini-card'><div class='mini-label'>{_safe(label)}</div>"
        f"<div class='mini-title'>{_safe(title)}</div><p>{_safe(body)}</p></div>"
        for label, title, body in items
    )
    st.markdown(
        f"<div class='mini-card-grid columns-{columns}'>{cards}</div>",
        unsafe_allow_html=True,
    )


def research_scope_card(title: str, body: str) -> None:
    st.markdown(
        _clean_html(
            f"""
            <div class="research-card">
                <div class="research-label">Research Scope</div>
                <div class="research-title">{_safe(title)}</div>
                <p>{_safe(body)}</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def download_dataframe_button(df: pd.DataFrame, file_name: str, label: str = "Download CSV") -> None:
    if df.empty:
        st.caption("Tidak ada data untuk diunduh.")
        return

    st.download_button(
        label=label,
        data=dataframe_to_csv_bytes(df),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def render_dataframe(df: pd.DataFrame, height: int = 420) -> None:
    column_config: dict[str, object] = {}
    if "confidence" in df.columns:
        column_config["confidence"] = st.column_config.NumberColumn(
            "confidence",
            format="%.4f",
            help="Confidence score dari model, jika tersedia.",
        )
    if "text" in df.columns:
        column_config["text"] = st.column_config.TextColumn(
            "text",
            width="large",
            help="Komentar YouTube yang sudah masuk data mart.",
        )
    for column in ["actual_label", "predicted_label", "error_status", "error_pair", "model"]:
        if column in df.columns:
            column_config[column] = st.column_config.TextColumn(column, width="small")

    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        hide_index=True,
        column_config=column_config,
    )


def pipeline_step_grid() -> None:
    steps = [
        ("01", "YouTube API", "Mengambil komentar dan metadata video."),
        ("02", "MongoDB Atlas", "Menyimpan data mentah dan hasil proses."),
        ("03", "Preprocessing", "Cleaning, case folding, tokenisasi, dan normalisasi teks."),
        ("04", "Labeling", "Memberikan label sentimen berdasarkan rubrik anotasi."),
        ("05", "Balanced Dataset", "Menyusun 1.000 komentar per kelas sentimen."),
        ("06", "Modeling", "Melatih SVM sklearn, Spark MLlib, dan IndoBERT."),
        ("07", "Evaluation", "Menghasilkan metrik, confusion matrix, dan error pair."),
        ("08", "Analytics App", "Menyajikan insight secara interaktif dan reprodusibel."),
    ]
    cards = "".join(
        f"<div class='step-card'><div class='step-number'>{_safe(number)}</div>"
        f"<div class='step-title'>{_safe(title)}</div><div class='step-caption'>{_safe(caption)}</div></div>"
        for number, title, caption in steps
    )
    st.markdown(f"<div class='step-grid'>{cards}</div>", unsafe_allow_html=True)


def label_definition_cards() -> None:
    cards = [
        ("Negatif", "Komentar yang menolak, membantah, atau melemahkan narasi isu.", "negative"),
        ("Netral", "Komentar informatif, bertanya, atau tidak menunjukkan sikap jelas.", "neutral"),
        ("Positif", "Komentar yang mendukung, mempercayai, atau memperkuat narasi isu.", "positive"),
    ]
    html = "".join(
        f"<div class='label-card {tone}'><div class='step-title'>{_safe(title)}</div><p>{_safe(body)}</p></div>"
        for title, body, tone in cards
    )
    st.markdown(f"<div class='label-grid'>{html}</div>", unsafe_allow_html=True)


def model_explanation_cards() -> None:
    cards = [
        ("IndoBERT", "Transformer bahasa Indonesia untuk menangkap konteks semantik komentar."),
        ("SVM sklearn", "Baseline klasik berbasis fitur teks. Cepat, stabil, dan tetap kompetitif."),
        ("SVM Spark MLlib", "Distributed machine learning pipeline untuk aspek pemrosesan Big Data."),
    ]
    html = "".join(
        f"<div class='model-card'><div class='step-title'>{_safe(title)}</div><p>{_safe(body)}</p></div>"
        for title, body in cards
    )
    st.markdown(f"<div class='model-grid'>{html}</div>", unsafe_allow_html=True)

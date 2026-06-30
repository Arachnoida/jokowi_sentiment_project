from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data_loader import format_compact, get_metric_label

LABEL_ORDER = ["Negatif", "Netral", "Positif"]

SENTIMENT_COLORS = {
    "Negatif": "#F46D84",
    "Netral": "#9AA8BA",
    "Positif": "#34D399",
}

MODEL_COLORS = {
    "IndoBERT": "#38BDF8",
    "SVM sklearn": "#A78BFA",
    "SVM Spark MLlib": "#F59E0B",
}

STATUS_COLORS = {
    "Benar": "#34D399",
    "Salah": "#F46D84",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "ypo_analytics_chart",
        "height": 720,
        "width": 1280,
        "scale": 2,
    },
}

FONT_FAMILY = '"Segoe UI", Inter, Arial, sans-serif'
GRID_COLOR = "rgba(148,163,184,0.105)"
AXIS_COLOR = "#CBD5E1"
TITLE_COLOR = "#F8FAFC"


def apply_common_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=48, r=46, t=58, b=42),
        font=dict(family=FONT_FAMILY, color="#E5E7EB", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.05)",
        bargap=0.36,
        bargroupgap=0.16,
        uniformtext=dict(minsize=10, mode="show"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.025,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color="#CBD5E1", size=11, family=FONT_FAMILY),
            itemclick=False,
            itemdoubleclick=False,
        ),
        legend_title_text="",
        title=dict(
            font=dict(size=14, color=TITLE_COLOR, family=FONT_FAMILY, weight=700),
            x=0.015,
            y=0.97,
            xanchor="left",
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#334155",
            font_size=12,
            font_family=FONT_FAMILY,
            font_color="#E5E7EB",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        color=AXIS_COLOR,
        title_font=dict(size=12, color=AXIS_COLOR, family=FONT_FAMILY),
        tickfont=dict(size=11, color=AXIS_COLOR, family=FONT_FAMILY),
        linecolor="rgba(148,163,184,0.12)",
        ticks="",
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        color=AXIS_COLOR,
        title_font=dict(size=12, color=AXIS_COLOR, family=FONT_FAMILY),
        tickfont=dict(size=11, color=AXIS_COLOR, family=FONT_FAMILY),
        linecolor="rgba(148,163,184,0.12)",
        ticks="",
    )
    return fig


def _safe_max(series: pd.Series, default: float = 1.0) -> float:
    if series.empty:
        return default
    value = pd.to_numeric(series, errors="coerce").max()
    if pd.isna(value) or value <= 0:
        return default
    return float(value)


def _percent_axis(fig: go.Figure, max_value: float | None = None) -> None:
    upper = 1.04 if max_value is None else min(1.08, max(max_value * 1.18, 0.12))
    fig.update_yaxes(range=[0, upper], tickformat=".0%")


def sentiment_distribution_chart(df: pd.DataFrame, title: str = "Distribusi sentimen") -> go.Figure:
    if df.empty:
        return empty_figure("Data distribusi sentimen tidak tersedia.")

    plot_df = df.copy()
    plot_df["label"] = plot_df["label"].astype(str)
    plot_df["count_label"] = plot_df["count"].map(format_compact)

    fig = px.bar(
        plot_df,
        x="dataset",
        y="count",
        color="label",
        text="count_label",
        category_orders={"label": LABEL_ORDER, "dataset": ["full14k", "balanced3k"]},
        color_discrete_map=SENTIMENT_COLORS,
        title=title,
    )
    fig.update_traces(
        textposition="inside",
        textfont=dict(color="#0B1220", size=10, family=FONT_FAMILY),
        marker_line_width=0,
        opacity=0.92,
        hovertemplate="Dataset=%{x}<br>Sentimen=%{fullData.name}<br>Jumlah=%{y:,}<extra></extra>",
    )
    fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="Jumlah komentar")
    return apply_common_layout(fig, height=368)


def selected_dataset_distribution_chart(df: pd.DataFrame, dataset: str) -> go.Figure:
    if df.empty:
        return empty_figure("Data tidak tersedia.")

    plot_df = df[df["dataset"].astype(str) == dataset].copy()
    plot_df["label"] = plot_df["label"].astype(str)
    plot_df["count_label"] = plot_df["count"].map(format_compact)

    fig = px.bar(
        plot_df,
        x="label",
        y="count",
        color="label",
        text="count_label",
        category_orders={"label": LABEL_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title=f"Distribusi Sentimen pada {dataset}",
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=11, color="#E5E7EB", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Sentimen=%{x}<br>Jumlah=%{y:,}<extra></extra>",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah komentar")
    fig.update_yaxes(range=[0, _safe_max(plot_df["count"]) * 1.18])
    return apply_common_layout(fig, height=348)


def top_terms_chart(df: pd.DataFrame, top_n: int = 10, title: str = "Top terms") -> go.Figure:
    if df.empty:
        return empty_figure("Data top terms tidak tersedia.")

    plot_df = df.sort_values("rank").head(top_n).copy()
    plot_df = plot_df.sort_values("count", ascending=True)
    plot_df["count_label"] = plot_df["count"].map(format_compact)

    fig = px.bar(
        plot_df,
        x="count",
        y="term",
        orientation="h",
        text="count_label",
        title=title,
    )
    fig.update_traces(
        textposition="outside",
        marker_color="#38BDF8",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=10, color="#DFF6FF", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Term=%{y}<br>Frekuensi=%{x:,}<extra></extra>",
    )
    max_count = _safe_max(plot_df["count"])
    fig.update_layout(xaxis_title="Frekuensi", yaxis_title="", showlegend=False)
    fig.update_xaxes(range=[0, max_count * 1.18])
    return apply_common_layout(fig, height=max(318, 25 * len(plot_df) + 104))


def model_metric_chart(df: pd.DataFrame, dataset: str, metric: str) -> go.Figure:
    if df.empty:
        return empty_figure("Data perbandingan model tidak tersedia.")

    plot_df = df[df["dataset"].astype(str) == dataset].copy()
    if metric not in plot_df.columns:
        return empty_figure(f"Metrik {get_metric_label(metric)} tidak tersedia.")

    plot_df = plot_df.sort_values(metric, ascending=False)
    metric_label = get_metric_label(metric)

    fig = px.bar(
        plot_df,
        x="model",
        y=metric,
        color="model",
        text=metric,
        color_discrete_map=MODEL_COLORS,
        title=f"Peringkat Model Berdasarkan {metric_label}",
    )
    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=10, color="#E5E7EB", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Model=%{x}<br>Skor=%{y:.2%}<extra></extra>",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Skor", showlegend=False)
    _percent_axis(fig, _safe_max(plot_df[metric]))
    return apply_common_layout(fig, height=366)


def model_comparison_dual_chart(df: pd.DataFrame, dataset: str) -> go.Figure:
    if df.empty:
        return empty_figure("Data model comparison tidak tersedia.")

    plot_df = df[df["dataset"].astype(str) == dataset].copy()
    plot_df = plot_df.sort_values("macro_f1", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["model"],
            y=plot_df["accuracy"],
            name="Accuracy",
            text=[f"{v:.2%}" for v in plot_df["accuracy"]],
            textposition="outside",
            marker=dict(color="#38BDF8", line=dict(width=0)),
            textfont=dict(size=10, color="#E5E7EB", family=FONT_FAMILY),
            cliponaxis=False,
            hovertemplate="Model=%{x}<br>Accuracy=%{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["model"],
            y=plot_df["macro_f1"],
            name="Macro-F1",
            text=[f"{v:.2%}" for v in plot_df["macro_f1"]],
            textposition="outside",
            marker=dict(color="#A78BFA", line=dict(width=0)),
            textfont=dict(size=10, color="#E5E7EB", family=FONT_FAMILY),
            cliponaxis=False,
            hovertemplate="Model=%{x}<br>Macro-F1=%{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Accuracy dan Macro-F1 pada {dataset}",
        barmode="group",
        xaxis_title="",
        yaxis_title="Skor",
    )
    _percent_axis(fig)
    return apply_common_layout(fig, height=368)


def per_class_metric_chart(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    models: list[str] | None = None,
) -> go.Figure:
    if df.empty:
        return empty_figure("Data per-class metrics tidak tersedia.")

    plot_df = df[df["dataset"].astype(str) == dataset].copy()
    if models:
        plot_df = plot_df[plot_df["model"].astype(str).isin(models)]

    if metric not in plot_df.columns:
        return empty_figure(f"Metrik {get_metric_label(metric)} tidak tersedia.")

    plot_df["label"] = plot_df["label"].astype(str)
    metric_label = get_metric_label(metric)

    fig = px.bar(
        plot_df,
        x="label",
        y=metric,
        color="model",
        barmode="group",
        text=metric,
        category_orders={"label": LABEL_ORDER, "model": ["IndoBERT", "SVM sklearn", "SVM Spark MLlib"]},
        color_discrete_map=MODEL_COLORS,
        title=f"{metric_label} per Kelas Sentimen",
    )
    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=10, color="#E5E7EB", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Label=%{x}<br>Skor=%{y:.2%}<extra></extra>",
    )
    fig.update_layout(xaxis_title="", yaxis_title=metric_label)
    _percent_axis(fig)
    return apply_common_layout(fig, height=366)


def confusion_matrix_heatmap(df: pd.DataFrame, dataset: str, model: str) -> go.Figure:
    if df.empty:
        return empty_figure("Data confusion matrix tidak tersedia.")

    plot_df = df[
        (df["dataset"].astype(str) == dataset)
        & (df["model"].astype(str) == model)
    ].copy()

    if plot_df.empty:
        return empty_figure(f"Confusion matrix untuk {model} pada {dataset} tidak tersedia.")

    matrix = (
        plot_df.pivot_table(
            index="actual_label",
            columns="predicted_label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.astype(str),
            y=matrix.index.astype(str),
            colorscale=[[0, "#06111F"], [0.36, "#0E3B4E"], [0.72, "#2894B8"], [1, "#BAE6FD"]],
            colorbar=dict(title="Count", thickness=12, len=0.74, tickfont=dict(color=AXIS_COLOR)),
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
        )
    )

    max_value = matrix.values.max() if matrix.size else 0
    threshold = max_value * 0.58 if max_value else 0
    for row_idx, actual in enumerate(matrix.index.astype(str)):
        for col_idx, predicted in enumerate(matrix.columns.astype(str)):
            value = int(matrix.iloc[row_idx, col_idx])
            fig.add_annotation(
                x=predicted,
                y=actual,
                text=format_compact(value),
                showarrow=False,
                font=dict(
                    color="#07111F" if value >= threshold else "#DDEBFF",
                    size=13,
                    family=FONT_FAMILY,
                ),
            )

    fig.update_layout(
        title=f"Confusion Matrix: {model} | {dataset}",
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
    )
    fig.update_xaxes(side="top")
    return apply_common_layout(fig, height=390)


def correct_wrong_chart(df: pd.DataFrame, title: str = "Prediksi Benar vs Salah") -> go.Figure:
    if df.empty:
        return empty_figure("Data prediksi tidak tersedia.")

    plot_df = df.groupby("error_status", as_index=False).agg(count=("comment_id", "count"))
    order = ["Benar", "Salah"]
    plot_df["error_status"] = pd.Categorical(plot_df["error_status"], order, ordered=True)
    plot_df = plot_df.sort_values("error_status")
    plot_df["count_label"] = plot_df["count"].map(format_compact)

    fig = px.bar(
        plot_df,
        x="error_status",
        y="count",
        color="error_status",
        text="count_label",
        color_discrete_map=STATUS_COLORS,
        title=title,
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=10, color="#E5E7EB", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Status=%{x}<br>Jumlah=%{y:,}<extra></extra>",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Jumlah komentar", showlegend=False)
    fig.update_yaxes(range=[0, _safe_max(plot_df["count"]) * 1.18])
    return apply_common_layout(fig, height=318)


def error_pair_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if df.empty:
        return empty_figure("Data error tidak tersedia.")

    wrong_df = df[df["error_status"].astype(str) == "Salah"].copy()
    if wrong_df.empty:
        return empty_figure("Tidak ada prediksi salah pada filter ini.")

    plot_df = (
        wrong_df.groupby("error_pair", as_index=False)
        .agg(count=("comment_id", "count"))
        .sort_values("count", ascending=False)
        .head(top_n)
        .sort_values("count", ascending=True)
    )
    plot_df["count_label"] = plot_df["count"].map(format_compact)

    fig = px.bar(
        plot_df,
        x="count",
        y="error_pair",
        orientation="h",
        text="count_label",
        title="Error Pair Terbanyak",
    )
    fig.update_traces(
        textposition="outside",
        marker_color="#F46D84",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=10, color="#FFE4EA", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Error pair=%{y}<br>Jumlah=%{x:,}<extra></extra>",
    )
    max_count = _safe_max(plot_df["count"])
    fig.update_layout(xaxis_title="Jumlah kesalahan", yaxis_title="")
    fig.update_xaxes(range=[0, max_count * 1.18])
    return apply_common_layout(fig, height=max(318, 25 * len(plot_df) + 102))


def top_features_chart(df: pd.DataFrame, label: str, top_n: int = 15) -> go.Figure:
    if df.empty:
        return empty_figure("Data top features tidak tersedia.")

    plot_df = df[df["label"].astype(str) == label].copy()
    plot_df = plot_df.sort_values("rank").head(top_n).sort_values("weight", ascending=True)

    if plot_df.empty:
        return empty_figure(f"Top features untuk label {label} tidak tersedia.")

    fig = px.bar(
        plot_df,
        x="weight",
        y="feature",
        orientation="h",
        text="weight",
        title=f"Top SVM Features untuk Kelas {label}",
        color_discrete_sequence=[SENTIMENT_COLORS.get(label, "#38BDF8")],
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=10, color="#E5E7EB", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Fitur=%{y}<br>Bobot=%{x:.3f}<extra></extra>",
    )
    max_weight = _safe_max(plot_df["weight"])
    fig.update_layout(xaxis_title="Bobot koefisien", yaxis_title="", showlegend=False)
    fig.update_xaxes(range=[0, max_weight * 1.14])
    return apply_common_layout(fig, height=max(330, 25 * len(plot_df) + 108))


def full14k_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Data distribusi tidak tersedia.")

    plot_df = df[df["dataset"].astype(str) == "full14k"].copy()
    plot_df["label"] = plot_df["label"].astype(str)
    plot_df["count_label"] = plot_df["count"].map(format_compact)

    fig = px.bar(
        plot_df,
        x="label",
        y="count",
        color="label",
        text="count_label",
        category_orders={"label": LABEL_ORDER},
        color_discrete_map=SENTIMENT_COLORS,
        title="Distribusi Sentimen pada Full14k",
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        opacity=0.92,
        textfont=dict(size=10, color="#E5E7EB", family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="Sentimen=%{x}<br>Jumlah=%{y:,}<extra></extra>",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Jumlah komentar", showlegend=False)
    fig.update_yaxes(range=[0, _safe_max(plot_df["count"]) * 1.16])
    return apply_common_layout(fig, height=348)


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#94A3B8", family=FONT_FAMILY),
    )
    fig.update_layout(
        height=330,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=35, b=20),
        font=dict(family=FONT_FAMILY, color="#E5E7EB", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

# app.py
# Streamlit dashboard for analyzing three NDJSON datasets:
#   1) sampled.ndjson        (submissions)
#   2) labels_3000.ndjson    (comment-level labels, nested JSON)
#   3) thread_metrics.ndjson (thread-level metrics, nested JSON)
#
# Run:
#   pip install -r requirements.txt
#   python -m streamlit run app.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEFAULT_SUBMISSIONS_PATH = "sampled.ndjson"
DEFAULT_LABELS_PATH = "labels_3000.ndjson"
DEFAULT_THREAD_PATH = "thread_metrics.ndjson"

st.set_page_config(page_title="NDJSON Dashboard", layout="wide")


# -----------------------------
# Helpers: parsing / flattening
# -----------------------------
def _maybe_epoch_to_datetime(s: pd.Series) -> pd.Series:
    """
    Convert epoch seconds or epoch milliseconds to pandas datetime (UTC).
    Heuristic: values > 1e12 -> milliseconds.
    """
    if s is None or s.empty:
        return s
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.notna().sum() == 0:
        return s
    median_val = float(s2.dropna().median())
    unit = "ms" if median_val > 1e12 else "s"
    return pd.to_datetime(s2, unit=unit, utc=True, errors="coerce")


def _json_normalize_lines(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load NDJSON and flatten nested objects using pandas.json_normalize.
    Works well for labels/thread_metrics where result/meta are nested.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows, sep=".")
    return df


def _stringify_unhashables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list/dict/set/tuple values in object columns to stable JSON strings.
    This avoids crashes in pandas operations like duplicated(), nunique(), value_counts(), etc.
    """
    def make_hashable(x: Any) -> Any:
        if isinstance(x, (list, dict, set, tuple)):
            try:
                return json.dumps(x, sort_keys=True, ensure_ascii=False)
            except TypeError:
                return str(x)
        return x

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        df[c] = df[c].map(make_hashable)
    return df


def _safe_duplicated_count(df: pd.DataFrame) -> int:
    """
    Count duplicate rows robustly even if the dataframe contains list/dict cells.
    """
    if df.empty:
        return 0
    work = df.copy()
    work = _stringify_unhashables(work)
    return int(work.duplicated().sum())


@st.cache_data(show_spinner=False)
def load_submissions(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    if "created_utc" in df.columns:
        df["created_dt"] = _maybe_epoch_to_datetime(df["created_utc"])

    for c in ["title", "selftext"]:
        if c in df.columns:
            df[f"{c}_len"] = df[c].astype(str).str.len().replace({0: np.nan})

    # Ensure robustness if any nested field exists (rare here, but cheap insurance)
    df = _stringify_unhashables(df)
    return df


@st.cache_data(show_spinner=False)
def load_labels(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    df = _json_normalize_lines(path, max_rows=max_rows)

    for col in ["meta.created_utc", "meta.submission_created_utc"]:
        if col in df.columns:
            df[col.replace("meta.", "meta_dt.")] = _maybe_epoch_to_datetime(df[col])

    # Generic numeric value field across tasks:
    # - stance_intensity: result.score
    # - civility: result.label
    if "result.score" in df.columns and "result.label" in df.columns:
        df["result_value"] = pd.to_numeric(df["result.score"], errors="coerce").fillna(
            pd.to_numeric(df["result.label"], errors="coerce")
        )
    elif "result.score" in df.columns:
        df["result_value"] = pd.to_numeric(df["result.score"], errors="coerce")
    elif "result.label" in df.columns:
        df["result_value"] = pd.to_numeric(df["result.label"], errors="coerce")

    if "result.confidence" in df.columns:
        df["result_confidence"] = pd.to_numeric(df["result.confidence"], errors="coerce")

    if "body" in df.columns:
        df["body_len"] = df["body"].astype(str).str.len().replace({0: np.nan})

    # Crucial: remove unhashable types for stable pandas operations
    df = _stringify_unhashables(df)
    return df


@st.cache_data(show_spinner=False)
def load_thread_metrics(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    df = _json_normalize_lines(path, max_rows=max_rows)

    for col in ["meta.created_utc", "meta.submission_created_utc"]:
        if col in df.columns:
            df[col.replace("meta.", "meta_dt.")] = _maybe_epoch_to_datetime(df[col])

    if "result.score" in df.columns:
        df["metric_score"] = pd.to_numeric(df["result.score"], errors="coerce")
    if "result.confidence" in df.columns:
        df["metric_confidence"] = pd.to_numeric(df["result.confidence"], errors="coerce")

    if "body" in df.columns:
        df["body_len"] = df["body"].astype(str).str.len().replace({0: np.nan})

    # Crucial: remove unhashable types for stable pandas operations
    df = _stringify_unhashables(df)
    return df


def _infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _infer_categorical_cols(df: pd.DataFrame, max_unique: int = 200) -> List[str]:
    cat_cols: List[str] = []
    for c in df.columns:
        if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= max_unique:
                cat_cols.append(c)
    return cat_cols


def _kpi_row(items: List[Tuple[str, Any]]) -> None:
    cols = st.columns(len(items))
    for i, (label, value) in enumerate(items):
        cols[i].metric(label, value)


def _missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"missing_share": miss, "missing_pct": (miss * 100).round(2)})


def _safe_topn(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=True).head(n)
    return vc.rename_axis(col).reset_index(name="count")


def _apply_common_filters(
    df: pd.DataFrame,
    *,
    subreddit_col: Optional[str],
    topic_col: Optional[str],
    dt_col: Optional[str],
    numeric_cols: List[str],
    text_search_cols: List[str],
    sidebar_prefix: str,
) -> pd.DataFrame:
    """
    Apply a standardized set of filters from the sidebar.
    Reused across datasets to keep UX consistent.
    """
    out = df

    # Subreddit filter
    if subreddit_col and subreddit_col in out.columns:
        subs = sorted([x for x in out[subreddit_col].dropna().unique().tolist() if str(x).strip() != ""])
        if subs:
            sel = st.sidebar.multiselect(
                f"{sidebar_prefix} Subreddit",
                options=subs,
                default=[],
                help="Filter to one or more subreddits. Empty means: no filter.",
            )
            if sel:
                out = out[out[subreddit_col].isin(sel)]

    # Topic filter
    if topic_col and topic_col in out.columns:
        topics = sorted([x for x in out[topic_col].dropna().unique().tolist() if str(x).strip() != ""])
        if topics:
            sel = st.sidebar.multiselect(
                f"{sidebar_prefix} Topic",
                options=topics,
                default=[],
                help="Filter to one or more topics. Empty means: no filter.",
            )
            if sel:
                out = out[out[topic_col].isin(sel)]

    # Datetime range filter
    if dt_col and dt_col in out.columns:
        dt = pd.to_datetime(out[dt_col], utc=True, errors="coerce").dropna()
        if not dt.empty:
            min_dt, max_dt = dt.min(), dt.max()
            start, end = st.sidebar.date_input(
                f"{sidebar_prefix} Date range (UTC)",
                value=(min_dt.date(), max_dt.date()),
            )
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            out = out[(pd.to_datetime(out[dt_col], utc=True, errors="coerce") >= start_ts) &
                      (pd.to_datetime(out[dt_col], utc=True, errors="coerce") <= end_ts)]

    # Numeric filters
    with st.sidebar.expander(f"{sidebar_prefix} Numeric filters", expanded=False):
        pick = st.multiselect(
            "Columns",
            options=numeric_cols,
            default=[],
            help="Pick numeric columns to filter by min/max.",
            key=f"{sidebar_prefix}_numcols",
        )
        for c in pick:
            series = pd.to_numeric(out[c], errors="coerce").dropna()
            if series.empty:
                continue
            lo, hi = float(series.min()), float(series.max())
            if lo == hi:
                continue
            v = st.slider(
                c,
                min_value=lo,
                max_value=hi,
                value=(lo, hi),
                key=f"{sidebar_prefix}_slider_{c}",
            )
            out = out[
                (pd.to_numeric(out[c], errors="coerce") >= v[0])
                & (pd.to_numeric(out[c], errors="coerce") <= v[1])
            ]

    # Text search filter
    if text_search_cols:
        q = st.sidebar.text_input(
            f"{sidebar_prefix} Text search",
            value="",
            help=f"Search substring across: {', '.join(text_search_cols[:6])}{'...' if len(text_search_cols) > 6 else ''}",
        ).strip()
        if q:
            mask = pd.Series(False, index=out.index)
            for c in text_search_cols:
                if c in out.columns:
                    mask = mask | out[c].astype(str).str.contains(q, case=False, na=False)
            out = out[mask]

    return out


def chart_builder(df: pd.DataFrame, *, title: str) -> None:
    """
    Flexible chart builder: choose chart type and axes/columns interactively.
    """
    if df.empty:
        st.info("No rows after filtering.")
        return

    numeric_cols = _infer_numeric_cols(df)
    cat_cols = _infer_categorical_cols(df)

    st.subheader(title)

    with st.expander("Chart builder", expanded=True):
        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Box",
                "Scatter",
                "Bar (Top-N categories)",
                "Time series (count)",
                "Time series (mean of numeric)",
            ],
        )

        if chart_type == "Histogram":
            x = st.selectbox("Column", options=numeric_cols if numeric_cols else df.columns.tolist())
            bins = st.slider("Bins", 10, 200, 50)
            st.plotly_chart(px.histogram(df, x=x, nbins=bins), use_container_width=True)

        elif chart_type == "Box":
            y = st.selectbox("Numeric column", options=numeric_cols if numeric_cols else df.columns.tolist())
            x = st.selectbox("Group by (optional)", options=["(none)"] + cat_cols)
            fig = px.box(df, y=y) if x == "(none)" else px.box(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for scatter.")
            else:
                x = st.selectbox("X", options=numeric_cols, index=0)
                y = st.selectbox("Y", options=numeric_cols, index=1)
                color = st.selectbox("Color (optional)", options=["(none)"] + cat_cols)
                fig = px.scatter(df, x=x, y=y, color=None if color == "(none)" else color, hover_data=cat_cols[:6])
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar (Top-N categories)":
            if not cat_cols:
                st.warning("No suitable categorical columns detected (object columns with limited cardinality).")
            else:
                c = st.selectbox("Category column", options=cat_cols)
                n = st.slider("Top N", 5, 50, 20)
                top = _safe_topn(df, c, n=n)
                st.plotly_chart(px.bar(top, x=c, y="count"), use_container_width=True)

        elif chart_type == "Time series (count)":
            dt_candidates = [c for c in df.columns if "dt" in c.lower() or "date" in c.lower()]
            dt_col = st.selectbox("Datetime column", options=dt_candidates if dt_candidates else df.columns.tolist())
            freq = st.selectbox("Resample frequency", ["H", "D", "W", "M"])
            if dt_col in df.columns:
                tmp = df[[dt_col]].copy()
                tmp[dt_col] = pd.to_datetime(tmp[dt_col], utc=True, errors="coerce")
                tmp = tmp.dropna()
                if tmp.empty:
                    st.warning("Selected datetime column cannot be parsed.")
                else:
                    ts = tmp.set_index(dt_col).resample(freq).size().reset_index(name="count")
                    st.plotly_chart(px.line(ts, x=dt_col, y="count"), use_container_width=True)

        elif chart_type == "Time series (mean of numeric)":
            dt_candidates = [c for c in df.columns if "dt" in c.lower() or "date" in c.lower()]
            dt_col = st.selectbox("Datetime column", options=dt_candidates if dt_candidates else df.columns.tolist())
            metric = st.selectbox("Numeric metric", options=numeric_cols if numeric_cols else df.columns.tolist())
            freq = st.selectbox("Resample frequency", ["H", "D", "W", "M"])
            if dt_col in df.columns:
                tmp = df[[dt_col, metric]].copy()
                tmp[dt_col] = pd.to_datetime(tmp[dt_col], utc=True, errors="coerce")
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                tmp = tmp.dropna()
                if tmp.empty:
                    st.warning("Selected columns cannot be parsed as datetime/numeric.")
                else:
                    ts = tmp.set_index(dt_col)[metric].resample(freq).mean().reset_index()
                    st.plotly_chart(px.line(ts, x=dt_col, y=metric), use_container_width=True)


def quantitative_overview(df: pd.DataFrame, *, title: str, dt_cols_hint: Optional[List[str]] = None) -> None:
    """
    Standard quantitative diagnostics: KPIs, descriptive stats, missingness, correlations.
    """
    st.subheader(title)

    _kpi_row(
        [
            ("Rows", f"{len(df):,}"),
            ("Columns", f"{df.shape[1]:,}"),
            ("Duplicates (rows)", f"{_safe_duplicated_count(df):,}"),
            ("Missing cells (%)", f"{(df.isna().sum().sum() / max(1, df.size) * 100):.2f}"),
        ]
    )

    # Attempt to infer a time span
    dt_candidates: List[str] = []
    if dt_cols_hint:
        dt_candidates.extend([c for c in dt_cols_hint if c in df.columns])
    dt_candidates.extend([c for c in df.columns if "dt" in c.lower() or "date" in c.lower()])
    dt_candidates = list(dict.fromkeys(dt_candidates))  # unique, preserve order

    if dt_candidates:
        pick_dt = st.selectbox("Time column for summary (optional)", options=["(none)"] + dt_candidates, key=f"{title}_dtpick")
        if pick_dt != "(none)":
            series = pd.to_datetime(df[pick_dt], utc=True, errors="coerce").dropna()
            if not series.empty:
                _kpi_row(
                    [
                        ("Min time (UTC)", series.min().strftime("%Y-%m-%d %H:%M")),
                        ("Max time (UTC)", series.max().strftime("%Y-%m-%d %H:%M")),
                        ("Span (days)", f"{(series.max() - series.min()).days:,}"),
                        ("Non-null time rows", f"{len(series):,}"),
                    ]
                )

    with st.expander("Descriptive statistics (numeric)", expanded=False):
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T)

    with st.expander("Missingness (by column)", expanded=False):
        st.dataframe(_missingness_table(df).head(50))

    with st.expander("Correlation matrix (numeric)", expanded=False):
        num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if num.shape[1] < 2:
            st.info("Need at least two numeric columns.")
        else:
            corr = num.corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, aspect="auto"), use_container_width=True)


def download_filtered(df: pd.DataFrame, filename_prefix: str) -> None:
    """
    Offer a download of the filtered frame as CSV.
    """
    if df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"{filename_prefix}_filtered.csv",
        mime="text/csv",
    )


# -----------------------------
# UI: Sidebar - Paths / Loading
# -----------------------------
st.title("NDJSON Analysis Dashboard")

with st.sidebar:
    st.header("Data sources")

    submissions_path = st.text_input("Submissions NDJSON path", value=DEFAULT_SUBMISSIONS_PATH)
    labels_path = st.text_input("Labels NDJSON path", value=DEFAULT_LABELS_PATH)
    thread_path = st.text_input("Thread metrics NDJSON path", value=DEFAULT_THREAD_PATH)

    st.caption("Tip: You can also point these to other NDJSON files on your system.")

    st.divider()
    st.header("Performance")
    max_rows_labels = st.number_input("Max rows (labels)", min_value=0, value=0, step=500, help="0 means: load all")
    max_rows_thread = st.number_input("Max rows (thread metrics)", min_value=0, value=0, step=500, help="0 means: load all")

    load_btn = st.button("Load / Refresh data", type="primary")


def _path_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


# Load data (only when user clicks, to avoid rerun thrash)
if "loaded" not in st.session_state:
    st.session_state["loaded"] = False

if load_btn or not st.session_state["loaded"]:
    errors = []
    if not _path_exists(submissions_path):
        errors.append(f"Submissions path not found: {submissions_path}")
    if not _path_exists(labels_path):
        errors.append(f"Labels path not found: {labels_path}")
    if not _path_exists(thread_path):
        errors.append(f"Thread metrics path not found: {thread_path}")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    with st.spinner("Loading datasets..."):
        df_sub = load_submissions(submissions_path)
        df_lab = load_labels(labels_path, max_rows=None if max_rows_labels == 0 else int(max_rows_labels))
        df_thr = load_thread_metrics(thread_path, max_rows=None if max_rows_thread == 0 else int(max_rows_thread))

    st.session_state["df_sub"] = df_sub
    st.session_state["df_lab"] = df_lab
    st.session_state["df_thr"] = df_thr
    st.session_state["loaded"] = True


df_sub: pd.DataFrame = st.session_state["df_sub"]
df_lab: pd.DataFrame = st.session_state["df_lab"]
df_thr: pd.DataFrame = st.session_state["df_thr"]


# -----------------------------
# Tabs: Separate dataset views
# -----------------------------
tab_sub, tab_lab, tab_thr = st.tabs(["1) Submissions", "2) Labels", "3) Thread metrics"])


# ====== 1) Submissions ======
with tab_sub:
    st.markdown("### Submissions dataset")
    st.caption("Filters are applied on the left sidebar (shared pattern).")

    numeric_cols = _infer_numeric_cols(df_sub)

    df_sub_f = _apply_common_filters(
        df_sub,
        subreddit_col=None,          # sampled.ndjson in your sample had no subreddit column
        topic_col="best_topic",      # available in your sample
        dt_col="created_dt" if "created_dt" in df_sub.columns else None,
        numeric_cols=numeric_cols,
        text_search_cols=[c for c in ["title", "selftext", "author", "best_topic"] if c in df_sub.columns],
        sidebar_prefix="[Submissions]",
    )

    quantitative_overview(df_sub_f, title="Overview (filtered) - Submissions", dt_cols_hint=["created_dt"])

    with st.expander("Preview (filtered)", expanded=True):
        st.dataframe(df_sub_f.head(200))

    chart_builder(df_sub_f, title="Charts (Submissions)")

    st.divider()
    download_filtered(df_sub_f, filename_prefix="submissions")


# ====== 2) Labels ======
with tab_lab:
    st.markdown("### Labels dataset")
    st.caption("This dataset typically contains repeated comment_id rows for multiple tasks (e.g., civility, stance_intensity).")

    # Labels-specific: task filter
    selected_tasks: List[str] = []
    if "task" in df_lab.columns:
        with st.sidebar.expander("[Labels] Task filter", expanded=True):
            tasks = sorted(df_lab["task"].dropna().unique().tolist())
            selected_tasks = st.multiselect("Task", options=tasks, default=[], key="labels_task_filter")

    df_lab_work = df_lab if not selected_tasks else df_lab[df_lab["task"].isin(selected_tasks)]

    numeric_cols = _infer_numeric_cols(df_lab_work)

    df_lab_f = _apply_common_filters(
        df_lab_work,
        subreddit_col="meta.subreddit" if "meta.subreddit" in df_lab_work.columns else None,
        topic_col="meta.best_topic_index" if "meta.best_topic_index" in df_lab_work.columns else None,
        dt_col="meta_dt.created_utc" if "meta_dt.created_utc" in df_lab_work.columns else None,
        numeric_cols=numeric_cols,
        text_search_cols=[c for c in ["body", "meta.subreddit", "meta.submission_title", "meta.user"] if c in df_lab_work.columns],
        sidebar_prefix="[Labels]",
    )

    quantitative_overview(df_lab_f, title="Overview (filtered) - Labels", dt_cols_hint=["meta_dt.created_utc", "meta_dt.submission_created_utc"])

    with st.expander("Label distributions", expanded=True):
        cols = st.columns(2)
        if "task" in df_lab_f.columns and not df_lab_f.empty:
            top_task = _safe_topn(df_lab_f, "task", n=30)
            cols[0].plotly_chart(px.bar(top_task, x="task", y="count"), use_container_width=True)
        else:
            cols[0].info("No task column or no data after filtering.")
        if "result_value" in df_lab_f.columns and not df_lab_f.empty:
            cols[1].plotly_chart(px.histogram(df_lab_f, x="result_value", nbins=20), use_container_width=True)
        else:
            cols[1].info("No numeric label value detected.")

    with st.expander("Preview (filtered)", expanded=False):
        st.dataframe(df_lab_f.head(200))

    chart_builder(df_lab_f, title="Charts (Labels)")

    st.divider()
    download_filtered(df_lab_f, filename_prefix="labels")


# ====== 3) Thread metrics ======
with tab_thr:
    st.markdown("### Thread metrics dataset")
    st.caption("Contains thread-level metrics per comment/task (e.g., argument_novelty, semantic_entropy).")

    # Thread-specific: metric task filter
    selected_tasks: List[str] = []
    if "task" in df_thr.columns:
        with st.sidebar.expander("[Thread metrics] Task filter", expanded=True):
            tasks = sorted(df_thr["task"].dropna().unique().tolist())
            selected_tasks = st.multiselect("Metric task", options=tasks, default=[], key="thread_task_filter")

    df_thr_work = df_thr if not selected_tasks else df_thr[df_thr["task"].isin(selected_tasks)]

    numeric_cols = _infer_numeric_cols(df_thr_work)

    df_thr_f = _apply_common_filters(
        df_thr_work,
        subreddit_col="meta.subreddit" if "meta.subreddit" in df_thr_work.columns else None,
        topic_col="meta.best_topic_index" if "meta.best_topic_index" in df_thr_work.columns else None,
        dt_col="meta_dt.created_utc" if "meta_dt.created_utc" in df_thr_work.columns else None,
        numeric_cols=numeric_cols,
        text_search_cols=[c for c in ["body", "meta.subreddit", "meta.submission_title", "meta.user", "source_task"] if c in df_thr_work.columns],
        sidebar_prefix="[Thread metrics]",
    )

    quantitative_overview(df_thr_f, title="Overview (filtered) - Thread metrics", dt_cols_hint=["meta_dt.created_utc", "meta_dt.submission_created_utc"])

    with st.expander("Metric distributions", expanded=True):
        cols = st.columns(2)
        if "task" in df_thr_f.columns and not df_thr_f.empty:
            top_task = _safe_topn(df_thr_f, "task", n=30)
            cols[0].plotly_chart(px.bar(top_task, x="task", y="count"), use_container_width=True)
        else:
            cols[0].info("No task column or no data after filtering.")
        if "metric_score" in df_thr_f.columns and not df_thr_f.empty:
            cols[1].plotly_chart(px.histogram(df_thr_f, x="metric_score", nbins=30), use_container_width=True)
        else:
            cols[1].info("No metric_score detected.")

    with st.expander("Preview (filtered)", expanded=False):
        st.dataframe(df_thr_f.head(200))

    chart_builder(df_thr_f, title="Charts (Thread metrics)")

    st.divider()
    download_filtered(df_thr_f, filename_prefix="thread_metrics")

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# App configuration
# =============================================================================
st.set_page_config(page_title="NDJSON Dashboard (Simple KPIs)", layout="wide")
st.title("NDJSON Dashboard (Simple KPIs)")

BASE_DIR = Path(__file__).resolve().parent.parent

#DEFAULT_SUBMISSIONS_PATH = str(BASE_DIR / "sampled.ndjson")
DEFAULT_SUBMISSIONS_PATH = BASE_DIR / "sample_data" / "sampled.ndjson"
DEFAULT_LABELS_PATH = BASE_DIR / "pipeline" / "labels_3000.ndjson"
DEFAULT_THREAD_PATH = BASE_DIR / "pipeline" / "thread_metrics.ndjson"


# =============================================================================
# Utility helpers
# =============================================================================
def path_exists(p: str) -> bool:
    """Return True if a path exists (with safe expansion/resolution)."""
    try:
        return Path(p).expanduser().resolve().exists()
    except Exception:
        return False


def stringify_unhashables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list/dict/set/tuple values in object columns into stable JSON strings.
    This prevents pandas errors in operations such as nunique(), duplicated(), groupby(), etc.
    """
    def make_hashable(x: Any) -> Any:
        if isinstance(x, (list, dict, set, tuple)):
            try:
                return json.dumps(x, sort_keys=True, ensure_ascii=False)
            except TypeError:
                return str(x)
        return x

    if df.empty:
        return df

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        df[c] = df[c].map(make_hashable)
    return df


def normalize_subreddit(x: Any) -> Optional[str]:
    """
    Normalize subreddit strings across datasets so merges are stable.
    Handles variations like:
      - 'AskReddit' vs 'askreddit'
      - 'r/AskReddit' vs '/r/AskReddit'
      - surrounding whitespace
    Returns a lowercase normalized name or None.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    s = s.replace("\\", "/").strip()
    s = re.sub(r"^/r/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^r/", "", s, flags=re.IGNORECASE)
    s = s.strip().lower()
    return s or None


def maybe_epoch_to_datetime(s: pd.Series) -> pd.Series:
    """
    Convert epoch seconds or epoch milliseconds to UTC datetime.
    Heuristic: median > 1e12 implies milliseconds.
    """
    if s is None or getattr(s, "empty", False):
        return s
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.notna().sum() == 0:
        return s
    unit = "ms" if float(s2.dropna().median()) > 1e12 else "s"
    return pd.to_datetime(s2, unit=unit, utc=True, errors="coerce")


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    """Convert a series to numeric (coerce errors)."""
    return pd.to_numeric(s, errors="coerce")


def infer_abstain_mask(df: pd.DataFrame) -> pd.Series:
    """
    Infer an 'abstain' mask robustly across different label schemas.

    Rules (OR-combined):
      1) Any column name containing 'abstain' that is boolean-like and truthy.
      2) Any string value field (result.label / result.score) equals 'abstain' (case-insensitive).
    """
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    mask = pd.Series(False, index=df.index)

    # (1) Boolean-like abstain columns
    abstain_cols = [c for c in df.columns if re.search(r"abstain", str(c), flags=re.IGNORECASE)]
    for c in abstain_cols:
        try:
            s = df[c]
            # treat {True, False} / {1,0} / "true"/"false" as boolean-like
            if s.dtype == bool:
                mask = mask | s.fillna(False)
            else:
                s_num = pd.to_numeric(s, errors="coerce")
                if s_num.notna().mean() > 0.2:
                    mask = mask | (s_num.fillna(0) != 0)
                else:
                    s_str = s.astype("string")
                    mask = mask | (s_str.str.strip().str.lower().isin({"abstain", "true", "yes"}))
        except Exception:
            continue

    # (2) Explicit label fields equal 'abstain'
    for c in ["result.label", "result.score", "result_label", "result_score", "label", "score"]:
        if c in df.columns:
            try:
                s_str = df[c].astype("string").str.strip().str.lower()
                mask = mask | s_str.eq("abstain")
            except Exception:
                continue

    return mask.fillna(False)


def add_label_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - is_abstain: boolean
      - label_category: categorical label including 'abstain'

    This ensures abstain is never silently dropped in downstream aggregations.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["is_abstain"] = infer_abstain_mask(out)

    # Pick a best-effort raw label column for categorization
    raw_label_col = first_existing_col(out, ["result.label", "result.score", "label", "score"])
    if raw_label_col and raw_label_col in out.columns:
        raw = out[raw_label_col].astype("string").str.strip()
        raw_norm = raw.str.lower()
        # For non-abstain rows, keep whatever label is present; for abstain rows force 'abstain'
        label_cat = raw_norm.where(~out["is_abstain"], other="abstain")
        # Keep missing explicit (so users can see it rather than silently dropping)
        label_cat = label_cat.fillna("missing")
    else:
        # If no raw label column exists, we still keep abstain / missing
        label_cat = pd.Series("missing", index=out.index)
        label_cat = label_cat.where(~out["is_abstain"], other="abstain")

    out["label_category"] = label_cat.astype("string")
    return out


# =============================================================================
# Loading NDJSON
# =============================================================================
def read_ndjson_objects(path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read NDJSON file into a list of JSON objects."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_submissions(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load submissions. Typically already tabular NDJSON, so pd.read_json is fine.
    Falls back to manual read if max_rows is set.
    """
    p = str(Path(path).expanduser().resolve())
    if max_rows is None:
        df = pd.read_json(p, lines=True)
    else:
        objs = read_ndjson_objects(p, max_rows=max_rows)
        df = pd.DataFrame(objs)

    df = stringify_unhashables(df)

    # Optional datetime parsing for convenience
    c_utc = first_existing_col(df, ["created_utc", "createdUTC", "created"])
    if c_utc:
        df["created_dt"] = maybe_epoch_to_datetime(df[c_utc])

    return df


@st.cache_data(show_spinner=False)
def load_nested_as_flat(path: str, max_rows: Optional[int] = None, sep: str = ".") -> pd.DataFrame:
    """
    Load nested NDJSON (labels / thread metrics) and flatten with json_normalize.
    Ensures abstain is materialized as a category (label_category + is_abstain).
    """
    p = str(Path(path).expanduser().resolve())
    objs = read_ndjson_objects(p, max_rows=max_rows)
    if not objs:
        return pd.DataFrame()

    df = pd.json_normalize(objs, sep=sep)
    df = stringify_unhashables(df)

    # Optional datetime parsing for common metadata keys
    for col in ["meta.created_utc", "meta.submission_created_utc", "created_utc", "submission_created_utc"]:
        if col in df.columns:
            df[col.replace(".", "_") + "_dt"] = maybe_epoch_to_datetime(df[col])

    # Standardize numeric "value" and "confidence" fields where possible
    if "result.score" in df.columns:
        df["result_score_num"] = safe_numeric(df["result.score"])
    if "result.label" in df.columns:
        df["result_label_num"] = safe_numeric(df["result.label"])
    if "result.confidence" in df.columns:
        df["result_conf_num"] = safe_numeric(df["result.confidence"])

    # Make abstain explicit for all downstream aggregations/plots
    df = add_label_category_columns(df)

    return df


# =============================================================================
# KPI computations
# =============================================================================
def compute_subreddit_kpis(
        df_sub: pd.DataFrame,
        df_lab: pd.DataFrame,
        subreddit_col_sub: str,
        subreddit_col_lab: str,
        submission_id_col_sub: Optional[str],
        comment_id_col_lab: Optional[str],
        num_comments_col_sub: Optional[str],
        *,
        use_distinct_comments: bool = True,
) -> pd.DataFrame:
    """
    Robust per-subreddit KPIs.

    Key design choices:
      - Merges use a normalized key (lowercase, strips 'r/' prefixes).
      - Comment counts are deduplicated by comment_id when available, otherwise row-count.
      - Two engagement metrics:
          * engagement_from_submissions: sum(num_comments)/submissions_count
          * engagement_from_labels: distinct_comments/submissions_count (proxy)

    Abstain handling:
      - 'abstain' rows are explicitly counted (abstain_rows, abstain_share_of_label_rows).
    """
    if df_sub.empty:
        return pd.DataFrame()

    sub = df_sub.copy()
    lab = df_lab.copy() if not df_lab.empty else pd.DataFrame()

    # --- Normalize subreddit keys (critical for stable merges) ---
    sub["subreddit_key"] = sub[subreddit_col_sub].map(normalize_subreddit)
    sub = sub.dropna(subset=["subreddit_key"])

    if not lab.empty:
        lab["subreddit_key"] = lab[subreddit_col_lab].map(normalize_subreddit)
        lab = lab.dropna(subset=["subreddit_key"])

    # --- Submissions count per subreddit ---
    g_sub = (
        sub.groupby("subreddit_key", dropna=True)
        .size()
        .rename("submissions_count")
        .reset_index()
    )

    # --- Distinct submission ids (metadata) ---
    if submission_id_col_sub and submission_id_col_sub in sub.columns:
        g_ids = (
            sub.groupby("subreddit_key", dropna=True)[submission_id_col_sub]
            .nunique(dropna=True)
            .rename("distinct_submission_ids")
            .reset_index()
        )
    else:
        g_ids = pd.DataFrame(columns=["subreddit_key", "distinct_submission_ids"])

    # --- num_comments stats from submissions (ground truth-ish for submission volume) ---
    g_nc = pd.DataFrame(columns=["subreddit_key", "num_comments_sum", "num_comments_mean", "num_comments_median"])
    if num_comments_col_sub and num_comments_col_sub in sub.columns:
        tmp = sub[["subreddit_key", num_comments_col_sub]].copy()
        tmp[num_comments_col_sub] = pd.to_numeric(tmp[num_comments_col_sub], errors="coerce")
        g_nc = (
            tmp.groupby("subreddit_key", dropna=True)[num_comments_col_sub]
            .agg(num_comments_sum="sum", num_comments_mean="mean", num_comments_median="median")
            .reset_index()
        )

    # --- Comments proxy + abstain stats from labels ---
    g_lab = pd.DataFrame(columns=["subreddit_key", "label_rows", "labeled_comment_ids", "abstain_rows", "abstain_share_of_label_rows"])

    if not lab.empty:
        # label rows per subreddit
        g_rows = (
            lab.groupby("subreddit_key", dropna=True)
            .size()
            .rename("label_rows")
            .reset_index()
        )

        # abstain rows per subreddit (requires is_abstain; load_nested_as_flat ensures it exists)
        if "is_abstain" in lab.columns:
            g_ab = (
                lab.groupby("subreddit_key", dropna=True)["is_abstain"]
                .sum()
                .rename("abstain_rows")
                .reset_index()
            )
        else:
            g_ab = pd.DataFrame(columns=["subreddit_key", "abstain_rows"])

        # distinct labeled comments (avoid double counting across tasks/replicates)
        labeled_comment_ids = None
        comment_id_ok = (
                use_distinct_comments
                and comment_id_col_lab
                and comment_id_col_lab in lab.columns
                and lab[comment_id_col_lab].notna().mean() > 0.05
        )

        if comment_id_ok:
            cid = lab[comment_id_col_lab]
            cid = cid.where(cid.notna(), np.nan).astype("string")
            g_cids = (
                lab.assign(_cid=cid)
                .dropna(subset=["_cid"])
                .groupby("subreddit_key", dropna=True)["_cid"]
                .nunique()
                .rename("labeled_comment_ids")
                .reset_index()
            )
        else:
            g_cids = pd.DataFrame(columns=["subreddit_key", "labeled_comment_ids"])

        g_lab = g_rows.merge(g_cids, how="left", on="subreddit_key").merge(g_ab, how="left", on="subreddit_key")
        g_lab["abstain_rows"] = g_lab["abstain_rows"].fillna(0).astype(int)
        g_lab["abstain_share_of_label_rows"] = g_lab["abstain_rows"] / g_lab["label_rows"].replace({0: np.nan})

    # --- Merge everything ---
    out = g_sub.merge(g_ids, how="left", on="subreddit_key").merge(g_nc, how="left", on="subreddit_key").merge(g_lab, how="left", on="subreddit_key")

    out["label_rows"] = out.get("label_rows", 0).fillna(0).astype(int)
    out["labeled_comment_ids"] = out.get("labeled_comment_ids", np.nan)

    out["comments_count"] = out["labeled_comment_ids"].fillna(0).astype(int)

    out["rows_per_labeled_comment"] = out["label_rows"] / out["comments_count"].replace({0: np.nan})

    # Engagement from submissions: uses num_comments sum (preferred when present)
    if "num_comments_sum" in out.columns:
        out["engagement_from_submissions"] = out["num_comments_sum"] / out["submissions_count"].replace({0: np.nan})
    else:
        out["engagement_from_submissions"] = np.nan

    # Engagement from labels: proxy
    out["engagement_from_labels"] = out["comments_count"] / out["submissions_count"].replace({0: np.nan})

    # Pretty display name
    out = out.rename(columns={"subreddit_key": "subreddit"})
    out = out.sort_values(["submissions_count", "comments_count"], ascending=False)

    return out


def compute_topics_by_subreddit(
        df_sub: pd.DataFrame,
        subreddit_col: str,
        topic_col: str,
        top_n_topics: int = 20,
) -> pd.DataFrame:
    """Per subreddit: counts of topics (e.g., best_topic)."""
    tmp = df_sub[[subreddit_col, topic_col]].dropna()
    out = (
        tmp.groupby([subreddit_col, topic_col], dropna=True)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={subreddit_col: "subreddit", topic_col: "topic"})
    )

    top_topics = out.groupby("topic", dropna=True)["count"].sum().sort_values(ascending=False).head(top_n_topics).index
    out = out[out["topic"].isin(top_topics)]
    return out.sort_values(["subreddit", "count"], ascending=[True, False])


def compute_topics_overall(df_sub: pd.DataFrame, topic_col: str) -> pd.DataFrame:
    """Overall: topic counts (e.g., best_topic)."""
    tmp = df_sub[[topic_col]].dropna()
    return tmp[topic_col].value_counts().rename_axis("topic").reset_index(name="count")


def compute_task_distributions(
        df_any: pd.DataFrame,
        task_col: str,
        value_cols: List[str],
        conf_col: Optional[str],
        label_cat_col: str = "label_category",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For a labels-like dataset:
      A) Label-category distribution per task (ALWAYS includes 'abstain' if present)
      B) Numeric value distribution per task (only if a usable numeric column exists; abstain is excluded here)
      C) Confidence summary per task (if available)

    Returns: (label_dist, value_dist, conf_summary)
    """
    if df_any is None or df_any.empty or task_col not in df_any.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    tmp = df_any.copy()

    # ---- A) Label-category distribution (incl abstain) ----
    if label_cat_col not in tmp.columns:
        tmp = add_label_category_columns(tmp)

    tmp[label_cat_col] = tmp[label_cat_col].astype("string").fillna("missing")

    label_dist = (
        tmp.groupby([task_col, label_cat_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={task_col: "task", label_cat_col: "label"})
        .sort_values(["task", "count"], ascending=[True, False])
    )

    # ---- B) Numeric value distribution (if present) ----
    value_col = first_existing_col(tmp, value_cols)
    value_dist = pd.DataFrame()
    if value_col is not None and value_col in tmp.columns:
        tmp[value_col] = safe_numeric(tmp[value_col])

        # Exclude abstain for numeric distributions (numeric y cannot represent "abstain")
        if "is_abstain" in tmp.columns:
            tmp_num = tmp[~tmp["is_abstain"]].copy()
        else:
            tmp_num = tmp.copy()

        nunique = tmp_num[value_col].nunique(dropna=True)
        if nunique > 0:
            if nunique <= 25:
                value_dist = (
                    tmp_num.dropna(subset=[value_col])
                    .groupby([task_col, value_col], dropna=True)
                    .size()
                    .rename("count")
                    .reset_index()
                    .rename(columns={task_col: "task", value_col: "result"})
                )
            else:
                bins = 20
                tmp2 = tmp_num.dropna(subset=[value_col]).copy()
                tmp2["result_bin"] = pd.cut(tmp2[value_col], bins=bins)
                value_dist = (
                    tmp2.groupby([task_col, "result_bin"], dropna=True)
                    .size()
                    .rename("count")
                    .reset_index()
                    .rename(columns={task_col: "task", "result_bin": "result"})
                )

    # ---- C) Confidence summary ----
    conf_summary = pd.DataFrame()
    if conf_col and conf_col in tmp.columns:
        tmp[conf_col] = safe_numeric(tmp[conf_col])
        conf_summary = (
            tmp.groupby(task_col, dropna=True)[conf_col]
            .agg(["count", "mean", "median"])
            .reset_index()
            .rename(columns={task_col: "task", "count": "n"})
        )
        q = (
            tmp.groupby(task_col, dropna=True)[conf_col]
            .quantile([0.1, 0.9])
            .unstack()
            .reset_index()
            .rename(columns={0.1: "p10", 0.9: "p90", task_col: "task"})
        )
        conf_summary = conf_summary.merge(q, on="task", how="left").sort_values("n", ascending=False)

    return label_dist, value_dist, conf_summary


# =============================================================================
# Sidebar: paths + load controls
# =============================================================================
st.sidebar.header("Data files")
sub_path = st.sidebar.text_input("Submissions NDJSON", value=DEFAULT_SUBMISSIONS_PATH)
lab_path = st.sidebar.text_input("Labels NDJSON", value=DEFAULT_LABELS_PATH)
thr_path = st.sidebar.text_input("Thread metrics NDJSON", value=DEFAULT_THREAD_PATH)

st.sidebar.divider()
st.sidebar.header("Loading options")
max_rows = st.sidebar.number_input("Max rows per file (0 = all)", min_value=0, value=0, step=1000)
reload_btn = st.sidebar.button("Load / Reload", type="primary")

max_rows_opt: Optional[int] = None if int(max_rows) == 0 else int(max_rows)

with st.expander("Status", expanded=True):
    st.write("app.py directory:", str(BASE_DIR))
    st.write("cwd:", str(Path.cwd()))
    st.write("Submissions exists:", path_exists(sub_path), sub_path)
    st.write("Labels exists:", path_exists(lab_path), lab_path)
    st.write("Thread metrics exists:", path_exists(thr_path), thr_path)


# =============================================================================
# Load datasets (auto on first run + on reload)
# =============================================================================
if "loaded" not in st.session_state:
    st.session_state["loaded"] = False

should_load = reload_btn or (not st.session_state["loaded"])

if should_load:
    errors = []
    if not path_exists(sub_path):
        errors.append(f"Submissions file not found: {sub_path}")
    if not path_exists(lab_path):
        errors.append(f"Labels file not found: {lab_path}")
    if not path_exists(thr_path):
        errors.append(f"Thread metrics file not found: {thr_path}")

    if errors:
        for e in errors:
            st.error(e)
        st.session_state["df_sub"] = pd.DataFrame()
        st.session_state["df_lab"] = pd.DataFrame()
        st.session_state["df_thr"] = pd.DataFrame()
        st.session_state["loaded"] = True
    else:
        with st.spinner("Loading NDJSON files..."):
            df_sub = load_submissions(sub_path, max_rows=max_rows_opt)
            df_lab = load_nested_as_flat(lab_path, max_rows=max_rows_opt)
            df_thr = load_nested_as_flat(thr_path, max_rows=max_rows_opt)

        st.session_state["df_sub"] = df_sub
        st.session_state["df_lab"] = df_lab
        st.session_state["df_thr"] = df_thr
        st.session_state["loaded"] = True

df_sub: pd.DataFrame = st.session_state.get("df_sub", pd.DataFrame())
df_lab: pd.DataFrame = st.session_state.get("df_lab", pd.DataFrame())
df_thr: pd.DataFrame = st.session_state.get("df_thr", pd.DataFrame())

c1, c2, c3 = st.columns(3)
c1.metric("Submissions rows", f"{len(df_sub):,}")
c2.metric("Labels rows", f"{len(df_lab):,}")
c3.metric("Thread metrics rows", f"{len(df_thr):,}")

st.divider()


# =============================================================================
# Resolve key columns using pragmatic defaults
# =============================================================================
subreddit_sub = first_existing_col(df_sub, ["subreddit", "meta.subreddit", "subreddit_name_prefixed"])
subreddit_lab = first_existing_col(df_lab, ["meta.subreddit", "subreddit", "meta.subverse", "meta.subverse_name"])
subreddit_thr = first_existing_col(df_thr, ["meta.subreddit", "subreddit", "meta.subverse", "meta.subverse_name"])

topic_sub = first_existing_col(df_sub, ["best_topic", "meta.best_topic", "topic", "meta.topic"])

submission_id_sub = first_existing_col(df_sub, ["id", "submission_id", "name", "link_id"])
comment_id_lab = first_existing_col(df_lab, ["comment_id", "id", "name", "meta.comment_id"])
comment_id_thr = first_existing_col(df_thr, ["comment_id", "id", "name", "meta.comment_id"])

num_comments_sub = first_existing_col(df_sub, ["num_comments", "meta.num_comments", "comment_count"])

task_lab = first_existing_col(df_lab, ["task"])
task_thr = first_existing_col(df_thr, ["task", "source_task"])

conf_lab = first_existing_col(df_lab, ["result_conf_num", "result.confidence"])
conf_thr = first_existing_col(df_thr, ["result_conf_num", "result.confidence"])

value_candidates = ["result_score_num", "result_label_num", "result.score", "result.label"]


# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Subreddit KPIs", "Topics", "Tasks (Labels)", "Tasks (Thread metrics)", "Label Boxplots"]
)

# -----------------------------------------------------------------------------
# Tab 1: Subreddit KPIs
# -----------------------------------------------------------------------------
with tab1:
    st.header("KPIs per subreddit")

    if df_sub.empty or df_lab.empty:
        st.warning(
            "Subreddit KPIs require both submissions and labels to be loaded. "
            "Check file paths and ensure the datasets contain data."
        )
    elif not subreddit_sub or not subreddit_lab:
        st.warning(
            "Could not detect a subreddit column in submissions and/or labels. "
            "Expected columns like 'subreddit' or 'meta.subreddit'."
        )
    else:
        use_distinct = st.checkbox(
            "Count distinct comments (recommended)",
            value=True,
            help="Uses comment_id nunique when available; avoids double counting across tasks.",
            key="kpi_distinct_comments",
        )

        kpi = compute_subreddit_kpis(
            df_sub=df_sub,
            df_lab=df_lab,
            subreddit_col_sub=subreddit_sub,
            subreddit_col_lab=subreddit_lab,
            submission_id_col_sub=submission_id_sub,
            comment_id_col_lab=comment_id_lab,
            num_comments_col_sub=num_comments_sub,
            use_distinct_comments=use_distinct,
        )

        top_n = st.slider("Show top N subreddits (by submissions)", 5, 200, 30, key="kpi_top_n")
        kpi_show = kpi.head(top_n)

        st.dataframe(kpi_show, use_container_width=True)

        colA, colB = st.columns(2)
        colA.subheader("Submissions per subreddit")
        colA.plotly_chart(px.bar(kpi_show, x="subreddit", y="submissions_count"), use_container_width=True)

        colB.subheader("Comments per subreddit (proxy from labels)")
        colB.plotly_chart(px.bar(kpi_show, x="subreddit", y="comments_count"), use_container_width=True)

        st.subheader("Engagement")
        colC, colD = st.columns(2)
        colC.caption("Based on submissions (sum(num_comments)/#submissions)")
        colC.plotly_chart(px.bar(kpi_show, x="subreddit", y="engagement_from_submissions"), use_container_width=True)
        colD.caption("Based on labels (distinct labeled comments/#submissions)")
        colD.plotly_chart(px.bar(kpi_show, x="subreddit", y="engagement_from_labels"), use_container_width=True)

        st.subheader("Abstain coverage (labels)")
        if "abstain_rows" in kpi_show.columns:
            colE, colF = st.columns(2)
            colE.plotly_chart(px.bar(kpi_show, x="subreddit", y="abstain_rows"), use_container_width=True)
            colF.plotly_chart(px.bar(kpi_show, x="subreddit", y="abstain_share_of_label_rows"), use_container_width=True)
        else:
            st.info("No abstain fields detected in labels (or labels are empty).")

        with st.expander("Diagnostic: labels coverage by subreddit", expanded=False):
            tmp = df_lab.copy()
            tmp["subreddit_norm"] = tmp[subreddit_lab].map(normalize_subreddit)
            cov = (
                tmp["subreddit_norm"]
                .value_counts(dropna=True)
                .head(50)
                .rename_axis("subreddit")
                .reset_index(name="label_rows")
            )
            st.dataframe(cov, use_container_width=True)
            st.plotly_chart(px.bar(cov.head(20), x="subreddit", y="label_rows"), use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 2: Topics
# -----------------------------------------------------------------------------
with tab2:
    st.header("Topic distributions")

    if df_sub.empty:
        st.warning("Topics require submissions to be loaded.")
    elif not subreddit_sub or not topic_sub:
        st.warning(
            "Could not detect required columns for topics. "
            "Expected a subreddit column (e.g. 'subreddit') and a topic column (e.g. 'best_topic')."
        )
    else:
        st.subheader("Overall topics")
        overall = compute_topics_overall(df_sub, topic_col=topic_sub)
        st.dataframe(overall.head(50), use_container_width=True)
        st.plotly_chart(px.bar(overall.head(30), x="topic", y="count"), use_container_width=True)

        st.divider()

        st.subheader("Topics per subreddit (top topics only)")
        top_topics_n = st.slider("Keep top N topics overall (to reduce clutter)", 5, 50, 15)
        topic_by_sub = compute_topics_by_subreddit(
            df_sub=df_sub,
            subreddit_col=subreddit_sub,
            topic_col=topic_sub,
            top_n_topics=top_topics_n,
        )

        sub_list = sorted(topic_by_sub["subreddit"].dropna().unique().tolist())
        selected_subs = st.multiselect(
            "Select subreddits (empty = show top 10 by submissions)",
            options=sub_list,
            default=[],
        )

        if not selected_subs:
            top10 = df_sub[subreddit_sub].value_counts(dropna=True).head(10).index.tolist()
            selected_subs = top10

        view = topic_by_sub[topic_by_sub["subreddit"].isin(selected_subs)]
        st.dataframe(view, use_container_width=True)

        st.plotly_chart(
            px.bar(view, x="topic", y="count", facet_col="subreddit", facet_col_wrap=2),
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Tab 3: Tasks (Labels)
# -----------------------------------------------------------------------------
with tab3:
    st.header("Task KPIs — Labels")

    if df_lab.empty:
        st.warning("Labels dataset is empty.")
    elif not task_lab:
        st.warning("Could not detect a 'task' column in labels.")
    else:
        tasks = sorted(df_lab[task_lab].dropna().unique().tolist())
        selected = st.selectbox("Select task", options=tasks, key="labels_task_select")

        dft = df_lab[df_lab[task_lab] == selected].copy()

        # Ensure abstain columns exist for KPI tiles (compute_task_distributions uses a copy)
        if "is_abstain" not in dft.columns or "label_category" not in dft.columns:
            dft = add_label_category_columns(dft)

        label_dist, value_dist, conf = compute_task_distributions(
            df_any=dft,
            task_col=task_lab,
            value_cols=value_candidates,
            conf_col=conf_lab,
            label_cat_col="label_category",
        )

        st.subheader("Task summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(dft):,}")
        if "is_abstain" in dft.columns:
            c2.metric("Abstain rows", f"{int(dft['is_abstain'].sum()):,}")
            c3.metric("Abstain share", f"{float(dft['is_abstain'].mean()):.3f}")
        else:
            c2.metric("Abstain rows", "N/A")
            c3.metric("Abstain share", "N/A")

        if conf_lab and conf_lab in dft.columns:
            c4.metric("Mean confidence", f"{safe_numeric(dft[conf_lab]).mean():.3f}")
        else:
            c4.metric("Mean confidence", "N/A")

        st.subheader("Label category distribution (includes 'abstain')")
        if label_dist.empty:
            st.info("No label categories available.")
        else:
            st.dataframe(label_dist, use_container_width=True)
            st.plotly_chart(px.bar(label_dist, x="label", y="count"), use_container_width=True)

        st.subheader("Numeric result distribution (abstain excluded by design)")
        if value_dist.empty:
            st.info("No usable numeric result field found for this task.")
        else:
            st.dataframe(value_dist.head(200), use_container_width=True)
            st.plotly_chart(px.bar(value_dist, x="result", y="count"), use_container_width=True)

        st.subheader("Confidence summary (per task)")
        if conf.empty:
            st.info("No confidence field available for labels.")
        else:
            st.dataframe(conf, use_container_width=True)

        if conf_lab and conf_lab in dft.columns:
            st.subheader("Confidence distribution")
            st.plotly_chart(px.histogram(dft, x=conf_lab, nbins=30), use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 4: Tasks (Thread metrics)
# -----------------------------------------------------------------------------
with tab4:
    st.header("Task KPIs — Thread metrics")

    if df_thr.empty:
        st.warning("Thread metrics dataset is empty.")
    elif not task_thr:
        st.warning("Could not detect a task column in thread metrics (expected 'task' or 'source_task').")
    else:
        tasks = sorted(df_thr[task_thr].dropna().unique().tolist())
        selected = st.selectbox("Select metric task", options=tasks, key="thr_task_select")

        dft = df_thr[df_thr[task_thr] == selected].copy()

        # Ensure abstain columns exist for KPI tiles (compute_task_distributions uses a copy)
        if "is_abstain" not in dft.columns or "label_category" not in dft.columns:
            dft = add_label_category_columns(dft)

        label_dist, value_dist, conf = compute_task_distributions(
            df_any=dft,
            task_col=task_thr,
            value_cols=value_candidates,
            conf_col=conf_thr,
            label_cat_col="label_category",
        )

        st.subheader("Task summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(dft):,}")
        if "is_abstain" in dft.columns:
            c2.metric("Abstain rows", f"{int(dft['is_abstain'].sum()):,}")
            c3.metric("Abstain share", f"{float(dft['is_abstain'].mean()):.3f}")
        else:
            c2.metric("Abstain rows", "N/A")
            c3.metric("Abstain share", "N/A")

        if conf_thr and conf_thr in dft.columns:
            c4.metric("Mean confidence", f"{safe_numeric(dft[conf_thr]).mean():.3f}")
        else:
            c4.metric("Mean confidence", "N/A")

        st.subheader("Label category distribution (includes 'abstain')")
        if label_dist.empty:
            st.info("No label categories available.")
        else:
            st.dataframe(label_dist, use_container_width=True)
            st.plotly_chart(px.bar(label_dist, x="label", y="count"), use_container_width=True)

        st.subheader("Numeric metric distribution (abstain excluded by design)")
        if value_dist.empty:
            st.info("No usable numeric metric field found for this task.")
        else:
            st.dataframe(value_dist.head(200), use_container_width=True)
            st.plotly_chart(px.bar(value_dist, x="result", y="count"), use_container_width=True)

        st.subheader("Confidence summary (per task)")
        if conf.empty:
            st.info("No confidence field available for thread metrics.")
        else:
            st.dataframe(conf, use_container_width=True)

        if conf_thr and conf_thr in dft.columns:
            st.subheader("Confidence distribution")
            st.plotly_chart(px.histogram(dft, x=conf_thr, nbins=30), use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 5: Label Boxplots per subreddit (flexible subreddit selection)
# -----------------------------------------------------------------------------
with tab5:
    st.header("Label Boxplots per subreddit")

    source = st.radio("Data source", options=["Labels", "Thread metrics"], horizontal=True, key="box_source")
    df_src = df_lab if source == "Labels" else df_thr
    subreddit_src = subreddit_lab if source == "Labels" else subreddit_thr
    task_src = task_lab if source == "Labels" else task_thr
    conf_src = conf_lab if source == "Labels" else conf_thr

    if df_src.empty:
        st.warning("Selected dataset is empty.")
    elif not subreddit_src:
        st.warning("Could not detect a subreddit column in the selected dataset.")
    else:
        dfp = df_src.copy()
        dfp["subreddit_norm"] = dfp[subreddit_src].map(normalize_subreddit)

        # Ensure abstain/category columns exist (load_nested_as_flat already does this, but keep safe)
        if "label_category" not in dfp.columns or "is_abstain" not in dfp.columns:
            dfp = add_label_category_columns(dfp)

        # Subreddit selector (flexible)
        all_subs = (
            dfp["subreddit_norm"]
            .dropna()
            .value_counts()
        )
        sub_options = all_subs.index.tolist()

        default_subs = sub_options[:10]  # top 10 by label rows
        selected_subs = st.multiselect(
            "Select subreddits to plot",
            options=sub_options,
            default=default_subs,
            key="box_subs",
        )

        if not selected_subs:
            st.info("Select at least one subreddit.")
        else:
            dfp = dfp[dfp["subreddit_norm"].isin(selected_subs)].copy()

            # Optional: task filtering
            if task_src and task_src in dfp.columns:
                task_options = sorted(dfp[task_src].dropna().unique().tolist())
                selected_tasks = st.multiselect(
                    "Filter by task (empty = all)",
                    options=task_options,
                    default=[],
                    key="box_tasks",
                )
                if selected_tasks:
                    dfp = dfp[dfp[task_src].isin(selected_tasks)].copy()
            else:
                st.caption("No task column detected for this dataset; plotting across all rows.")

            # Choose what to boxplot
            y_mode = st.selectbox(
                "Boxplot Y variable",
                options=["Confidence", "Numeric result (if available)"],
                key="box_y_mode",
            )

            if y_mode == "Confidence":
                if not conf_src or conf_src not in dfp.columns:
                    st.warning("No confidence column available in this dataset.")
                else:
                    dfp["_y"] = safe_numeric(dfp[conf_src])
                    st.caption("Boxes show the confidence distribution; colors include 'abstain'.")
                    fig = px.box(
                        dfp.dropna(subset=["_y"]),
                        x="subreddit_norm",
                        y="_y",
                        color="label_category",
                        points="outliers",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                val_col = first_existing_col(dfp, value_candidates)
                if not val_col or val_col not in dfp.columns:
                    st.warning("No usable numeric result column found (checked: result_score_num/result_label_num/result.score/result.label).")
                else:
                    dfp["_y"] = safe_numeric(dfp[val_col])

                    # IMPORTANT: abstain cannot be represented as numeric in y; keep it counted, but exclude from box
                    abstain_rate = float(dfp["is_abstain"].mean()) if "is_abstain" in dfp.columns and len(dfp) else np.nan
                    st.caption(
                        f"Boxes show numeric results; abstain is excluded from the boxplot by design. "
                        f"Current abstain share in the filtered data: {abstain_rate:.3f}"
                    )

                    df_num = dfp[(~dfp["is_abstain"]) & (dfp["_y"].notna())].copy()
                    fig = px.box(
                        df_num,
                        x="subreddit_norm",
                        y="_y",
                        points="outliers",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Always show a compact abstain overview for the selected slice
            st.subheader("Abstain overview (selected slice)")
            if "is_abstain" in dfp.columns:
                ab = (
                    dfp.groupby(["subreddit_norm", "label_category"], dropna=False)
                    .size()
                    .rename("count")
                    .reset_index()
                    .sort_values(["subreddit_norm", "count"], ascending=[True, False])
                )
                st.dataframe(ab, use_container_width=True)
                st.plotly_chart(
                    px.bar(ab, x="subreddit_norm", y="count", color="label_category"),
                    use_container_width=True,
                )
            else:
                st.info("No abstain flag available for this dataset.")

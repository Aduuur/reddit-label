#%%
# app_thread_viewer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


# -----------------------------
# NDJSON -> list[dict]
# -----------------------------
@st.cache_data(show_spinner=False)
def read_ndjson_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                raise TypeError(f"Line {line_no} is not a JSON object (dict). Got: {type(obj)}")
    return records


# -----------------------------
# Extractors for YOUR structure
# -----------------------------
def get_thread_id(r: Dict[str, Any]) -> Optional[str]:
    meta = r.get("meta") or {}
    if isinstance(meta, dict):
        return meta.get("thread_id")
    return None


def get_comment_id(r: Dict[str, Any]) -> Optional[str]:
    return r.get("comment_id_canon") or r.get("comment_id")


def get_parent_id(r: Dict[str, Any]) -> Optional[str]:
    return r.get("parent_id_canon") or r.get("parent_id")


# -----------------------------
# Thread index (thread_id -> unique comment count)
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_thread_sizes(records: List[Dict[str, Any]]) -> Dict[str, int]:
    # count unique comments per thread (since tasks duplicate comment lines)
    per_thread_seen: Dict[str, set[str]] = {}
    for r in records:
        tid = get_thread_id(r)
        cid = get_comment_id(r)
        if not tid or not cid:
            continue
        per_thread_seen.setdefault(tid, set()).add(cid)
    return {tid: len(s) for tid, s in per_thread_seen.items()}


# -----------------------------
# Build directed thread graph
# -----------------------------
def build_thread_graph(records: List[Dict[str, Any]], thread_id: str) -> nx.DiGraph:
    # filter to thread
    thread_recs = [r for r in records if get_thread_id(r) == thread_id]

    # dedupe by comment_id
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for r in thread_recs:
        cid = get_comment_id(r)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        unique.append(r)

    # build graph parent -> child
    G = nx.DiGraph()
    for r in unique:
        cid = get_comment_id(r)
        pid = get_parent_id(r)
        if not cid:
            continue

        G.add_node(cid)
        if pid:
            G.add_node(pid)
            G.add_edge(pid, cid)

    return G


# -----------------------------
# PyVis HTML generator
# -----------------------------
def graph_to_pyvis_html(G: nx.DiGraph, height_px: int = 850) -> str:
    net = Network(height=f"{height_px}px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # ONLY id as label
    for n in G.nodes():
        net.add_node(n, label=n)

    for u, v in G.edges():
        net.add_edge(u, v)

    net.set_options(
        """
        var options = {
          "nodes": {"font": {"size": 14}},
          "edges": {"arrows": {"to": {"enabled": true}}},
          "physics": {"stabilization": {"iterations": 200}}
        }
        """
    )

    return net.generate_html()


# =============================================================================
# Streamlit UI
# =============================================================================
st.set_page_config(page_title="Thread Viewer", layout="wide")
st.title("Thread Viewer (thread_id / parent_id)")

with st.sidebar:
    st.header("Input")
    default_path = "thread_metrics.ndjson"
    ndjson_path = st.text_input("NDJSON path", value=default_path)

    st.header("Performance")
    top_k = st.number_input("Show only top-K largest threads (0 = all)", min_value=0, max_value=20000, value=500, step=50)
    graph_height = st.slider("Graph height (px)", min_value=400, max_value=1400, value=850, step=50)

# Load records
try:
    records = read_ndjson_records(ndjson_path)
except Exception as e:
    st.error(f"Could not read NDJSON: {e}")
    st.stop()

# Build thread list (optionally by size)
sizes = compute_thread_sizes(records)
if not sizes:
    st.error("No threads found. Expected meta.thread_id and comment_id(_canon).")
    st.stop()

# Sort threads by size desc
sorted_threads = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
if top_k and top_k > 0:
    sorted_threads = sorted_threads[: int(top_k)]

thread_options = [tid for tid, _ in sorted_threads]

# Pick thread
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    selected_thread = st.selectbox("Select thread_id", options=thread_options, index=0)
with col2:
    st.metric("Unique comments", sizes.get(selected_thread, 0))
with col3:
    # quick info
    st.metric("Total threads (in file)", len(sizes))

# Build & render
G = build_thread_graph(records, selected_thread)
st.caption(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (labels are IDs only).")

html = graph_to_pyvis_html(G, height_px=graph_height)
components.html(html, height=graph_height + 40, scrolling=True)

with st.expander("Debug: show sample record for this thread"):
    # find first record matching thread for inspection
    sample = next((r for r in records if get_thread_id(r) == selected_thread), None)
    st.json(sample if sample is not None else {})

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration.py

starten über

python calibration.py \
  --human_excel "/Users/arthur/DataspellProjects/reddit-l/data/sampled_threads.xlsx" \
  --llm_ndjson "/Users/arthur/DataspellProjects/reddit-l/labels_from_excel.ndjson" \
  --output_dir "/Users/arthur/DataspellProjects/reddit-l/results_agreement_arthur" \
  --active_annotators Arthur

Zweck
-----
Vergleicht Human- und LLM-Annotationen für Reddit-Kommentare.

Input:
1) Human-Annotationen in Excel
   Erwartete Struktur:
   - comment_id
   - best_topic_index
   - body
   - predecessor
   - Arthur / Veronika als Annotator-Gruppen
   - darunter Task-Spalten wie:
       stance intensity [1,6]
       epistemic modality [0,1]
       justificartion density
       responsivness [0,1] / responsivness [-1,1]
       agreement [-1,1] / agreement 0,1]
       civility [1,6]
       sarcasm[0,1]

2) LLM-Annotationen in NDJSON
   - eine Zeile pro (comment_id, task)
   - result.score oder result.label
   - result.confidence optional
   - task z.B. stance_intensity, civility, epistemic_modality, ...

Features
--------
- frei wählbare aktive Human-Annotator:innen
- Human-Human nur wenn 2 Humans aktiv sind
- Human-LLM für alle aktiven Humans
- automatische Umwandlung des NDJSON long -> wide
- ordinale, binäre und kontinuierliche Tasks werden unterschiedlich ausgewertet
- robuste Spaltennormalisierung für leicht unterschiedliche Excel-Benennungen
- Coverage- und Duplikat-Checks
- Warnungen bei fehlenden Spalten

Beispiel:
python agreement_analysis_reddit.py \
    --human_excel /path/to/human_annotations.xlsx \
    --llm_ndjson /path/to/llm_annotations.ndjson \
    --output_dir /path/to/results \
    --active_annotators Arthur Veronika
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)


# =============================================================================
# KONFIGURATION
# =============================================================================

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "stance_intensity": {
        "type": "ordinal",
        "human_patterns": [
            "stance intensity [1,6]",
            "stance intensity",
        ],
        "llm_task_names": [
            "stance_intensity",
            "stance intensity",
            "stance-intensity",
        ],
        "label_min": 1,
        "label_max": 6,
        "adjacent_tolerance": 1,
    },
    "epistemic_modality": {
        "type": "continuous",
        "human_patterns": [
            "epistemic modality [0,1]",
            "epistemic modality",
        ],
        "llm_task_names": [
            "epistemic_modality",
            "epistemic modality",
            "epistemic-modality",
        ],
        "label_min": 0.0,
        "label_max": 1.0,
    },
    "justification_density": {
        "type": "continuous",
        "human_patterns": [
            "justificartion density [0,1]",
            "justification density [0,1]",
            "justificartion density",
            "justification density",
        ],
        "llm_task_names": [
            "justification_density",
            "justification density",
            "justification-density",
        ],
        "label_min": 0.0,
        "label_max": 1.0,
    },
    "responsiveness": {
        "type": "continuous",
        "human_patterns": [
            "responsivness [0,1]",
            "responsiveness [0,1]",
            "responsivness [-1,1]",
            "responsiveness [-1,1]",
            "responsivness",
            "responsiveness",
        ],
        "llm_task_names": [
            "responsiveness",
            "responsivness",
            "responseiveness",
        ],
        "label_min": -1.0,
        "label_max": 1.0,
    },
    "agreement": {
        "type": "continuous",
        "human_patterns": [
            "agreement [-1,1]",
            "agreement [0,1]",
            "agreement 0,1]",
            "agreement",
        ],
        "llm_task_names": [
            "agreement",
        ],
        "label_min": -1.0,
        "label_max": 1.0,
    },
    "civility": {
        "type": "ordinal",
        "human_patterns": [
            "civility [1,6]",
            "civility",
        ],
        "llm_task_names": [
            "civility",
        ],
        "label_min": 1,
        "label_max": 6,
        "adjacent_tolerance": 1,
    },
    "sarcasm": {
        "type": "binary",
        "human_patterns": [
            "sarcasm[0,1]",
            "sarcasm [0,1]",
            "sarcasm",
        ],
        "llm_task_names": [
            "sarcasm",
        ],
        "label_min": 0,
        "label_max": 1,
    },
}

BASE_META_COLS = ["comment_id", "best_topic_index", "body", "predecessor"]
SUPPORTED_ANNOTATORS = ["Arthur", "Veronika"]


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def simplify_for_matching(s: str) -> str:
    """
    Etwas aggressivere Normalisierung für robustes Header-Matching.
    Entfernt Sonderzeichen weitgehend, damit z.B.
    'agreement 0,1]' und 'agreement [0,1]' ähnlich matchen.
    """
    s = normalize_text(s)
    s = s.replace(",", ".")
    s = re.sub(r"[\[\]\(\)\{\}_\-:/\\]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_numeric(series: pd.Series) -> pd.Series:
    def _convert(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return np.nan
            if x.upper() == "ABSTAIN":
                return np.nan
            x = x.replace(",", ".")
        return pd.to_numeric(x, errors="coerce")

    return series.map(_convert)


def read_excel_raw(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, header=None)


def is_effectively_empty(x: object) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower().startswith("unnamed")


def flatten_two_header_excel(path: Path) -> pd.DataFrame:
    """
    Erwartet eine Datei wie deine:
    Zeile 0: Gruppenköpfe (Arthur / Veronika, Meta leer)
    Zeile 1: eigentliche Spaltennamen
    Ab Zeile 2: Daten

    Wichtig:
    Die Gruppenköpfe in Zeile 0 werden nach rechts forward-filled,
    damit Arthur/Veronika für alle zugehörigen Taskspalten gelten.
    """
    raw = read_excel_raw(path)
    if raw.shape[0] < 3:
        raise ValueError("Excel-Datei hat zu wenige Zeilen für zweistufige Header-Erkennung.")

    row0 = pd.Series(raw.iloc[0].tolist(), dtype="object")
    row1 = raw.iloc[1].tolist()

    # Nur echte Annotator-Namen behalten, Rest leer setzen
    row0 = row0.map(
        lambda x: str(x).strip()
        if (not is_effectively_empty(x) and normalize_text(x) in {normalize_text(a) for a in SUPPORTED_ANNOTATORS})
        else np.nan
    )

    # Nach rechts auffüllen: Arthur gilt dann für alle folgenden Arthur-Spalten,
    # bis Veronika kommt
    row0 = row0.ffill().tolist()

    flattened_cols: List[str] = []
    for idx, (top, sub) in enumerate(zip(row0, row1)):
        top_s = "" if is_effectively_empty(top) else str(top).strip()
        sub_s = "" if is_effectively_empty(sub) else str(sub).strip()

        # Meta-Spalten links sollen NICHT fälschlich Arthur bekommen
        if idx < len(BASE_META_COLS):
            flattened_cols.append(sub_s if sub_s else top_s)
            continue

        if top_s and sub_s and normalize_text(top_s) in {normalize_text(a) for a in SUPPORTED_ANNOTATORS}:
            flattened_cols.append(f"{top_s}__{sub_s}")
        elif sub_s:
            flattened_cols.append(sub_s)
        elif top_s:
            flattened_cols.append(top_s)
        else:
            flattened_cols.append("")

    data = raw.iloc[2:].copy()
    data.columns = flattened_cols
    data = data.reset_index(drop=True)

    keep_cols = [c for c in data.columns if str(c).strip() != ""]
    data = data[keep_cols].copy()

    return data


def load_excel_with_flattened_headers(path: Path) -> pd.DataFrame:
    """
    Robustes Einlesen:
    1) Erst explizit zweizeilige Header-Struktur wie in deiner Datei
    2) Dann Versuch mit MultiHeader
    3) Fallback auf einfache Kopfzeile
    """
    # 1) Expliziter Zweizeilen-Header
    try:
        df_two = flatten_two_header_excel(path)
        cols_norm = [normalize_text(c) for c in df_two.columns]
        if any(c == "comment_id" for c in cols_norm):
            return df_two
    except Exception:
        pass

    # 2) MultiHeader-Versuch
    try:
        df_multi = pd.read_excel(path, header=[0, 1])
        if isinstance(df_multi.columns, pd.MultiIndex):
            flattened = []
            for top, sub in df_multi.columns:
                top_s = "" if pd.isna(top) else str(top).strip()
                sub_s = "" if pd.isna(sub) else str(sub).strip()

                top_empty = is_effectively_empty(top_s)
                sub_empty = is_effectively_empty(sub_s)

                if not top_empty and not sub_empty:
                    if normalize_text(top_s) in {normalize_text(a) for a in SUPPORTED_ANNOTATORS}:
                        flattened.append(f"{top_s}__{sub_s}")
                    else:
                        flattened.append(sub_s)
                elif not sub_empty:
                    flattened.append(sub_s)
                elif not top_empty:
                    flattened.append(top_s)
                else:
                    flattened.append("")

            df_multi.columns = flattened
            df_multi = df_multi[[c for c in df_multi.columns if str(c).strip() != ""]].copy()
            cols_norm = [normalize_text(c) for c in df_multi.columns]
            if any(c == "comment_id" for c in cols_norm):
                return df_multi
    except Exception:
        pass

    # 3) Fallback
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_base_column(df: pd.DataFrame, target_name: str) -> Optional[str]:
    target_norm = normalize_text(target_name)

    # exakter Match
    for col in df.columns:
        if normalize_text(col) == target_norm:
            return col

    # vereinfachter Match
    target_simple = simplify_for_matching(target_name)
    for col in df.columns:
        if simplify_for_matching(col) == target_simple:
            return col

    # enthält target
    for col in df.columns:
        if target_norm in normalize_text(col):
            return col

    return None


def find_annotator_task_column(df: pd.DataFrame, annotator: str, task_key: str) -> Optional[str]:
    annotator_norm = normalize_text(annotator)
    patterns_norm = [normalize_text(p) for p in TASK_CONFIG[task_key]["human_patterns"]]
    patterns_simple = [simplify_for_matching(p) for p in TASK_CONFIG[task_key]["human_patterns"]]

    candidates = []

    for col in df.columns:
        col_norm = normalize_text(col)
        col_simple = simplify_for_matching(col)

        if annotator_norm not in col_norm:
            continue

        score = 0
        for p in patterns_norm:
            if p in col_norm:
                score = max(score, 3)
        for p in patterns_simple:
            if p in col_simple:
                score = max(score, 2)

        # Schwächerer Fallback: Task-Key ohne Unterstrich
        fallback = task_key.replace("_", " ")
        if fallback in col_simple:
            score = max(score, 1)

        if score > 0:
            candidates.append((score, col))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def build_human_dataframe(
        df_raw: pd.DataFrame,
        active_annotators: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Baut ein standardisiertes Human-DataFrame:
    - comment_id
    - best_topic_index
    - body
    - predecessor
    - Arthur__stance_intensity, ...
    - Veronika__stance_intensity, ...
    """
    out = pd.DataFrame()
    column_mapping: Dict[str, str] = {}
    missing_columns: List[str] = []

    # Basisfelder
    for base_col in BASE_META_COLS:
        found = find_base_column(df_raw, base_col)
        if found is not None:
            out[base_col] = df_raw[found]
            column_mapping[base_col] = found
        else:
            if base_col == "comment_id":
                raise ValueError(f"Pflichtspalte '{base_col}' wurde in der Excel nicht gefunden.")
            out[base_col] = np.nan
            missing_columns.append(base_col)

    # Annotator-Spalten
    for annotator in active_annotators:
        for task_key in TASK_CONFIG.keys():
            found = find_annotator_task_column(df_raw, annotator=annotator, task_key=task_key)
            std_name = f"{annotator}__{task_key}"

            if found is None:
                out[std_name] = np.nan
                missing_columns.append(std_name)
            else:
                out[std_name] = safe_numeric(df_raw[found])
                column_mapping[std_name] = found

    out["comment_id"] = out["comment_id"].astype(str).str.strip()
    out = out[out["comment_id"].notna()].copy()
    out = out[out["comment_id"] != ""].copy()

    info = {
        "column_mapping": column_mapping,
        "missing_columns": missing_columns,
        "n_rows_raw": int(len(df_raw)),
        "n_rows_standardized": int(len(out)),
    }
    return out, info


def extract_llm_value(result_obj: dict) -> Optional[float]:
    """
    Im NDJSON kann der Wert in result.score oder result.label liegen.
    ABSTAIN -> NaN
    """
    if not isinstance(result_obj, dict):
        return np.nan

    if "score" in result_obj:
        val = result_obj.get("score")
    elif "label" in result_obj:
        val = result_obj.get("label")
    else:
        return np.nan

    if isinstance(val, str):
        val = val.strip()
        if val.upper() == "ABSTAIN":
            return np.nan
        val = val.replace(",", ".")

    try:
        return float(val)
    except Exception:
        return np.nan


def load_llm_ndjson(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Liest NDJSON long-format und pivotiert zu:
    - comment_id
    - llm__stance_intensity
    - llm__civility
    - ...
    plus optionale confidence-Spalten
    """
    rows = []
    parse_errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append({"line_no": line_no, "error": str(e)})
                continue

            comment_id = str(obj.get("comment_id", "")).strip()
            task = obj.get("task")
            result = obj.get("result", {})

            value = extract_llm_value(result)

            confidence = np.nan
            if isinstance(result, dict):
                conf = result.get("confidence")
                try:
                    if isinstance(conf, str):
                        conf = conf.replace(",", ".").strip()
                    confidence = float(conf)
                except Exception:
                    confidence = np.nan

            if task is None or comment_id == "":
                continue

            rows.append({
                "comment_id": comment_id,
                "task": str(task).strip(),
                "value": value,
                "confidence": confidence,
            })

    if not rows:
        raise ValueError("Keine gültigen LLM-Annotationen im NDJSON gefunden.")

    df_long = pd.DataFrame(rows)
    n_rows_before_filter = len(df_long)

    # Mapping LLM task -> Standardtask
    llm_task_to_std = {}
    for task_key, cfg in TASK_CONFIG.items():
        llm_task_to_std[normalize_task_name(task_key)] = task_key
        for llm_name in cfg["llm_task_names"]:
            llm_task_to_std[normalize_task_name(llm_name)] = task_key

    df_long["task_norm"] = df_long["task"].map(normalize_task_name)

    unknown_tasks = sorted(df_long.loc[~df_long["task_norm"].isin(llm_task_to_std.keys()), "task"].dropna().unique().tolist())

    df_long = df_long[df_long["task_norm"].isin(llm_task_to_std.keys())].copy()
    df_long["task_std"] = df_long["task_norm"].map(llm_task_to_std)

    # Duplikate prüfen
    dup_counts = (
        df_long.groupby(["comment_id", "task_std"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    dup_rows = dup_counts[dup_counts["n"] > 1].copy()

    # Werte pivotieren
    df_val = df_long.pivot_table(
        index="comment_id",
        columns="task_std",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Confidence pivotieren
    df_conf = df_long.pivot_table(
        index="comment_id",
        columns="task_std",
        values="confidence",
        aggfunc="first",
    ).reset_index()

    # Spalten umbenennen
    if "comment_id" in df_val.columns:
        df_val.columns = ["comment_id"] + [f"llm__{c}" for c in df_val.columns[1:]]
    if "comment_id" in df_conf.columns:
        df_conf.columns = ["comment_id"] + [f"llm_conf__{c}" for c in df_conf.columns[1:]]

    df = df_val.merge(df_conf, on="comment_id", how="outer")
    df["comment_id"] = df["comment_id"].astype(str).str.strip()

    info = {
        "n_rows_long_total": int(n_rows_before_filter),
        "n_rows_long_kept_known_tasks": int(len(df_long)),
        "n_rows_wide": int(len(df)),
        "parse_errors": parse_errors,
        "n_parse_errors": int(len(parse_errors)),
        "duplicate_comment_task_pairs_n": int(len(dup_rows)),
        "duplicate_comment_task_pairs_preview": dup_rows.head(50).to_dict(orient="records"),
        "known_tasks_present": sorted(df_long["task_std"].dropna().unique().tolist()),
        "raw_tasks_present": sorted(pd.Series(rows).map(lambda x: x["task"]).dropna().unique().tolist()),
        "unknown_tasks_after_normalization": unknown_tasks,
    }
    return df, info


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 1) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tolerance))


def is_binary_like(arr: np.ndarray) -> bool:
    vals = pd.Series(arr).dropna().unique().tolist()
    return set(vals).issubset({0, 1})


def safe_corr(func, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    try:
        stat, p = func(y_true, y_pred)
        stat = np.nan if pd.isna(stat) else float(stat)
        p = np.nan if pd.isna(p) else float(p)
        return stat, p
    except Exception:
        return np.nan, np.nan


def compute_discrete_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_min: int,
        label_max: int,
        adjacent_tolerance: Optional[int] = None,
) -> Dict[str, Any]:
    labels = list(range(label_min, label_max + 1))

    out: Dict[str, Any] = {
        "n_valid": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "cohens_kappa_unweighted": np.nan,
        "cohens_kappa_linear": np.nan,
        "cohens_kappa_quadratic": np.nan,
        "adjacent_accuracy": np.nan,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    if len(y_true) >= 2:
        rho, p = safe_corr(spearmanr, y_true, y_pred)
        out["spearman_rho"] = rho
        out["spearman_p"] = p

        r, p = safe_corr(pearsonr, y_true, y_pred)
        out["pearson_r"] = r
        out["pearson_p"] = p

        try:
            out["cohens_kappa_unweighted"] = float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            pass

        try:
            out["cohens_kappa_linear"] = float(cohen_kappa_score(y_true, y_pred, weights="linear"))
        except Exception:
            pass

        try:
            out["cohens_kappa_quadratic"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        except Exception:
            pass

    if adjacent_tolerance is not None:
        out["adjacent_accuracy"] = adjacent_accuracy(y_true, y_pred, tolerance=adjacent_tolerance)

    return out


def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_valid": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "cohens_kappa_unweighted": np.nan,
        "accuracy_exact": float(np.mean(y_true == y_pred)),
    }

    if len(y_true) >= 2:
        rho, p = safe_corr(spearmanr, y_true, y_pred)
        out["spearman_rho"] = rho
        out["spearman_p"] = p

        r, p = safe_corr(pearsonr, y_true, y_pred)
        out["pearson_r"] = r
        out["pearson_p"] = p

        if is_binary_like(y_true) and is_binary_like(y_pred):
            try:
                out["cohens_kappa_unweighted"] = float(
                    cohen_kappa_score(y_true.astype(int), y_pred.astype(int))
                )
            except Exception:
                pass

    return out


def compare_two_columns(
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        task_key: str,
        pair_name: str,
) -> Dict[str, Any]:
    cfg = TASK_CONFIG[task_key]

    tmp = df[[col_a, col_b]].copy()
    tmp[col_a] = safe_numeric(tmp[col_a])
    tmp[col_b] = safe_numeric(tmp[col_b])
    tmp = tmp.dropna(subset=[col_a, col_b]).copy()

    if tmp.empty:
        return {
            "pair_name": pair_name,
            "task": task_key,
            "task_type": cfg["type"],
            "n_valid": 0,
            "metrics": {},
        }

    y_true = tmp[col_a].to_numpy()
    y_pred = tmp[col_b].to_numpy()

    if cfg["type"] == "ordinal":
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        metrics = compute_discrete_metrics(
            y_true=y_true,
            y_pred=y_pred,
            label_min=int(cfg["label_min"]),
            label_max=int(cfg["label_max"]),
            adjacent_tolerance=cfg.get("adjacent_tolerance"),
        )

    elif cfg["type"] == "binary":
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        metrics = compute_discrete_metrics(
            y_true=y_true,
            y_pred=y_pred,
            label_min=int(cfg["label_min"]),
            label_max=int(cfg["label_max"]),
            adjacent_tolerance=None,
        )

    elif cfg["type"] == "binary_or_continuous":
        if is_binary_like(y_true) and is_binary_like(y_pred):
            y_true = y_true.astype(int)
            y_pred = y_pred.astype(int)
            metrics = compute_discrete_metrics(
                y_true=y_true,
                y_pred=y_pred,
                label_min=0,
                label_max=1,
                adjacent_tolerance=None,
            )
        else:
            metrics = compute_continuous_metrics(y_true=y_true, y_pred=y_pred)

    else:
        metrics = compute_continuous_metrics(y_true=y_true, y_pred=y_pred)

    return {
        "pair_name": pair_name,
        "task": task_key,
        "task_type": cfg["type"],
        "n_valid": int(metrics.get("n_valid", len(tmp))),
        "metrics": metrics,
    }


def flatten_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.get("metrics", {})
        row = {
            "pair_name": r.get("pair_name"),
            "task": r.get("task"),
            "task_type": r.get("task_type"),
            "n_valid": r.get("n_valid"),
        }
        for k, v in m.items():
            if k in {"confusion_matrix", "labels"}:
                continue
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion_matrices(results: List[Dict[str, Any]], output_dir: Path) -> None:
    for r in results:
        m = r.get("metrics", {})
        cm = m.get("confusion_matrix")
        labels = m.get("labels")
        if cm is None or labels is None:
            continue

        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        out_name = f"confusion_matrix__{r['pair_name']}__{r['task']}.csv"
        df_cm.to_csv(output_dir / out_name)


def write_json(data: dict, path: Path) -> None:
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)


def build_coverage_report(
        df_human: pd.DataFrame,
        df_llm: pd.DataFrame,
        df_merged: pd.DataFrame,
        active_annotators: List[str],
        tasks_to_run: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    human_ids = set(df_human["comment_id"].astype(str).tolist())
    llm_ids = set(df_llm["comment_id"].astype(str).tolist())
    matched_ids = human_ids & llm_ids

    pair_rows = []

    # Human-Human
    if len(active_annotators) == 2:
        ann_a, ann_b = active_annotators[0], active_annotators[1]
        for task_key in tasks_to_run:
            col_a = f"{ann_a}__{task_key}"
            col_b = f"{ann_b}__{task_key}"
            if col_a in df_merged.columns and col_b in df_merged.columns:
                n_valid = int(
                    df_merged[[col_a, col_b]]
                    .pipe(lambda x: x.assign(**{col_a: safe_numeric(x[col_a]), col_b: safe_numeric(x[col_b])}))
                    .dropna()
                    .shape[0]
                )
                pair_rows.append({
                    "pair_name": f"{ann_a}__vs__{ann_b}",
                    "task": task_key,
                    "n_valid": n_valid,
                })

    # Human-LLM
    for annotator in active_annotators:
        for task_key in tasks_to_run:
            col_h = f"{annotator}__{task_key}"
            col_l = f"llm__{task_key}"
            if col_h in df_merged.columns and col_l in df_merged.columns:
                n_valid = int(
                    df_merged[[col_h, col_l]]
                    .pipe(lambda x: x.assign(**{col_h: safe_numeric(x[col_h]), col_l: safe_numeric(x[col_l])}))
                    .dropna()
                    .shape[0]
                )
                pair_rows.append({
                    "pair_name": f"{annotator}__vs__llm",
                    "task": task_key,
                    "n_valid": n_valid,
                })

    coverage_meta = {
        "n_human_ids": int(len(human_ids)),
        "n_llm_ids": int(len(llm_ids)),
        "n_matched_ids": int(len(matched_ids)),
        "n_human_only_ids": int(len(human_ids - llm_ids)),
        "n_llm_only_ids": int(len(llm_ids - human_ids)),
        "matched_id_rate_vs_human": float(len(matched_ids) / len(human_ids)) if human_ids else np.nan,
        "matched_id_rate_vs_llm": float(len(matched_ids) / len(llm_ids)) if llm_ids else np.nan,
    }

    return pd.DataFrame(pair_rows), coverage_meta

def normalize_task_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = s.strip("_")
    return s

def print_missing_column_warnings(human_info: Dict[str, Any]) -> None:
    missing = human_info.get("missing_columns", [])
    if missing:
        print("[WARN] Einige erwartete Spalten wurden nicht gefunden und als NaN angelegt:")
        for col in missing:
            print(f"  - {col}")


def print_ndjson_warnings(llm_info: Dict[str, Any]) -> None:
    n_parse_errors = llm_info.get("n_parse_errors", 0)
    if n_parse_errors > 0:
        print(f"[WARN] {n_parse_errors} NDJSON-Zeilen konnten nicht geparst werden.")

    n_dups = llm_info.get("duplicate_comment_task_pairs_n", 0)
    if n_dups > 0:
        print(f"[WARN] {n_dups} doppelte (comment_id, task)-Paare im NDJSON gefunden. Pivot verwendet jeweils den ersten Wert.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_excel", type=str, required=True, help="Pfad zur Excel-Datei mit Human-Annotationen")
    parser.add_argument("--llm_ndjson", type=str, required=True, help="Pfad zur NDJSON-Datei mit LLM-Annotationen")
    parser.add_argument("--output_dir", type=str, required=True, help="Ausgabeordner")
    parser.add_argument(
        "--active_annotators",
        nargs="+",
        required=True,
        choices=SUPPORTED_ANNOTATORS,
        help="Welche Human-Annotator:innen aktiv ausgewertet werden sollen, z.B. Arthur Veronika",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASK_CONFIG.keys()),
        choices=list(TASK_CONFIG.keys()),
        help="Optional: nur bestimmte Tasks auswerten",
    )
    args = parser.parse_args()

    human_excel = Path(args.human_excel)
    llm_ndjson = Path(args.llm_ndjson)
    output_dir = Path(args.output_dir)
    active_annotators = args.active_annotators
    tasks_to_run = args.tasks

    ensure_dir(output_dir)

    print("[INFO] Lade Human-Excel ...")
    df_human_raw = load_excel_with_flattened_headers(human_excel)
    df_human, human_info = build_human_dataframe(df_human_raw, active_annotators=active_annotators)
    print_missing_column_warnings(human_info)

    print("[INFO] Lade LLM-NDJSON ...")
    df_llm, llm_info = load_llm_ndjson(llm_ndjson)
    print_ndjson_warnings(llm_info)

    print("[INFO] Merge Human + LLM über comment_id ...")
    df = df_human.merge(df_llm, on="comment_id", how="left", suffixes=("", "_llmdup"))

    # Speichern der gemergten Tabelle
    df.to_csv(output_dir / "merged_annotations.csv", index=False)

    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Human-Human nur wenn beide aktiv
    # -------------------------------------------------------------------------
    if len(active_annotators) == 2:
        ann_a, ann_b = active_annotators[0], active_annotators[1]
        for task_key in tasks_to_run:
            col_a = f"{ann_a}__{task_key}"
            col_b = f"{ann_b}__{task_key}"

            if col_a not in df.columns or col_b not in df.columns:
                continue

            res = compare_two_columns(
                df=df,
                col_a=col_a,
                col_b=col_b,
                task_key=task_key,
                pair_name=f"{ann_a}__vs__{ann_b}",
            )
            results.append(res)

    # -------------------------------------------------------------------------
    # Human-LLM für alle aktiven Humans
    # -------------------------------------------------------------------------
    for annotator in active_annotators:
        for task_key in tasks_to_run:
            col_h = f"{annotator}__{task_key}"
            col_l = f"llm__{task_key}"

            if col_h not in df.columns or col_l not in df.columns:
                continue

            res = compare_two_columns(
                df=df,
                col_a=col_h,
                col_b=col_l,
                task_key=task_key,
                pair_name=f"{annotator}__vs__llm",
            )
            results.append(res)

    # -------------------------------------------------------------------------
    # Ergebnis-Tabellen speichern
    # -------------------------------------------------------------------------
    df_summary = flatten_results(results)
    df_summary.to_csv(output_dir / "agreement_summary.csv", index=False)
    save_confusion_matrices(results, output_dir=output_dir)

    # Coverage
    df_coverage_pairs, coverage_meta = build_coverage_report(
        df_human=df_human,
        df_llm=df_llm,
        df_merged=df,
        active_annotators=active_annotators,
        tasks_to_run=tasks_to_run,
    )
    df_coverage_pairs.to_csv(output_dir / "coverage_by_pair_and_task.csv", index=False)

    # Column mapping
    df_column_mapping = pd.DataFrame(
        [
            {"standard_column": k, "source_column_in_excel": v}
            for k, v in human_info.get("column_mapping", {}).items()
        ]
    )
    df_column_mapping.to_csv(output_dir / "human_column_mapping.csv", index=False)

    # NDJSON duplicate preview
    dup_preview = llm_info.get("duplicate_comment_task_pairs_preview", [])
    pd.DataFrame(dup_preview).to_csv(output_dir / "llm_duplicate_comment_task_pairs_preview.csv", index=False)

    meta = {
        "human_excel": str(human_excel),
        "llm_ndjson": str(llm_ndjson),
        "output_dir": str(output_dir),
        "active_annotators": active_annotators,
        "tasks_run": tasks_to_run,
        "n_human_rows": int(len(df_human)),
        "n_llm_rows_wide": int(len(df_llm)),
        "n_merged_rows": int(len(df)),
        "available_columns": list(df.columns),
        "human_info": human_info,
        "llm_info": llm_info,
        "coverage_meta": coverage_meta,
        "results": results,
    }
    write_json(meta, output_dir / "agreement_results.json")

    print("\n[OK] Fertig.")
    print(f"[OK] Output: {output_dir}")

    print("\n[INFO] Coverage:")
    for k, v in coverage_meta.items():
        print(f"  {k}: {v}")

    print("\n[INFO] Summary:")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    else:
        print("Keine vergleichbaren Daten gefunden.")


if __name__ == "__main__":
    main()
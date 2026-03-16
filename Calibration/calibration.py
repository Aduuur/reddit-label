#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agreement_analysis_reddit.py

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
   - Arthur:    stance..., epistemic..., justification..., responsiveness..., agreement..., civility..., sarcasm...
   - Veronika:  stance..., epistemic..., justification..., responsiveness..., agreement..., civility..., sarcasm...

2) LLM-Annotationen in NDJSON
   - eine Zeile pro (comment_id, task)
   - result.score oder result.label
   - task z.B. stance_intensity, civility, epistemic_modality, ...

Features
--------
- frei wählbare aktive Human-Annotator:innen
- Human-Human nur wenn 2 Humans aktiv sind
- Human-LLM für alle aktiven Humans
- automatische Umwandlung des NDJSON long -> wide
- ordinale und kontinuierliche Tasks werden unterschiedlich ausgewertet
- robuste Spaltennormalisierung für leicht unterschiedliche Excel-Benennungen

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
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Kanonische Tasknamen intern
TASK_CONFIG = {
    "stance_intensity": {
        "type": "ordinal",
        "human_patterns": [
            "stance intensity [1,6]",
            "stance intensity",
        ],
        "llm_task_names": ["stance_intensity"],
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
        "llm_task_names": ["epistemic_modality"],
        "label_min": 0.0,
        "label_max": 1.0,
    },
    "justification_density": {
        "type": "continuous",
        "human_patterns": [
            "justificartion density [0,1]",   # absichtlicher Excel-Typo mit abgedeckt
            "justification density [0,1]",
            "justificartion density",
            "justification density",
        ],
        "llm_task_names": ["justification_density"],
        "label_min": 0.0,
        "label_max": 1.0,
    },
    "responsiveness": {
        "type": "continuous",
        "human_patterns": [
            "responsivness [0,1]",           # absichtlicher Excel-Typo mit abgedeckt
            "responsiveness [0,1]",
            "responsivness",
            "responsiveness",
        ],
        "llm_task_names": ["responsiveness"],
        "label_min": 0.0,
        "label_max": 1.0,
    },
    "agreement": {
        "type": "continuous",
        "human_patterns": [
            "agreement [0,1]",
            "agreement",
        ],
        "llm_task_names": ["agreement"],
        "label_min": 0.0,
        "label_max": 1.0,
    },
    "civility": {
        "type": "ordinal",
        "human_patterns": [
            "civility [1,6]",
            "civility",
        ],
        "llm_task_names": ["civility"],
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
        "llm_task_names": ["sarcasm"],
        "label_min": 0.0,
        "label_max": 1.0,
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
    s = re.sub(r"\s+", " ", s)
    return s


def safe_numeric(series: pd.Series) -> pd.Series:
    def _convert(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            if x.strip().upper() == "ABSTAIN":
                return np.nan
            x = x.strip()
        return pd.to_numeric(x, errors="coerce")
    return series.map(_convert)


def load_excel_with_flattened_headers(path: Path) -> pd.DataFrame:
    """
    Liest Excel robust ein.
    Falls 2 Kopfzeilen vorhanden sind, werden sie zusammengeführt.
    Falls nur 1 Kopfzeile vorhanden ist, wird diese normal verwendet.
    """
    # Versuch 1: Multi-Header lesen
    try:
        df_multi = pd.read_excel(path, header=[0, 1])
        # Wenn wirklich MultiIndex-artig
        if isinstance(df_multi.columns, pd.MultiIndex):
            flattened = []
            for top, sub in df_multi.columns:
                top_s = "" if pd.isna(top) else str(top).strip()
                sub_s = "" if pd.isna(sub) else str(sub).strip()

                if sub_s and sub_s.lower() != "unnamed":
                    if top_s and not sub_s.startswith("Unnamed"):
                        if normalize_text(top_s) in {"arthur", "veronika"}:
                            flattened.append(f"{top_s}__{sub_s}")
                        else:
                            # z.B. Basisfelder comment_id etc.
                            flattened.append(sub_s if sub_s else top_s)
                    else:
                        flattened.append(sub_s)
                else:
                    flattened.append(top_s)

            df_multi.columns = flattened
            # Heuristik: sinnvoll nur, wenn comment_id o.ä. auftaucht
            cols_norm = [normalize_text(c) for c in df_multi.columns]
            if any("comment_id" in c for c in cols_norm):
                return df_multi
    except Exception:
        pass

    # Fallback: einfache Kopfzeile
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_base_column(df: pd.DataFrame, target_name: str) -> Optional[str]:
    target_norm = normalize_text(target_name)
    for col in df.columns:
        if normalize_text(col) == target_norm:
            return col
    for col in df.columns:
        if target_norm in normalize_text(col):
            return col
    return None


def find_annotator_task_column(df: pd.DataFrame, annotator: str, task_key: str) -> Optional[str]:
    patterns = [normalize_text(p) for p in TASK_CONFIG[task_key]["human_patterns"]]
    annotator_norm = normalize_text(annotator)

    # 1) Bevorzugt Spalten mit Prefix "Arthur__..." oder "Veronika__..."
    for col in df.columns:
        col_norm = normalize_text(col)
        if annotator_norm in col_norm:
            if any(p in col_norm for p in patterns):
                return col

    # 2) Falls die Excel bereits manuell umbenannt wurde, z.B. "Arthur stance intensity [1,6]"
    for col in df.columns:
        col_norm = normalize_text(col)
        if annotator_norm in col_norm and any(p in col_norm for p in patterns):
            return col

    return None


def build_human_dataframe(df_raw: pd.DataFrame, active_annotators: List[str]) -> pd.DataFrame:
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

    # Basisfelder
    for base_col in BASE_META_COLS:
        found = find_base_column(df_raw, base_col)
        if found is not None:
            out[base_col] = df_raw[found]
        else:
            if base_col == "comment_id":
                raise ValueError(f"Pflichtspalte '{base_col}' wurde in der Excel nicht gefunden.")
            out[base_col] = np.nan

    # Annotator-Spalten
    for annotator in active_annotators:
        for task_key in TASK_CONFIG.keys():
            found = find_annotator_task_column(df_raw, annotator=annotator, task_key=task_key)
            std_name = f"{annotator}__{task_key}"

            if found is None:
                out[std_name] = np.nan
            else:
                out[std_name] = safe_numeric(df_raw[found])

    out["comment_id"] = out["comment_id"].astype(str).str.strip()
    return out


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

    if isinstance(val, str) and val.strip().upper() == "ABSTAIN":
        return np.nan

    try:
        return float(val)
    except Exception:
        return np.nan


def load_llm_ndjson(path: Path) -> pd.DataFrame:
    """
    Liest NDJSON long-format und pivotiert zu:
    - comment_id
    - llm__stance_intensity
    - llm__civility
    - ...
    plus optionale confidence-Spalten
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] NDJSON-Zeile {line_no} konnte nicht geparst werden: {e}")
                continue

            comment_id = str(obj.get("comment_id", "")).strip()
            task = obj.get("task")
            result = obj.get("result", {})
            value = extract_llm_value(result)

            confidence = np.nan
            if isinstance(result, dict):
                conf = result.get("confidence")
                try:
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

    # Nur Tasks behalten, die wir kennen
    llm_task_to_std = {}
    for task_key, cfg in TASK_CONFIG.items():
        for llm_name in cfg["llm_task_names"]:
            llm_task_to_std[llm_name] = task_key

    df_long = df_long[df_long["task"].isin(llm_task_to_std.keys())].copy()
    df_long["task_std"] = df_long["task"].map(llm_task_to_std)

    # Werte pivotieren
    df_val = df_long.pivot_table(
        index="comment_id",
        columns="task_std",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Confidence pivotieren
    df_conf = df_long.pivot_table(
        index="comment_id",
        columns="task_std",
        values="confidence",
        aggfunc="first"
    ).reset_index()

    # Spalten umbenennen
    if "comment_id" in df_val.columns:
        df_val.columns = ["comment_id"] + [f"llm__{c}" for c in df_val.columns[1:]]
    if "comment_id" in df_conf.columns:
        df_conf.columns = ["comment_id"] + [f"llm_conf__{c}" for c in df_conf.columns[1:]]

    df = df_val.merge(df_conf, on="comment_id", how="outer")
    df["comment_id"] = df["comment_id"].astype(str).str.strip()
    return df


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 1) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tolerance))


def is_binary_like(arr: np.ndarray) -> bool:
    vals = pd.Series(arr).dropna().unique().tolist()
    vals = sorted(vals)
    return set(vals).issubset({0, 1})


def compute_discrete_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_min: int,
        label_max: int,
        adjacent_tolerance: Optional[int] = None,
) -> Dict:
    labels = list(range(label_min, label_max + 1))

    out = {
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
        try:
            rho, p = spearmanr(y_true, y_pred)
            out["spearman_rho"] = float(rho)
            out["spearman_p"] = float(p)
        except Exception:
            pass

        try:
            r, p = pearsonr(y_true, y_pred)
            out["pearson_r"] = float(r)
            out["pearson_p"] = float(p)
        except Exception:
            pass

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


def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    out = {
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
        try:
            rho, p = spearmanr(y_true, y_pred)
            out["spearman_rho"] = float(rho)
            out["spearman_p"] = float(p)
        except Exception:
            pass

        try:
            r, p = pearsonr(y_true, y_pred)
            out["pearson_r"] = float(r)
            out["pearson_p"] = float(p)
        except Exception:
            pass

        # Kappa nur wenn binär/diskret
        if is_binary_like(y_true) and is_binary_like(y_pred):
            try:
                out["cohens_kappa_unweighted"] = float(cohen_kappa_score(y_true.astype(int), y_pred.astype(int)))
            except Exception:
                pass

    return out


def compare_two_columns(
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        task_key: str,
        pair_name: str,
) -> Dict:
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
    elif cfg["type"] == "binary_or_continuous":
        # Falls tatsächlich binär -> diskrete Metriken, sonst kontinuierlich
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


def flatten_results(results: List[Dict]) -> pd.DataFrame:
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


def save_confusion_matrices(results: List[Dict], output_dir: Path) -> None:
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
        help="Welche Human-Annotator:innen aktiv ausgewertet werden sollen, z.B. Arthur Veronika"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASK_CONFIG.keys()),
        choices=list(TASK_CONFIG.keys()),
        help="Optional: nur bestimmte Tasks auswerten"
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
    df_human = build_human_dataframe(df_human_raw, active_annotators=active_annotators)

    print("[INFO] Lade LLM-NDJSON ...")
    df_llm = load_llm_ndjson(llm_ndjson)

    print("[INFO] Merge Human + LLM über comment_id ...")
    df = df_human.merge(df_llm, on="comment_id", how="left", suffixes=("", "_llmdup"))

    # Speichern der gemergten Tabelle
    df.to_csv(output_dir / "merged_annotations.csv", index=False)

    results = []

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
        "results": results,
    }
    write_json(meta, output_dir / "agreement_results.json")

    print("\n[OK] Fertig.")
    print(f"[OK] Output: {output_dir}")
    print("\n[INFO] Summary:")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    else:
        print("Keine vergleichbaren Daten gefunden.")


if __name__ == "__main__":
    main()
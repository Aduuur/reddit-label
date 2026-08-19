"""
data_io.py
==========
Daten-Ein-/Ausgabe ohne pandas/numpy (nur Standardbibliothek + optional
openpyxl für xlsx). Wird von beiden Pipelines (seriell + async) genutzt.

Rückgabe von load_input: Liste von dicts (records) statt DataFrame.
Jede Zeile ist ein dict mit den Spalten als Keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple


def _clean_col(x: Any) -> str:
    import re
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", "_", s.strip())
    return re.sub(r"[^A-Za-z0-9_\-]", "", s).strip("_").lower()


def load_input(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Lädt Input-Daten. Unterstützt .ndjson/.jsonl/.json und .xlsx/.xls.
    Gibt (records, columns) zurück. Überspringt kaputte NDJSON-Zeilen robust.
    """
    suffix = path.suffix.lower()

    if suffix in {".ndjson", ".jsonl", ".json"}:
        records: List[Dict[str, Any]] = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1
        if skipped:
            print(f"  {skipped} kaputte/ungültige Zeile(n) übersprungen", flush=True)
        # Spalten = Vereinigung aller Keys (Reihenfolge: erste Vorkommen)
        cols: List[str] = []
        seen: Set[str] = set()
        for r in records:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        return records, cols

    if suffix in {".xlsx", ".xls"}:
        try:
            from openpyxl import load_workbook
        except ImportError as e:
            raise ImportError(
                "openpyxl fehlt im Container. Entweder Input als .ndjson liefern "
                "oder openpyxl in den Container installieren."
            ) from e
        wb = load_workbook(filename=str(path), read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        try:
            header = next(rows_iter)
        except StopIteration:
            return [], []
        cols = [_clean_col(h) for h in header]
        records = []
        for row in rows_iter:
            rec = {}
            for c, v in zip(cols, row):
                rec[c] = v
            records.append(rec)
        wb.close()
        return records, cols

    raise ValueError(f"Unbekanntes Format: {suffix} (nur .ndjson/.xlsx)")


def is_empty(v: Any) -> bool:
    """Ersetzt pd.isna für unsere Zwecke: None, leerer String, oder NaN-float."""
    if v is None:
        return True
    if isinstance(v, float):
        # NaN ist das einzige float das != sich selbst ist
        return v != v
    if isinstance(v, str):
        return v.strip() == ""
    return False


# Schwelle für das deterministische Längen-ABSTAIN.
# Texte mit weniger als so vielen Tokens (grob = Wörter) gelten als zu kurz,
# um zuverlässig gelabelt zu werden -> alle Tasks ABSTAIN, ohne LLM-Aufruf.
MIN_TOKENS = 15


def count_tokens(text: str) -> int:
    """Grobe Tokenzahl über Whitespace-Split. Reicht für die Längen-Schwelle;
    wir brauchen keine exakte Tokenizer-Zählung, nur eine robuste Näherung."""
    if not text:
        return 0
    return len(text.split())


def is_too_short(text: str, min_tokens: int = MIN_TOKENS) -> bool:
    """True, wenn der Text unter der Mindestlänge liegt (deterministisches
    ABSTAIN-Kriterium)."""
    return count_tokens(text) < min_tokens


def jsonable(v: Any) -> Any:
    """Macht einen Wert JSON-serialisierbar (ohne numpy-Abhängigkeit)."""
    if is_empty(v):
        return None
    # numpy-Typen abfangen falls sie doch mal auftauchen (ohne numpy zu importieren)
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:
            pass
    return v


def load_done_pairs(ndjson_path: Path) -> Set[Tuple[str, str]]:
    """Lädt bereits gelabelte (comment_id, task)-Paare für Wiederaufnahme."""
    done: Set[Tuple[str, str]] = set()
    if not ndjson_path.exists():
        return done
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cid = str(obj.get("comment_id", ""))
                task = str(obj.get("task", ""))
                if cid and task:
                    done.add((cid, task))
            except Exception:
                continue
    return done

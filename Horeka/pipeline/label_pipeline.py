"""
label_pipeline.py
=================
Label-Pipeline für den HPC-Betrieb. Spricht einen vLLM-Server an, der auf
DEMSELBEN Node läuft (localhost) - daher kein SSH-Tunnel nötig.

Aufruf:
    python3 label_pipeline.py \
        --input   /pfad/zu/input.xlsx \
        --output  /pfad/zu/results.ndjson \
        --base-url http://localhost:8000 \
        --model   llama-70b

Der Server wird von run_labeling.sh gestartet, bevor dieses Script läuft.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Callable

import requests
from openai import OpenAI, APIConnectionError, APIStatusError

from prompts import (
    TASK_SPECS,
    SYSTEM_PROMPT,
    ARG_SYSTEM_PROMPT,
    build_user_prompt,
    build_argument_prompt,
)
from data_io import load_input, is_empty, jsonable, load_done_pairs, is_too_short

# =============================================================================
# Defaults (per CLI überschreibbar)
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "llama-70b"
TIMEOUT_SECONDS = 180
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0


# =============================================================================
# LLM-Aufruf
# =============================================================================

def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=f"{base_url}/v1", api_key="dummy", timeout=TIMEOUT_SECONDS)


def call_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
) -> str:
    """Ein Chat-Call gegen den lokalen vLLM-Server. Fehler werden als
    requests.RequestException geworfen, damit die Retry-Logik greift."""
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except (APIConnectionError, APIStatusError) as e:
        raise requests.RequestException(f"vLLM connection error: {e}") from e
    except Exception as e:
        raise requests.RequestException(f"vLLM unexpected error: {e}") from e


# =============================================================================
# Validierung
# =============================================================================

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_response(task: str, obj: Dict[str, Any]) -> Optional[str]:
    spec = TASK_SPECS[task]
    if not isinstance(obj, dict):
        return "Response is not a JSON object"
    missing = spec["schema_keys"] - set(obj.keys())
    if missing:
        return f"Missing required keys: {sorted(missing)}"
    if obj.get("task") != spec["task_value"]:
        return f'Field "task" must be "{spec["task_value"]}"'
    conf = obj.get("confidence")
    if not _is_number(conf):
        return '"confidence" must be a number'
    lo_c, hi_c = spec["confidence_range"]
    if not (lo_c <= conf <= hi_c):
        return f'"confidence" must be in [{lo_c}, {hi_c}]'

    label_key = spec.get("label_key")
    score_key = spec.get("score_key")

    if label_key is not None:
        val = obj.get(label_key)
        if isinstance(val, str):
            if spec.get("allow_abstain") and val == "ABSTAIN":
                return None
            return f'"{label_key}" must be an integer in range or "ABSTAIN"'
        if not isinstance(val, spec.get("label_type", (int,))):
            return f'"{label_key}" has wrong type'
        lo, hi = spec["score_range"]
        if not (lo <= float(val) <= hi):
            return f'"{label_key}" out of range [{lo}, {hi}]'
        return None

    if score_key is not None:
        val = obj.get(score_key)
        if isinstance(val, str):
            if spec.get("allow_abstain") and val == "ABSTAIN":
                return None
            return f'"{score_key}" must be a number or "ABSTAIN"'
        if not _is_number(val):
            return f'"{score_key}" must be a number'
        lo, hi = spec["score_range"]
        if not (lo <= float(val) <= hi):
            return f'"{score_key}" out of range [{lo}, {hi}]'
        return None
    return None


def validate_arguments(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj, dict):
        return "Response is not a JSON object"
    if obj.get("task") != "argument_extraction":
        return 'Field "task" must be "argument_extraction"'
    if "arguments" not in obj or "confidence" not in obj:
        return "Missing keys 'arguments'/'confidence'"
    if not isinstance(obj["arguments"], list):
        return '"arguments" must be a list'
    for i, a in enumerate(obj["arguments"]):
        if not isinstance(a, dict) or "claim" not in a or "proof" not in a:
            return f'"arguments[{i}]" needs "claim" and "proof"'
        if not isinstance(a["claim"], str) or not isinstance(a["proof"], str):
            return f'"arguments[{i}]" claim/proof must be strings'
    if not _is_number(obj["confidence"]) or not (0.0 <= obj["confidence"] <= 1.0):
        return '"confidence" must be a number in [0,1]'
    return None


# =============================================================================
# Annotation mit Retry
# =============================================================================

def annotate_one(client, model, task, text, parent_text=None) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task, text, parent_text)},
            ]
            raw = call_chat(client, model, messages, temperature=0.0)
            obj = json.loads(raw)
            err = validate_response(task, obj)
            if err is None:
                return obj
            last_err = f"schema (attempt {attempt}): {err}"
        except requests.RequestException as e:
            last_err = f"http (attempt {attempt}): {e}"
        except json.JSONDecodeError as e:
            last_err = f"json (attempt {attempt}): {e}"
        time.sleep(RETRY_BACKOFF_SECONDS ** min(attempt, 4))
    return {"task": task, "error": last_err or "unknown"}


def extract_arguments(client, model, text, parent_text=None) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": ARG_SYSTEM_PROMPT},
                {"role": "user", "content": build_argument_prompt(text, parent_text)},
            ]
            raw = call_chat(client, model, messages, temperature=0.0)
            obj = json.loads(raw)
            err = validate_arguments(obj)
            if err is None:
                return obj
            last_err = f"schema (attempt {attempt}): {err}"
        except requests.RequestException as e:
            last_err = f"http (attempt {attempt}): {e}"
        except json.JSONDecodeError as e:
            last_err = f"json (attempt {attempt}): {e}"
        time.sleep(RETRY_BACKOFF_SECONDS ** min(attempt, 4))
    return {"task": "argument_extraction", "error": last_err or "unknown"}


# =============================================================================
# Daten laden + Helfer -> jetzt in data_io.py (pandas-frei)
# load_input, is_empty, jsonable, load_done_pairs kommen aus data_io.
# =============================================================================

def _abstain(task: str, reason="EMPTY_BODY") -> Dict[str, Any]:
    if task in {"civility", "sarcasm"}:
        return {"task": task, "label": "ABSTAIN", "confidence": 0.0, "note": reason}
    return {"task": task, "score": "ABSTAIN", "confidence": 0.0, "note": reason}


# =============================================================================
# Hauptschleife
# =============================================================================

def run(
    rows: List[Dict[str, Any]],
    all_cols: List[str],
    client: OpenAI,
    model: str,
    ndjson_path: Path,
    tasks: List[str],
    id_col: str,
    text_col: str,
    parent_col: str,
    skip_existing: bool = True,
) -> Dict[str, Any]:

    for c in (id_col, text_col):
        if c not in all_cols:
            raise ValueError(f"Spalte '{c}' fehlt. Vorhanden: {all_cols}")

    done = load_done_pairs(ndjson_path) if skip_existing else set()
    written = skipped = errors = 0
    n = len(rows)

    with open(ndjson_path, "a", encoding="utf-8") as fp:
        for idx, row in enumerate(rows):
            cid = "" if is_empty(row.get(id_col)) else str(row.get(id_col))
            text = "" if is_empty(row.get(text_col)) else str(row.get(text_col))
            parent = None
            if not is_empty(row.get(parent_col)):
                parent = str(row.get(parent_col)) or None
            meta = {
                c: jsonable(row.get(c)) for c in all_cols
                if c not in {id_col, text_col, parent_col}
            }

            # Deterministisches ABSTAIN: leerer ODER zu kurzer Text (< MIN_TOKENS)
            if not cid or not text.strip() or is_too_short(text):
                reason = "EMPTY_BODY" if (not text.strip()) else "TOO_SHORT"
                for task in tasks:
                    if skip_existing and (cid, task) in done:
                        skipped += 1
                        continue
                    fp.write(json.dumps({
                        "comment_id": cid, "body": text, "predecessor": parent,
                        "comment_index": int(idx), "task": task,
                        "result": _abstain(task, reason=reason), "arguments": [],
                        "arguments_confidence": None, "arguments_error": reason,
                        "meta": meta,
                    }, ensure_ascii=False) + "\n")
                    written += 1
                if (idx + 1) % 50 == 0:
                    fp.flush()
                    print(f"[{idx+1}/{n}] written={written} skipped={skipped} errors={errors}", flush=True)
                continue

            # Kontextabhängige Tasks ohne Parent -> ABSTAIN (fehlende Grundlage)
            CONTEXT_TASKS = {"responsiveness", "agreement"}
            no_parent = parent is None
            llm_tasks = []
            for task in tasks:
                if skip_existing and (cid, task) in done:
                    skipped += 1
                    continue
                if no_parent and task in CONTEXT_TASKS:
                    fp.write(json.dumps({
                        "comment_id": cid, "body": text, "predecessor": parent,
                        "comment_index": int(idx), "task": task,
                        "result": _abstain(task, reason="NO_PARENT"), "arguments": [],
                        "arguments_confidence": None, "arguments_error": "NO_PARENT",
                        "meta": meta,
                    }, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    llm_tasks.append(task)

            if not llm_tasks:
                if (idx + 1) % 50 == 0:
                    fp.flush()
                    print(f"[{idx+1}/{n}] written={written} skipped={skipped} errors={errors}", flush=True)
                continue

            # Argument-Extraktion (einmal pro Kommentar)
            arg_res = extract_arguments(client, model, text, parent)
            if "error" in arg_res:
                arguments, arg_conf, arg_err = [], None, arg_res["error"]
            else:
                arguments = arg_res.get("arguments", [])
                arg_conf = arg_res.get("confidence")
                arg_err = None

            # Labeling-Tasks (nur die, die ans Modell gehen)
            for task in llm_tasks:
                try:
                    result = annotate_one(client, model, task, text, parent)
                    fp.write(json.dumps({
                        "comment_id": cid, "body": text, "predecessor": parent,
                        "comment_index": int(idx), "task": task,
                        "result": result, "arguments": arguments,
                        "arguments_confidence": arg_conf, "arguments_error": arg_err,
                        "meta": meta,
                    }, ensure_ascii=False) + "\n")
                    written += 1
                    if "error" in result:
                        errors += 1
                except Exception as e:
                    errors += 1
                    print(f"  ERROR bei {cid}/{task}: {e}", flush=True)

            if (idx + 1) % 50 == 0:
                fp.flush()
                print(f"[{idx+1}/{n}] written={written} skipped={skipped} errors={errors}", flush=True)

    return {"processed": n, "written": written, "skipped": skipped, "errors": errors}


# =============================================================================
# Server-Wartelogik
# =============================================================================

def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    """Wartet bis der vLLM-Server auf localhost antwortet (max. timeout Sek.)."""
    url = f"{base_url}/health"
    start = time.time()
    print(f"Warte auf Server unter {url} ...", flush=True)
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"Server bereit nach {int(time.time()-start)}s.", flush=True)
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="HPC Labeling Pipeline")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--id-col", default="comment_id")
    ap.add_argument("--text-col", default="body")
    ap.add_argument("--parent-col", default="predecessor")
    ap.add_argument("--tasks", nargs="*", default=list(TASK_SPECS.keys()))
    ap.add_argument("--no-skip", action="store_true", help="Bereits gelabelte NICHT überspringen")
    ap.add_argument("--wait-server", type=int, default=600, help="Max. Sek. auf Server warten")
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print("HPC LABELING PIPELINE", flush=True)
    print(f"  Input:   {args.input}", flush=True)
    print(f"  Output:  {args.output}", flush=True)
    print(f"  Server:  {args.base_url}  (model: {args.model})", flush=True)
    print(f"  Tasks:   {args.tasks}", flush=True)
    print("=" * 60, flush=True)

    # 1. Auf Server warten
    if not wait_for_server(args.base_url, timeout=args.wait_server):
        print("FEHLER: Server nicht erreichbar. Abbruch.", flush=True)
        sys.exit(1)

    # 2. Daten laden (data_io gibt (records, columns) zurück)
    print(f"Lade Input: {args.input}", flush=True)
    rows, all_cols = load_input(args.input)
    print(f"  {len(rows)} Zeilen geladen. Spalten: {all_cols}", flush=True)

    # 3. Labeln
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = make_client(args.base_url)
    stats = run(
        rows=rows, all_cols=all_cols, client=client, model=args.model,
        ndjson_path=args.output, tasks=args.tasks, id_col=args.id_col,
        text_col=args.text_col, parent_col=args.parent_col,
        skip_existing=not args.no_skip,
    )

    print("=" * 60, flush=True)
    print(f"FERTIG: {stats}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

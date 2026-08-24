"""
label_pipeline_async.py
=======================
Parallelisierte Label-Pipeline für den HPC-Vollbetrieb (mehrere 100.000
Kommentare). Nutzt asyncio + AsyncOpenAI, um viele Requests gleichzeitig an
den vLLM-Server zu schicken. vLLM batcht diese intern (continuous batching),
was den Durchsatz gegenüber der seriellen Version um ein Vielfaches erhöht.

Kernpunkte:
- Concurrency per --concurrency Flag begrenzt (Semaphore)
- Ein einziger Writer (Queue) schreibt NDJSON -> keine korrupten Zeilen
- Wiederaufnahme: bereits gelabelte (comment_id, task) werden übersprungen
- Durchsatz-Anzeige (Calls/s) zur Laufzeit-Hochrechnung

Aufruf:
    python3 label_pipeline_async.py \
        --input    /pfad/input.ndjson \
        --output   /pfad/results.ndjson \
        --base-url http://localhost:8000 \
        --model    llama-70b \
        --concurrency 48
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

from openai import AsyncOpenAI, APIConnectionError, APIStatusError

from prompts import (
    TASK_SPECS,
    SYSTEM_PROMPT,
    ARG_SYSTEM_PROMPT,
    build_user_prompt,
    build_argument_prompt,
)
from data_io import load_input, is_empty, jsonable, load_done_pairs, count_tokens

# =============================================================================
# Defaults
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "llama-70b"
DEFAULT_CONCURRENCY = 48
TIMEOUT_SECONDS = 180
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 2.0


# =============================================================================
# Validierung (identisch zur seriellen Version)
# =============================================================================

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_response(task: str, obj: Dict[str, Any]) -> Optional[str]:
    spec = TASK_SPECS[task]
    if not isinstance(obj, dict):
        return "not a JSON object"
    missing = spec["schema_keys"] - set(obj.keys())
    if missing:
        return f"missing keys: {sorted(missing)}"
    if obj.get("task") != spec["task_value"]:
        return f'task must be "{spec["task_value"]}"'
    conf = obj.get("confidence")
    if not _is_number(conf):
        return "confidence not a number"
    lo_c, hi_c = spec["confidence_range"]
    if not (lo_c <= conf <= hi_c):
        return f"confidence out of [{lo_c},{hi_c}]"

    label_key, score_key = spec.get("label_key"), spec.get("score_key")
    if label_key is not None:
        val = obj.get(label_key)
        if isinstance(val, str):
            return None if (spec.get("allow_abstain") and val == "ABSTAIN") \
                else f'{label_key} must be int or "ABSTAIN"'
        if not isinstance(val, spec.get("label_type", (int,))):
            return f"{label_key} wrong type"
        lo, hi = spec["score_range"]
        return None if lo <= float(val) <= hi else f"{label_key} out of [{lo},{hi}]"
    if score_key is not None:
        val = obj.get(score_key)
        if isinstance(val, str):
            return None if (spec.get("allow_abstain") and val == "ABSTAIN") \
                else f'{score_key} must be number or "ABSTAIN"'
        if not _is_number(val):
            return f"{score_key} not a number"
        lo, hi = spec["score_range"]
        return None if lo <= float(val) <= hi else f"{score_key} out of [{lo},{hi}]"
    return None


def validate_arguments(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not a JSON object"
    if obj.get("task") != "argument_extraction":
        return 'task must be "argument_extraction"'
    if "arguments" not in obj or "confidence" not in obj:
        return "missing arguments/confidence"
    if not isinstance(obj["arguments"], list):
        return "arguments not a list"
    for i, a in enumerate(obj["arguments"]):
        if not isinstance(a, dict) or "claim" not in a or "proof" not in a:
            return f"arguments[{i}] needs claim/proof"
        if not isinstance(a["claim"], str) or not isinstance(a["proof"], str):
            return f"arguments[{i}] claim/proof not strings"
    if not _is_number(obj["confidence"]) or not (0.0 <= obj["confidence"] <= 1.0):
        return "confidence out of [0,1]"
    return None


# =============================================================================
# Async LLM-Calls mit Retry
# =============================================================================

async def _chat(client, model, messages) -> str:
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.0
    )
    return resp.choices[0].message.content.strip()


async def annotate_one(client, model, task, text, parent) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task, text, parent)},
            ]
            raw = await _chat(client, model, messages)
            obj = json.loads(raw)
            err = validate_response(task, obj)
            if err is None:
                return obj
            last_err = f"schema({attempt}): {err}"
        except (APIConnectionError, APIStatusError) as e:
            last_err = f"conn({attempt}): {e}"
        except json.JSONDecodeError as e:
            last_err = f"json({attempt}): {e}"
        except Exception as e:
            last_err = f"err({attempt}): {e}"
        await asyncio.sleep(RETRY_BACKOFF_BASE ** min(attempt, 4))
    return {"task": task, "error": last_err or "unknown"}


async def extract_arguments(client, model, text, parent) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": ARG_SYSTEM_PROMPT},
                {"role": "user", "content": build_argument_prompt(text, parent)},
            ]
            raw = await _chat(client, model, messages)
            obj = json.loads(raw)
            err = validate_arguments(obj)
            if err is None:
                return obj
            last_err = f"schema({attempt}): {err}"
        except (APIConnectionError, APIStatusError) as e:
            last_err = f"conn({attempt}): {e}"
        except json.JSONDecodeError as e:
            last_err = f"json({attempt}): {e}"
        except Exception as e:
            last_err = f"err({attempt}): {e}"
        await asyncio.sleep(RETRY_BACKOFF_BASE ** min(attempt, 4))
    return {"task": "argument_extraction", "error": last_err or "unknown"}


# =============================================================================
# Daten laden -> jetzt in data_io.py (pandas-frei)
# load_input, is_empty, jsonable, load_done_pairs kommen aus data_io.
# =============================================================================

def _abstain(task: str, reason="EMPTY_BODY") -> Dict[str, Any]:
    if task in {"civility", "sarcasm"}:
        return {"task": task, "label": "ABSTAIN", "confidence": 0.0, "note": reason}
    return {"task": task, "score": "ABSTAIN", "confidence": 0.0, "note": reason}


# =============================================================================
# Writer-Coroutine (einziger Schreiber -> keine Race Conditions)
# =============================================================================

async def writer_task(queue: asyncio.Queue, fp, counter: Dict[str, int]):
    """Liest fertige Records aus der Queue und schreibt sie sequenziell.
    Beendet sich wenn None (Sentinel) empfangen wird."""
    while True:
        rec = await queue.get()
        if rec is None:
            queue.task_done()
            break
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counter["written"] += 1
        if counter["written"] % 200 == 0:
            fp.flush()
        queue.task_done()


# =============================================================================
# Verarbeitung eines Kommentars (alle Tasks)
# =============================================================================

async def process_comment(
    sem: asyncio.Semaphore,
    client, model,
    row: Dict[str, Any],
    tasks: List[str],
    done: Set[Tuple[str, str]],
    queue: asyncio.Queue,
    counter: Dict[str, int],
    id_col, text_col, parent_col,
    all_cols: List[str],
):
    cid = "" if is_empty(row.get(id_col)) else str(row.get(id_col))
    text = "" if is_empty(row.get(text_col)) else str(row.get(text_col))
    parent = None
    if not is_empty(row.get(parent_col)):
        p = str(row.get(parent_col))
        parent = p if p.strip() else None
    meta = {c: jsonable(row.get(c)) for c in all_cols
            if c not in {id_col, text_col, parent_col}}

    # Token-Anzahl als Feld mitschreiben, damit Länge NACHTRÄGLICH filterbar
    # ist (statt über ABSTAIN). ABSTAIN gibt es nur noch bei fehlendem Parent.
    n_tokens = count_tokens(text)

    # Nur noch komplett leerer Text kürzt ab (kein Text -> nichts zu labeln).
    if not cid or not text.strip():
        for task in tasks:
            if (cid, task) in done:
                counter["skipped"] += 1
                continue
            await queue.put({
                "comment_id": cid, "body": text, "predecessor": parent,
                "n_tokens": n_tokens, "task": task,
                "result": _abstain(task, reason="EMPTY_BODY"), "arguments": [],
                "arguments_confidence": None, "arguments_error": "EMPTY_BODY",
                "meta": meta,
            })
        counter["comments_done"] += 1
        return

    # Kontextabhängige Tasks OHNE Parent bekommen ABSTAIN (fehlende Grundlage).
    # Alle anderen Tasks - unabhängig von der Länge - gehen ans Modell.
    CONTEXT_TASKS = {"responsiveness", "agreement"}
    no_parent = parent is None
    llm_tasks = []
    for task in tasks:
        if (cid, task) in done:
            continue
        if no_parent and task in CONTEXT_TASKS:
            await queue.put({
                "comment_id": cid, "body": text, "predecessor": parent,
                "n_tokens": n_tokens, "task": task,
                "result": _abstain(task, reason="NO_PARENT"),
                "arguments": [], "arguments_confidence": None,
                "arguments_error": "NO_PARENT", "meta": meta,
            })
            counter["written_context_abstain"] = counter.get("written_context_abstain", 0) + 1
        else:
            llm_tasks.append(task)

    if not llm_tasks:
        counter["comments_done"] += 1
        return

    async with sem:
        arg_res = await extract_arguments(client, model, text, parent)
        if "error" in arg_res:
            arguments, arg_conf, arg_err = [], None, arg_res["error"]
        else:
            arguments = arg_res.get("arguments", [])
            arg_conf = arg_res.get("confidence")
            arg_err = None

        results = await asyncio.gather(
            *[annotate_one(client, model, t, text, parent) for t in llm_tasks]
        )
        for task, result in zip(llm_tasks, results):
            if "error" in result:
                counter["errors"] += 1
            await queue.put({
                "comment_id": cid, "body": text, "predecessor": parent,
                "n_tokens": n_tokens, "task": task,
                "result": result, "arguments": arguments,
                "arguments_confidence": arg_conf, "arguments_error": arg_err,
                "meta": meta,
            })

    counter["comments_done"] += 1


# =============================================================================
# Progress-Reporter
# =============================================================================

async def progress_reporter(counter: Dict[str, int], total: int, start: float):
    """Gibt regelmäßig Durchsatz und ETA aus."""
    while not counter.get("finished"):
        await asyncio.sleep(15)
        elapsed = time.time() - start
        written = counter["written"]
        rate = written / elapsed if elapsed > 0 else 0
        cd = counter["comments_done"]
        eta_min = ((total - cd) / (cd / elapsed) / 60) if cd > 0 else 0
        print(
            f"  [{cd}/{total} Kommentare] "
            f"written={written} skipped={counter['skipped']} "
            f"errors={counter['errors']} | "
            f"{rate:.1f} calls/s | ETA ~{eta_min:.0f} min",
            flush=True,
        )


# =============================================================================
# Server-Wartelogik
# =============================================================================

async def wait_for_server(base_url: str, timeout: int) -> bool:
    """Wartet bis der vLLM-Server antwortet. Nutzt synchrones requests in
    einem Thread-Executor, um keine zusätzliche async-HTTP-Abhängigkeit
    (aiohttp) zu brauchen - der Health-Check läuft nur einmal am Anfang."""
    import requests
    url = f"{base_url}/health"
    start = time.time()
    loop = asyncio.get_event_loop()
    print(f"Warte auf Server unter {url} ...", flush=True)

    def _check() -> bool:
        try:
            return requests.get(url, timeout=5).status_code == 200
        except requests.RequestException:
            return False

    while time.time() - start < timeout:
        if await loop.run_in_executor(None, _check):
            print(f"Server bereit nach {int(time.time()-start)}s.", flush=True)
            return True
        await asyncio.sleep(5)
    return False


# =============================================================================
# Main
# =============================================================================

async def main_async(args):
    print("=" * 60, flush=True)
    print("HPC LABELING PIPELINE (async)", flush=True)
    print(f"  Input:       {args.input}", flush=True)
    print(f"  Output:      {args.output}", flush=True)
    print(f"  Server:      {args.base_url}  (model: {args.model})", flush=True)
    print(f"  Concurrency: {args.concurrency}", flush=True)
    print(f"  Tasks:       {args.tasks}", flush=True)
    print("=" * 60, flush=True)

    # 1. Auf Server warten
    if not await wait_for_server(args.base_url, args.wait_server):
        print("FEHLER: Server nicht erreichbar.", flush=True)
        return 1

    # 2. Daten laden (data_io gibt (records, columns) zurück)
    rows, all_cols = load_input(args.input)
    if args.id_col not in all_cols or args.text_col not in all_cols:
        print(f"FEHLER: Spalten fehlen. Vorhanden: {all_cols}", flush=True)
        return 1
    total = len(rows)
    print(f"  {total} Zeilen geladen.", flush=True)

    # 3. Wiederaufnahme
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_pairs(args.output) if not args.no_skip else set()
    if done:
        print(f"  Wiederaufnahme: {len(done)} (comment_id,task)-Paare bereits erledigt.", flush=True)

    # 4. Async-Infrastruktur
    client = AsyncOpenAI(base_url=f"{args.base_url}/v1", api_key="dummy",
                         timeout=TIMEOUT_SECONDS, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    counter = {"written": 0, "skipped": 0, "errors": 0,
               "comments_done": 0, "finished": False}

    start = time.time()

    with open(args.output, "a", encoding="utf-8") as fp:
        writer = asyncio.create_task(writer_task(queue, fp, counter))
        reporter = asyncio.create_task(progress_reporter(counter, total, start))

        # Kommentare in Chunks verarbeiten, damit nicht Hunderttausende
        # Coroutine-Objekte gleichzeitig im Speicher liegen. Die Semaphore
        # begrenzt die echte Parallelität INNERHALB und ÜBER Chunks hinweg.
        # Chunk-Größe großzügig über der Concurrency, damit die Semaphore
        # immer ausgelastet bleibt (kein Leerlauf an Chunk-Grenzen).
        CHUNK = max(args.concurrency * 10, 500)
        for chunk_start in range(0, total, CHUNK):
            chunk = rows[chunk_start:chunk_start + CHUNK]
            workers = [
                process_comment(
                    sem, client, args.model, row, args.tasks, done, queue, counter,
                    args.id_col, args.text_col, args.parent_col, all_cols,
                )
                for row in chunk
            ]
            await asyncio.gather(*workers)

        # Writer sauber beenden
        await queue.put(None)
        await writer
        counter["finished"] = True
        reporter.cancel()

    await client.close()

    elapsed = time.time() - start
    rate = counter["written"] / elapsed if elapsed > 0 else 0
    print("=" * 60, flush=True)
    print(f"FERTIG in {elapsed/60:.1f} min", flush=True)
    print(f"  written={counter['written']} skipped={counter['skipped']} "
          f"errors={counter['errors']}", flush=True)
    print(f"  Durchschnitt: {rate:.1f} calls/s", flush=True)
    print("=" * 60, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="HPC Labeling Pipeline (async)")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="Anzahl gleichzeitiger Requests (Semaphore)")
    ap.add_argument("--id-col", default="comment_id")
    ap.add_argument("--text-col", default="body")
    ap.add_argument("--parent-col", default="predecessor")
    ap.add_argument("--tasks", nargs="*", default=list(TASK_SPECS.keys()))
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--wait-server", type=int, default=900)
    args = ap.parse_args()

    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()

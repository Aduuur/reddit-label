"""
label_rationale.py  —  TEST-VARIANTE mit Modell-Begründung
==========================================================
Wie die normale serielle Pipeline, ABER: Das Modell begründet zuerst in EINEM
Satz, welche Textstelle ausschlaggebend war, und gibt DANN das JSON aus
("erst denken, dann antworten" / Chain-of-Thought). Die Begründung wird als
Feld `rationale` mit in die Ausgabe geschrieben.

Nur für kleine Test-Datensätze gedacht (seriell, ein Kommentar nach dem
anderen). Die Produktions-Pipelines bleiben unangetastet.

Aufruf (identisch zur normalen Pipeline):
    python3 label_rationale.py --input test.ndjson --output labels_rat.ndjson \
        --base-url http://127.0.0.1:8000 --model llama-70b --wait-server 600

Ausgabe je Zeile: comment_id, task, result{score/label, confidence},
                  rationale, n_tokens, body, predecessor
"""

from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI, APIConnectionError, APIStatusError

# Bausteine aus dem bestehenden Code wiederverwenden
from prompts import TASK_SPECS, build_user_prompt
from data_io import load_input, is_empty, jsonable, count_tokens

MAX_RETRIES = 4
TIMEOUT_SECONDS = 300
CONTEXT_TASKS = {"responsiveness", "agreement"}

ALL_TASKS = ["stance_intensity", "epistemic_modality", "justification_density",
             "responsiveness", "agreement", "civility", "sarcasm"]


# ---------------------------------------------------------------------------
# CoT-System-Prompt: erst ein Satz Begründung, dann JSON
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_COT = """You are a careful, literal discourse annotator. Use ONLY the provided TEXT,
PARENT_TEXT (if given), and TOPIC_DEF. Do not use outside knowledge.

Answer in EXACTLY this two-line format, nothing else:

REASONING: <one short sentence naming the specific word(s) or phrase in the TEXT
that most drove your decision>
JSON: <the JSON object for the task>

Rules for the JSON:
- It must be a single valid JSON object on ONE line, matching the task schema.
- Always include a "confidence" field in [0,1]. Use the full range honestly:
  high only for clear textual evidence, low when you are guessing.
- ALWAYS commit to a value. Do NOT output "ABSTAIN" — a value is always required
  here (the test harness handles the no-parent case separately).
- Do not add any text after the JSON line.

Example (for a civility task):
REASONING: The insult "idiot" is the decisive cue for low civility.
JSON: {"task": "civility", "label": 2, "confidence": 0.83}

Task schemas:
- stance_intensity : {"task":"stance_intensity","score":<int 1-6>,"confidence":<float>}
- epistemic_modality: {"task":"epistemic_modality","score":<float 0-1>,"confidence":<float>}
- justification_density: {"task":"justification_density","score":<float>=0>,"confidence":<float>}
- responsiveness   : {"task":"responsiveness","score":<float 0-1>,"confidence":<float>}
- agreement        : {"task":"agreement","score":<float -1..1>,"confidence":<float>}
- civility         : {"task":"civility","label":<int 1-6>,"confidence":<float>}
- sarcasm          : {"task":"sarcasm","label":<0 or 1>,"confidence":<float>}
"""


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=f"{base_url}/v1", api_key="dummy", timeout=TIMEOUT_SECONDS)


def call_chat(client: OpenAI, model: str, messages, temperature: float = 0.0) -> str:
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=400,   # etwas mehr Platz für die Begründung
        )
        return resp.choices[0].message.content.strip()
    except (APIConnectionError, APIStatusError) as e:
        raise requests.RequestException(f"vLLM connection error: {e}") from e
    except Exception as e:
        raise requests.RequestException(f"vLLM unexpected error: {e}") from e


# ---------------------------------------------------------------------------
# Robustes Parsen: Begründung + letztes JSON-Objekt trennen
# ---------------------------------------------------------------------------
def extract_last_json(text: str) -> Optional[dict]:
    """Findet das LETZTE balancierte {...} im Text und parst es. Robust auch,
    wenn das Modell vor dem JSON noch Prosa schreibt."""
    # von hinten das letzte '}' suchen, dann passendes '{'
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # dieses '}' passt nicht -> nächstes suchen
        end = text.rfind("}", 0, end)
    return None


def extract_reasoning(text: str) -> str:
    """Zieht den Begründungssatz. Bevorzugt die REASONING:-Zeile, sonst alles
    vor dem JSON."""
    m = re.search(r"REASONING:\s*(.+?)(?:\n|JSON:|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: Text vor dem ersten '{'
    brace = text.find("{")
    if brace > 0:
        return text[:brace].replace("REASONING:", "").replace("JSON:", "").strip()
    return ""


def validate(task: str, obj: dict) -> Optional[str]:
    spec = TASK_SPECS[task]
    key = spec.get("score_key") or spec.get("label_key")
    if key not in obj:
        # gleiche Normalisierung wie Produktion: Wert unter anderem Schlüssel?
        for alt in (task, "score", "label", "value"):
            if alt in obj and alt not in ("task", "confidence"):
                obj[key] = obj[alt]
                break
    if key not in obj:
        return f"missing {key}"
    if "confidence" not in obj:
        return "missing confidence"
    return None


def annotate_one(client, model, task, text, parent_text=None):
    last_err = None
    last_raw = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            user = build_user_prompt(task, text, parent_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                {"role": "user", "content": user},
            ]
            raw = call_chat(client, model, messages, temperature=0.0)
            last_raw = raw
            obj = extract_last_json(raw)
            if obj is None:
                last_err = f"nojson({attempt})"
                continue
            err = validate(task, obj)
            if err:
                last_err = f"schema({attempt}): {err}"
                continue
            rationale = extract_reasoning(raw)
            return {"result": obj, "rationale": rationale}
        except requests.RequestException as e:
            last_err = f"conn({attempt}): {e}"
        time.sleep(2 ** min(attempt, 4))
    return {"result": {"task": task, "error": last_err or "unknown"},
            "rationale": "", "raw": (last_raw or "")[:400]}


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    url = f"{base_url}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="llama-70b")
    ap.add_argument("--wait-server", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Kommentare (0=alle)")
    args = ap.parse_args()

    print(f"Warte auf Server {args.base_url} ...", flush=True)
    if not wait_for_server(args.base_url, args.wait_server):
        print("Server nicht erreichbar.", flush=True); sys.exit(1)
    print("Server bereit.", flush=True)

    client = make_client(args.base_url)
    records, columns = load_input(args.input)
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} Kommentare zu labeln (mit Begründung).", flush=True)

    written = 0
    with open(args.output, "w", encoding="utf-8") as fp:
        for idx, rec in enumerate(records):
            cid = rec.get("comment_id")
            text = rec.get("body") or ""
            parent = rec.get("predecessor")
            if is_empty(parent):
                parent = None
            n_tok = count_tokens(text)

            if not cid or not text.strip():
                continue

            no_parent = parent is None
            for task in ALL_TASKS:
                if no_parent and task in CONTEXT_TASKS:
                    out = {"result": {"task": task, "score": "ABSTAIN",
                                      "confidence": 0.1, "note": "NO_PARENT"},
                           "rationale": "kein Eltern-Kommentar vorhanden"}
                else:
                    out = annotate_one(client, args.model, task, text, parent)
                row = {
                    "comment_id": cid, "task": task,
                    "result": jsonable(out["result"]),
                    "rationale": out.get("rationale", ""),
                    "n_tokens": n_tok,
                    "body": text, "predecessor": parent,
                }
                if "raw" in out:
                    row["raw"] = out["raw"]
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            if (idx + 1) % 10 == 0:
                fp.flush()
                print(f"[{idx+1}/{len(records)}] {written} Zeilen", flush=True)

    print(f"FERTIG: {written} Zeilen -> {args.output}", flush=True)


if __name__ == "__main__":
    main()

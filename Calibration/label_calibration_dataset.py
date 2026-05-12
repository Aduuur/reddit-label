from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Callable

from pathlib import Path
import numpy as np
import pandas as pd
import requests
from openai import OpenAI, APIConnectionError, APIStatusError

# =============================================================================
# Configuration
# =============================================================================

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234"
MODEL_NAME = "ibm/granite-3.2-8b"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

# HoreKa vLLM Server (via SSH-Tunnel)
HOREKA_BASE_URL = "http://localhost:8000"
HOREKA_MODEL = "/hkfs/work/workspace/scratch/unoim-llm_models/hf_cache/models/meta-llama/Llama-3.1-70B-Instruct"

# Welches Backend nutzen? "horeka" oder "lmstudio"
BACKEND = "horeka"

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

EXCEL_PATH = Path("../data/sampled_threads.xlsx")

OUTPUT_NDJSON = Path.cwd() / "labels_from_excel.ndjson"


# =============================================================================
# Task specification
# =============================================================================

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "stance_intensity": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "stance_intensity",
        "score_key": "score",
        "score_type": (int, float),
        "confidence_range": (0.0, 1.0),
        "score_range": (1, 6),
        "allow_abstain": True,
        "label_key": None,
    },
    "epistemic_modality": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "epistemic_modality",
        "score_key": "score",
        "score_type": (int, float),
        "confidence_range": (0.0, 1.0),
        "score_range": (0.0, 1.0),
        "allow_abstain": True,
        "label_key": None,
    },
    "justification_density": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "justification_density",
        "score_key": "score",
        "score_type": (int, float),
        "confidence_range": (0.0, 1.0),
        "score_range": (0.0, float("inf")),
        "allow_abstain": True,
        "label_key": None,
    },
    "responsiveness": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "responsiveness",
        "score_key": "score",
        "score_type": (int, float),
        "confidence_range": (0.0, 1.0),
        "score_range": (0.0, 1.0),
        "allow_abstain": True,
        "label_key": None,
    },
    "agreement": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "agreement",
        "score_key": "score",
        "score_type": (int, float),
        "confidence_range": (0.0, 1.0),
        "score_range": (-1.0, 1.0),
        "allow_abstain": True,
        "label_key": None,
    },
    "civility": {
        "schema_keys": {"task", "label", "confidence"},
        "task_value": "civility",
        "score_key": None,
        "label_key": "label",
        "label_type": (int,),
        "score_range": (1.0, 6.0),
        "confidence_range": (0.0, 1.0),
        "allow_abstain": True,
    },
    "sarcasm": {
        "schema_keys": {"task", "label", "confidence"},
        "task_value": "sarcasm",
        "score_key": None,
        "label_key": "label",
        "label_type": (int,),
        "score_range": (0.0, 1.0),
        "confidence_range": (0.0, 1.0),
        "allow_abstain": True,
    },
}


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are a careful, literal discourse annotator. Use ONLY the provided TEXT,
PARENT_TEXT (if given), and TOPIC_DEF. Do not infer beyond the text or use any external knowledge.
Output must be strictly valid JSON matching the registered schema for the task. Include a "confidence" field in [0,1].

You MUST prefer returning "ABSTAIN" over guessing. If the evidence for a numeric/ordinal label is weak,
conflicting, or the text is very short, return "ABSTAIN" for that task.

English only. If TEXT has fewer than ~15 tokens, or evidence is weak/ambiguous, return "ABSTAIN"
instead of trying to infer a score/label. When you return "ABSTAIN", still provide a confidence score
in [0,1] reflecting your confidence that abstaining is appropriate (typically low, e.g., <= 0.4).

B.2 Instruction templates and schemas
All tasks share the same idea: return a single JSON object for the requested task.

IMPORTANT:
- For each task, the main numeric/ordinal field ("score" or "label") can either be:
  - a valid number in the specified range, OR
  - the string "ABSTAIN" (if you cannot reliably decide).
- Never output any other strings in place of numeric scores/labels.

--------------------------------------------------
Stance intensity (ordinal; [1,6])
--------------------------------------------------
TASK: Extract the topic of this comment and then based on the topic the author's stance
on a 1-6 scale on this topic (1 = strongly against, 3-4 = neutral/unclear, 6 = strongly in favor).

Use "ABSTAIN" if:
- there is no clear evidence of stance toward the topic, or
- the text is very short or off-topic, or
- signals are too mixed/ambiguous to place on the scale.

JSON schema:
{
  "task": "stance_intensity",
  "score": <int in [1, 6] OR "ABSTAIN">,
  "confidence": <float in [0,1]>
}

--------------------------------------------------
Epistemic modality (continuous; [0,1])
--------------------------------------------------
TASK: Score epistemic modality: density of hedges, modal verbs, and explicit uncertainty,
normalized by length. Higher = more expressed humility/uncertainty.

If you cannot reliably judge (e.g., very short or unclear text), return "ABSTAIN".

JSON schema:
{
  "task": "epistemic_modality",
  "score": <float in [0.0,1.0] OR "ABSTAIN">,
  "confidence": <float in [0,1]>
}

--------------------------------------------------
Justification density (per 100 words)
--------------------------------------------------
TASK: Count distinct justification units (claim + warrant), normalized per 100 words.
If uncertain, or if there is not enough content to identify justification units, return "ABSTAIN".

JSON schema:
{
  "task": "justification_density",
  "score": <non-negative float OR "ABSTAIN">,
  "confidence": <float in [0,1]>
}

--------------------------------------------------
Responsiveness (continuous; [0,1])
--------------------------------------------------
TASK: Rate how directly this reply addresses its {PARENT_TEXT} (semantic overlap/engagement).

If PARENT_TEXT is not provided, or if the relation between TEXT and PARENT_TEXT is unclear,
return "ABSTAIN".

JSON schema:
{
  "task": "responsiveness",
  "score": <float in [0.0,1.0] OR "ABSTAIN">,
  "confidence": <float in [0,1]>
}

--------------------------------------------------
Agreement (continuous; [-1,1])
--------------------------------------------------
TASK: Rate agreement with {PARENT_TEXT}.
-1 = contradicts, 0 = neutral/unrelated, 1 = fully agrees.

If PARENT_TEXT is not provided, or if agreement cannot be reliably determined (e.g., off-topic,
sarcastic or ambiguous content without clear polarity), return "ABSTAIN".

JSON schema:
{
  "task": "agreement",
  "score": <float in [-1.0,1.0] OR "ABSTAIN">,
  "confidence": <float in [0,1]>
}

--------------------------------------------------
Civility (ordinal 1-6)
--------------------------------------------------
TASK: Rate civility on 1 (highly uncivil/insulting) to 6 (highly civil).

If civility is hard to judge (e.g., context missing, mixed cues, or text too short),
return "ABSTAIN".

JSON schema:
{
  "task": "civility",
  "label": <integer 1..6 OR "ABSTAIN">,
  "confidence": <float in [0,1]>
}

--------------------------------------------------
Sarcasm (binary 0/1)
--------------------------------------------------
TASK: Detect whether the TEXT is sarcastic or clearly ironic regarding its main topic.

- Extract the topic of this comment.
- Use 1 if there is clear sarcastic or ironic intent (e.g., praise used to convey criticism,
  exaggerated contrast between words and obvious reality, well-known sarcastic formulae).
- Use 0 if the text is clearly non-sarcastic and literal.
- If signals are ambiguous, context is insufficient, or the text is too short to decide,
  return "ABSTAIN".

JSON schema:
{
  "task": "sarcasm",
  "label": <0 or 1 OR "ABSTAIN">,
  "confidence": <float in [0,1]>
}
"""

TASK_INSTRUCTION_TEMPLATE = (
    "Now perform ONLY the task = {task_name}. "
    "Return strictly valid JSON for that task and nothing else (no Markdown). "
    "Ensure keys and value ranges match the schema exactly."
)

ARG_SYSTEM_PROMPT = """
You are an argument mining annotator.
Use ONLY the provided TEXT and PARENT_TEXT (if given). Do not infer beyond the text.

Your task: extract explicit argumentative units from TEXT.
Each argument must consist of:
- a 'claim' (the central assertion), and
- a 'proof' (the supporting reason/evidence),
both as short strings.

Output strictly valid JSON:
{
  "task": "argument_extraction",
  "arguments": [
      { "claim": "<string>", "proof": "<string>" },
      { "claim": "<string>", "proof": "<string>" }
  ],
  "confidence": <float in [0,1]>
}

Guidelines:
- "arguments" must be a list of objects, each with keys "claim" and "proof".
- If TEXT contains no clear arguments, use an empty list [].
- Claims and proofs must be grounded in the text; do NOT hallucinate.
- Claims and proofs should be concise summaries or minimal spans.
- Prefer under-detection (few arguments) over over-detection.
"""

ARG_USER_TEMPLATE = (
    "TEXT: {text}\n\n"
    "If available, PARENT_TEXT (context for replies): {parent_text}\n\n"
    "Now perform ONLY argument_extraction as described in the system prompt. "
    "Return strictly valid JSON and nothing else."
)


# =============================================================================
# Excel loading / cleaning
# =============================================================================

def _clean_column_name(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("[", " ").replace("]", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = s.replace(",", "_").replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s.strip("_").lower()


def _deduplicate_columns(cols: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        base = c if c else "unnamed"
        counts[base] = counts.get(base, 0) + 1
        out.append(base if counts[base] == 1 else f"{base}__{counts[base]}")
    return out


def _looks_like_two_row_header(df_raw: pd.DataFrame) -> bool:
    if len(df_raw) < 2:
        return False
    row0 = [str(x).strip().lower() for x in df_raw.iloc[0].tolist()]
    row1 = [str(x).strip().lower() for x in df_raw.iloc[1].tolist()]
    row1_has_core = any(x in {"comment_id", "body", "predecessor"} for x in row1)
    row0_has_groups = any(x in {"arthur", "veronika"} for x in row0)
    return row1_has_core or row0_has_groups


def load_excel_comments(
        excel_path: str | Path,
        sheet_name: Optional[str] = 0,
) -> pd.DataFrame:
    excel_path = str(excel_path)
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    if raw.empty:
        raise ValueError("Excel file is empty.")

    if _looks_like_two_row_header(raw):
        top = raw.iloc[0].tolist()
        bottom = raw.iloc[1].tolist()
        new_cols: List[str] = []
        for t, b in zip(top, bottom):
            t_clean = _clean_column_name(t)
            b_clean = _clean_column_name(b)
            if b_clean in {"comment_id", "best_topic_index", "body", "predecessor"}:
                new_cols.append(b_clean)
            elif t_clean in {"arthur", "veronika"} and b_clean:
                new_cols.append(f"{t_clean}__{b_clean}")
            elif b_clean:
                new_cols.append(b_clean)
            elif t_clean:
                new_cols.append(t_clean)
            else:
                new_cols.append("unnamed")
        new_cols = _deduplicate_columns(new_cols)
        df = raw.iloc[2:].copy()
        df.columns = new_cols
        df = df.reset_index(drop=True)
    else:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df.columns = _deduplicate_columns([_clean_column_name(c) for c in df.columns])

    required = {"comment_id", "body", "predecessor"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after parsing: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    df["comment_id"] = df["comment_id"].astype("string")
    df["body"] = df["body"].astype("string")
    df["predecessor"] = df["predecessor"].astype("string")

    return df


# =============================================================================
# HTTP utilities
# =============================================================================

def call_lmstudio_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """LM Studio backend."""
    url = f"{LMSTUDIO_BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_horeka_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """
    HoreKa vLLM backend via SSH-Tunnel.
    Wirft requests.RequestException bei Verbindungsfehlern,
    damit die Retry-Logik in annotate_one / extract_arguments greift.
    """
    try:
        client = OpenAI(
            base_url=f"{HOREKA_BASE_URL}/v1",
            api_key="dummy",
            timeout=120.0,
        )
        resp = client.chat.completions.create(
            model=HOREKA_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except (APIConnectionError, APIStatusError) as e:
        # In requests.RequestException umwandeln damit Retry-Logik greift
        raise requests.RequestException(f"HoreKa connection error: {e}") from e
    except Exception as e:
        raise requests.RequestException(f"HoreKa unexpected error: {e}") from e


def call_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """
    Zentraler Dispatcher: nutzt BACKEND-Variable.
    BACKEND = "horeka"    -> HoreKa vLLM
    BACKEND = "lmstudio"  -> LM Studio lokal
    """
    if BACKEND == "horeka":
        return call_horeka_chat(messages, temperature)
    return call_lmstudio_chat(messages, temperature)


# =============================================================================
# Validation
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
        label_type = spec.get("label_type", (int,))
        if not isinstance(val, label_type):
            return f'"{label_key}" has wrong type (expected {label_type})'
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
        return "Missing required keys: 'arguments' and/or 'confidence'"
    args = obj["arguments"]
    if not isinstance(args, list):
        return '"arguments" must be a list'
    for i, a in enumerate(args):
        if not isinstance(a, dict):
            return f'"arguments[{i}]" must be an object with keys "claim" and "proof"'
        if "claim" not in a or "proof" not in a:
            return f'"arguments[{i}]" must contain keys "claim" and "proof"'
        if not isinstance(a["claim"], str):
            return f'"arguments[{i}].claim" must be a string'
        if not isinstance(a["proof"], str):
            return f'"arguments[{i}].proof" must be a string'
    conf = obj["confidence"]
    if not _is_number(conf):
        return '"confidence" must be a number'
    if not (0.0 <= conf <= 1.0):
        return '"confidence" must be in [0,1]'
    return None


# =============================================================================
# Prompt construction
# =============================================================================

def build_user_prompt(task: str, text: str, parent_text: Optional[str] = None) -> str:
    parts: Dict[str, Any] = {"TEXT": text}
    if parent_text is not None and str(parent_text).strip():
        parts["PARENT_TEXT"] = str(parent_text)
    directive = TASK_INSTRUCTION_TEMPLATE.format(task_name=task)
    return json.dumps(parts, ensure_ascii=False) + "\n\n" + directive


def build_argument_prompt(text: str, parent_text: Optional[str] = None) -> str:
    pt = parent_text if (parent_text is not None and str(parent_text).strip()) else "N/A"
    return ARG_USER_TEMPLATE.format(text=text, parent_text=pt)


# =============================================================================
# Annotation calls with retries
# =============================================================================

def annotate_one(task: str, text: str, parent_text: Optional[str] = None) -> Dict[str, Any]:
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task, text, parent_text)},
            ]
            raw = call_chat(messages, temperature=0.0)
            obj = json.loads(raw)
            err = validate_response(task, obj)
            if err is None:
                return obj
            last_err = f"Schema validation failed (attempt {attempt}): {err}"

        except requests.RequestException as e:
            last_err = f"HTTP error (attempt {attempt}): {e}"
        except json.JSONDecodeError as e:
            last_err = f"JSON parse error (attempt {attempt}): {e}"

        time.sleep(RETRY_BACKOFF_SECONDS ** attempt)

    return {"task": task, "error": last_err or "Unknown error"}


def extract_arguments(text: str, parent_text: Optional[str] = None) -> Dict[str, Any]:
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [
                {"role": "system", "content": ARG_SYSTEM_PROMPT},
                {"role": "user", "content": build_argument_prompt(text, parent_text)},
            ]
            raw = call_chat(messages, temperature=0.0)
            obj = json.loads(raw)
            err = validate_arguments(obj)
            if err is None:
                return obj
            last_err = f"Argument schema validation failed (attempt {attempt}): {err}"

        except requests.RequestException as e:
            last_err = f"HTTP error (attempt {attempt}): {e}"
        except json.JSONDecodeError as e:
            last_err = f"JSON parse error (attempt {attempt}): {e}"

        time.sleep(RETRY_BACKOFF_SECONDS ** attempt)

    return {"task": "argument_extraction", "error": last_err or "Unknown error"}


# =============================================================================
# Output helpers
# =============================================================================

def write_ndjson_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_done_pairs(ndjson_path: str) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(ndjson_path):
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


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _make_abstain_result(task: str, reason: str = "EMPTY_BODY") -> Dict[str, Any]:
    if task in {"civility", "sarcasm"}:
        return {"task": task, "label": "ABSTAIN", "confidence": 0.0, "note": reason}
    return {"task": task, "score": "ABSTAIN", "confidence": 0.0, "note": reason}


# =============================================================================
# Main pipeline
# =============================================================================

def label_excel_dataframe(
        df: pd.DataFrame,
        tasks: Optional[List[str]] = None,
        ndjson_path: str = "labels_from_excel.ndjson",
        id_col: str = "comment_id",
        text_col: str = "body",
        predecessor_col: str = "predecessor",
        batch_size: int = 500,
        start_index: int = 0,
        end_index: Optional[int] = None,
        skip_existing: bool = True,
        progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    assert id_col in df.columns, f"Column '{id_col}' missing."
    assert text_col in df.columns, f"Column '{text_col}' missing."
    assert predecessor_col in df.columns, f"Column '{predecessor_col}' missing."

    if tasks is None:
        tasks = list(TASK_SPECS.keys())

    n_total = len(df)
    if end_index is None or end_index > n_total:
        end_index = n_total
    if start_index < 0:
        start_index = 0
    if start_index >= end_index:
        return {"processed_rows": 0, "written_records": 0,
                "skipped_records": 0, "errors": 0,
                "note": "Nothing to process (start_index >= end_index)."}

    already_done: Set[Tuple[str, str]] = set()
    if skip_existing:
        already_done = _load_done_pairs(ndjson_path)

    written_records = 0
    skipped_records = 0
    error_count = 0
    processed_rows = 0

    with open(ndjson_path, "a", encoding="utf-8") as fp:
        for batch_start in range(start_index, end_index, batch_size):
            batch_end = min(batch_start + batch_size, end_index)
            batch = df.iloc[batch_start:batch_end]

            for local_idx, row in batch.iterrows():
                comment_id = str(row[id_col]) if pd.notna(row[id_col]) else ""
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                predecessor_text = (
                    str(row[predecessor_col]) if pd.notna(row[predecessor_col]) else None
                )
                if predecessor_text is not None and not predecessor_text.strip():
                    predecessor_text = None

                meta = {
                    c: (_jsonable(row[c]) if c in row.index else None)
                    for c in df.columns
                    if c not in {id_col, text_col, predecessor_col}
                }

                # Empty body -> ABSTAIN ohne LLM-Aufruf
                if (not comment_id) or (not text) or (not text.strip()):
                    arguments, arg_conf, arg_err = [], None, "EMPTY_BODY"
                    for task in tasks:
                        if skip_existing and (comment_id, task) in already_done:
                            skipped_records += 1
                            continue
                        out = {
                            "comment_id": comment_id,
                            "body": text,
                            "predecessor": predecessor_text,
                            "comment_index": int(local_idx),
                            "task": task,
                            "result": _make_abstain_result(task, reason="EMPTY_BODY"),
                            "arguments": arguments,
                            "arguments_confidence": arg_conf,
                            "arguments_error": arg_err,
                            "meta": meta,
                        }
                        write_ndjson_line(fp, out)
                        written_records += 1
                        if written_records % 100 == 0:
                            fp.flush()
                    processed_rows += 1
                    if progress:
                        progress(processed_rows, end_index - start_index,
                                 {"written_records": written_records,
                                  "skipped_records": skipped_records,
                                  "errors": error_count})
                    continue

                # Argument extraction
                arg_res = extract_arguments(text, parent_text=predecessor_text)
                if "error" in arg_res:
                    arguments, arg_conf, arg_err = [], None, arg_res["error"]
                else:
                    arguments = arg_res.get("arguments", [])
                    arg_conf = arg_res.get("confidence", None)
                    arg_err = None

                # Labeling tasks
                for task in tasks:
                    if skip_existing and (comment_id, task) in already_done:
                        skipped_records += 1
                        continue
                    try:
                        result = annotate_one(task, text, parent_text=predecessor_text)
                        out = {
                            "comment_id": comment_id,
                            "body": text,
                            "predecessor": predecessor_text,
                            "comment_index": int(local_idx),
                            "task": task,
                            "result": result,
                            "arguments": arguments,
                            "arguments_confidence": arg_conf,
                            "arguments_error": arg_err,
                            "meta": meta,
                        }
                        write_ndjson_line(fp, out)
                        written_records += 1
                        if written_records % 100 == 0:
                            fp.flush()
                    except Exception:
                        error_count += 1

                processed_rows += 1
                if progress:
                    progress(processed_rows, end_index - start_index,
                             {"written_records": written_records,
                              "skipped_records": skipped_records,
                              "errors": error_count})

    return {
        "processed_rows": processed_rows,
        "written_records": written_records,
        "skipped_records": skipped_records,
        "errors": error_count,
    }


def label_excel_file(
        excel_path: str | Path,
        tasks: Optional[List[str]] = None,
        ndjson_path: str = "labels_from_excel.ndjson",
        sheet_name: Optional[str] = 0,
        id_col: str = "comment_id",
        text_col: str = "body",
        predecessor_col: str = "predecessor",
        batch_size: int = 500,
        start_index: int = 0,
        end_index: Optional[int] = None,
        skip_existing: bool = True,
        progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    df = load_excel_comments(excel_path, sheet_name=sheet_name)
    return label_excel_dataframe(
        df=df, tasks=tasks, ndjson_path=ndjson_path,
        id_col=id_col, text_col=text_col, predecessor_col=predecessor_col,
        batch_size=batch_size, start_index=start_index, end_index=end_index,
        skip_existing=skip_existing, progress=progress,
    )


# =============================================================================
# Demo run
# =============================================================================

if __name__ == "__main__":
    print(f"Backend: {BACKEND}")

    tasks = [
        "stance_intensity", "civility", "epistemic_modality",
        "justification_density", "responsiveness", "agreement", "sarcasm",
    ]

    def simple_progress(done: int, total: int, stats: Dict[str, Any]) -> None:
        if done % 100 == 0 or done == total:
            print(f"[{done}/{total}] written={stats['written_records']} "
                  f"skipped={stats['skipped_records']} errors={stats['errors']}")

    df_preview = load_excel_comments(EXCEL_PATH)
    print("Loaded Excel successfully.")
    print("Columns:", list(df_preview.columns))
    print(df_preview.head())

    stats = label_excel_file(
        excel_path=EXCEL_PATH,
        tasks=tasks,
        ndjson_path=str(OUTPUT_NDJSON),
        sheet_name=0,
        id_col="comment_id",
        text_col="body",
        predecessor_col="predecessor",
        batch_size=100,
        start_index=0,
        end_index=None,
        skip_existing=True,
        progress=simple_progress,
    )

    print("Labeling finished.")
    print(stats)
#%%

#%%

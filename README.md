# reddit-label
# Reddit Sampling Pipeline (Posts Only)

This repo contains a lightweight sampling pipeline for **Reddit submissions (posts)**.  
It **does not process comments** — comments must be matched/collected in a **second step**.

---

## Data layout

- Put your input files into a folder named **`data/`**
- Expected input format: **JSON Lines** (`.jsonl`)
- Expected filename pattern: **`data/*_posts.jsonl`**

The loader is robust against malformed JSON rows and will skip invalid lines while logging a short preview.

---

## Covered subreddits

This pipeline was used on:

- r/changemyview
- r/AskReddit
- r/TrueReddit
- r/AmItheAsshole
- r/debate
- r/PoliticalDiscussion
- r/liberal
- r/conservative
- r/moderatepolitics
- r/worldnews
- r/socialjustice

---

## What the pipeline does

### 1) Load submissions (robust JSONL ingestion)

- Reads one or multiple `*.jsonl` files line-by-line
- Keeps only the following columns:

`author, created_utc, downs, id, likes, num_comments, ups, selftext, title, subreddit`

- Converts `created_utc` (Unix timestamp) to **UTC datetime**
- Ensures a consistent schema across all loaded files
- Prints a load summary:
    - processed files
    - failed files
    - skipped malformed lines
    - total rows loaded

---

### 2) Filter submissions by topic similarity (title-based)

The pipeline filters posts by computing cosine similarity between each post **title** and a list of predefined **topic sentences**.

- Immediately drops titles that are `"[removed]"` or `"[deleted]"`
- Computes similarity and assigns the best-matching topic per post
- Adds metadata columns:
    - `best_topic_index`
    - `best_topic`
    - `best_topic_similarity`
- Keeps only rows where:

`best_topic_similarity >= similarity_threshold` (example default: `0.4`)

---

### 3) Similarity backend (SBERT with TF-IDF fallback)

- **Preferred:** SBERT embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- **Fallback:** TF-IDF vectors (if SBERT dependencies are not installed)

Both options use cosine similarity for scoring.

---

## Quick start

1. Place your input files under `data/`, e.g.:
    - `data/changemyview_posts.jsonl`
    - `data/AskReddit_posts.jsonl`
    - ...

2. Run the script (example behavior):
    - Collect all files via `glob("data/*_posts.jsonl")`
    - Load them into a single dataframe
    - Filter by topic similarity with a threshold (e.g. `0.4`)

3. Adjust as needed:
    - `SIMILARITY_THRESHOLD`
    - `DEFAULT_TOPICS` (topic sentence list)

---

## Notes / Scope

- **Posts only**: this pipeline samples and filters **submissions**.
- **Comments are out of scope** here and must be retrieved/matched in a separate step.




## Reddit Comment Labeling Pipeline (LM Studio, OpenAI-Compatible API)

This script implements a schema-validated labeling pipeline for Reddit comments using **LM Studio’s local HTTP server** (OpenAI-compatible Chat Completions API). It supports both a **generic input mode** (no parent context) and a **DataFrame mode** (parent lookup + metadata passthrough + resume support). Output is written as **NDJSON** (one JSON object per line).

---

### Key Features

- **Flexible input formats**
    - Reddit-style dictionaries with keys `id` and `body`
    - A mapping `id -> text`
    - A `pandas.DataFrame` with `['id','body']` or `['comment_id','body']`

- **Robust execution**
    - **Retries**: up to `MAX_RETRIES` attempts per `(comment, task)` with exponential backoff
    - **Strict JSON parsing** (`json.loads`) for every model response
    - **Strict schema validation** per task (required keys, types, ranges, and `"ABSTAIN"` handling)

- **Argument extraction extension**
    - Runs **once per comment** and attaches results to every `(comment, task)` output record

- **Parent context support (DataFrame mode)**
    - Stores `parent_id` in the NDJSON output
    - Builds `parent_text` using:
        1. `parent_id -> body` lookup (reply-to-comment), else
        2. submission context via `submission_title` + `submission_selftext` (if present)

- **Resume / skip existing records**
    - If `skip_existing=True`, the script reads the existing NDJSON file and skips already-processed `(comment_id, task)` pairs.

- **Empty-body handling**
    - If a comment body is empty/blank: no LLM calls are made.
    - The script emits deterministic per-task results with `"ABSTAIN"` and `confidence=0.0`.

---

### Implemented Labeling Tasks

Task schemas are defined in `TASK_SPECS` and enforced via strict validation.

**Score-based tasks** (field: `score`)
- `stance_intensity` (ordinal: `1..6` or `"ABSTAIN"`)
- `epistemic_modality` (continuous: `0..1` or `"ABSTAIN"`)
- `justification_density` (non-negative float or `"ABSTAIN"`)
- `responsiveness` (continuous: `0..1` or `"ABSTAIN"`; requires parent context)
- `agreement` (continuous: `-1..1` or `"ABSTAIN"`; requires parent context)

**Label-based tasks** (field: `label`)
- `civility` (ordinal: `1..6` or `"ABSTAIN"`)
- `sarcasm` (binary: `0/1` or `"ABSTAIN"`)

All tasks must include:
- `confidence` in `[0, 1]`

---

### Argument Extraction Extension

In addition to the labeling tasks, the pipeline runs a dedicated **argument mining** call once per comment and attaches the results to every output record.

Expected argument extraction schema:

```json
{
  "task": "argument_extraction",
  "arguments": [
    { "claim": "<string>", "proof": "<string>" }
  ],
  "confidence": 0.0
}
```
### Failure Behavior (Argument Extraction)

If argument extraction fails or the returned JSON is invalid, the pipeline writes:

- `arguments: []`
- `arguments_confidence: null`
- `arguments_error: "<error message>"`

---

### Output Format (NDJSON)

The output file is **NDJSON** (newline-delimited JSON). Each line corresponds to one `(comment_id, task)` pair.

**Typical record structure**
```json
{
  "comment_id": "<string>",
  "body": "<string>",
  "parent_id": "<string or null>",
  "comment_index": "<int>",
  "task": "<task_name>",
  "result": { "... validated task result ..." },
  "arguments": [ { "claim": "...", "proof": "..." } ],
  "arguments_confidence": "<float or null>",
  "arguments_error": "<string or null>",
  "meta": { "... other DataFrame columns ..." }
}
```

**Notes**
- In **generic mode** (`run_pipeline`), fields like `parent_id`, `submission_id`, `created_utc`, `user`, and `depth` are `null` unless provided by the input.
- In **DataFrame mode** (`label_dataframe`), all additional DataFrame columns (except `id_col` and `text_col`) are included under `meta` (JSON-safe conversion is applied).

---

### Requirements

Install dependencies:
```bash
pip install requests pandas numpy
```

## Pipeline 2: Thread-Based Metrics (LM Studio, OpenAI-Compatible API)

This script is the **second stage** of the Reddit labeling workflow. It consumes the **NDJSON output from Pipeline 1** (which already contains per-comment `arguments`, `parent_id`, and submission identifiers) and computes **thread-level discourse metrics** for each comment using the **direct ancestor chain** (parent → parent → …) back to the thread root.

### What Pipeline 2 Computes

Pipeline 2 adds two **thread-level metrics**:

- **`argument_novelty`** (float in `[0, 1]`)
  - Measures how *novel* the current comment’s extracted arguments are compared to the arguments in its ancestor chain.
  - `0.0` = no new arguments, fully reused
  - `1.0` = entirely new arguments

- **`semantic_entropy`** (float `>= 0`, no strict upper bound)
  - Approximates the semantic diversity of arguments in the thread up to and including the current comment.
  - Low values ≈ one dominant theme; higher values ≈ multiple balanced themes.

Both tasks support `"ABSTAIN"` when the evidence is too weak (e.g., no arguments).

---

## Thread Context Definition (Direct Parent Chain Only)

For every comment, the thread context is built by walking **only the direct parent chain** upward:

- Follow `parent_id` → parent comment → its `parent_id` → …  
- Stop when any of the following holds:
  - `parent_id` is `None`
  - the parent is a submission root (`t3_...`)
  - `parent_id == link_id` / `submission_id` (canonicalized)
  - the chain is broken (parent not found)
  - a cycle is detected (safety stop)

**Important:** Pipeline 2 does **not** use siblings, children, or the full tree—only the direct lineage.

The resulting `HISTORY_ARGUMENTS` is ordered **from earliest ancestor to immediate parent**.

---

## Input

**Input file:** NDJSON created by Pipeline 1 (multiple lines per `comment_id` are expected).

Pipeline 2:
- reads all NDJSON lines (`records`) unchanged
- builds a `comment_map` keyed by canonical comment id
- selects a “best” base record per comment (heuristic: more arguments wins; else first)

Required (best-effort) fields in the Pipeline 1 output:
- `comment_id` (or `id`)
- `parent_id` (or `parent`)
- `arguments` (list)
- submission/link identifier in one of:
  - `link_id`
  - `submission_id`
  - `meta.submission_id`
  - `meta.thread_id`

All additional fields are preserved.

---

## Model Interface (LM Studio)

Pipeline 2 uses LM Studio’s OpenAI-compatible endpoint:

- `POST /v1/chat/completions`
- controlled via:
  - `LMSTUDIO_BASE_URL`
  - `MODEL_NAME`
  - `TIMEOUT_SECONDS`
  - `MAX_RETRIES`
  - `RETRY_BACKOFF_SECONDS`

Each thread-level task call:
- sends a system prompt defining the metric and schema
- sends a user payload containing:
  - `CURRENT_ARGUMENTS`: arguments for the current comment
  - `HISTORY_ARGUMENTS`: arguments from ancestor comments (direct chain)

---

## Output Modes

Pipeline 2 supports two output modes:

### Option A: Separate Thread-Metrics File (Collapsed per Comment)

Function: `run_thread_metrics_pipeline(...)`

- Output file: `thread_metrics.ndjson`
- Writes **one line per `(comment_id, thread_task)`**
- Preserves all original dimensions from the selected base Pipeline 1 record:
  - `task` and `result` from Pipeline 1 are renamed to:
    - `source_task`
    - `source_result`

**Output schema (per line)**
```json
{
  "... all original fields from Pipeline 1 (base record) ...": "...",
  "source_task": "<pipeline1_task>",
  "source_result": { "... pipeline1_result ..." },

  "task": "argument_novelty",
  "result": { "task": "argument_novelty", "score": 0.7, "confidence": 0.6 },

  "current_arguments": [ "...", "..." ],
  "history_arguments": [ "[<ancestor_id>] ...", "..." ],

  "comment_id_canon": "<bare_id>",
  "parent_id_canon": "<bare_id or null>",
  "link_id_canon": "<bare_submission_id or null>"
}
```

Use **Option A** if you want a **clean thread-metrics dataset** without keeping the Pipeline 1 `(comment_id, task)` dimension.

---

### Option B: Enrich the Original NDJSON 1:1 (No Collapsing)

- **Function:** `run_thread_metrics_pipeline_enrich(...)`
- **Output file:** `labels_*_enriched.ndjson`
- **Behavior:**
  - Writes the **same number of lines** as the Pipeline 1 input
  - Adds **per-comment thread metrics** without changing the original Pipeline 1 labeling structure

**Added fields**
- `thread_history_arguments`: `[...]`
- `thread_metrics`: `{ "argument_novelty": {...}, "semantic_entropy": {...} }`
- Canonical ID helpers:
  - `comment_id_canon`
  - `parent_id_canon`
  - `link_id_canon`

**Output schema (per line)**
```json
{
  "... original Pipeline 1 line unchanged ...": "...",
  "thread_history_arguments": [ "[<ancestor_id>] ...", "..." ],
  "thread_metrics": {
    "argument_novelty": { "task": "argument_novelty", "score": 0.7, "confidence": 0.6 },
    "semantic_entropy": { "task": "semantic_entropy", "score": 1.4, "confidence": 0.5 }
  },
  "comment_id_canon": "<bare_id>",
  "parent_id_canon": "<bare_id or null>",
  "link_id_canon": "<bare_submission_id or null>"
}
```
Use this mode if you want to keep **all Pipeline 1 task-lines intact**, but add thread metrics for downstream analysis.

---

### Retry and Validation Behavior

- Each thread-level task call is retried up to `MAX_RETRIES` times.
- Responses must be **strict JSON** and validated against `THREAD_TASK_SPECS`.
- Allowed outputs for `score`:
    - a numeric value within the defined range, or
    - `"ABSTAIN"` (if `allow_abstain=True`)

On repeated failure, the script writes:
```json
{ "task": "<task_name>", "error": "<last_error>" }
```
### Requirements

```bash
pip install requests
```

### Example Usage
```bash
python pipeline2_thread_metrics.py
```


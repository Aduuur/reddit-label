import os
import json
import streamlit as st
from openai import OpenAI
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional, Iterable, Tuple, Union, Set, Callable

# =============================================================================
# App configuration
# =============================================================================
st.set_page_config(page_title="Dashboard for testing LLM annotations", layout="wide")
st.title("Dashboard for testing LLM annotations")

# =============================================================================
# Choose LLM
# =============================================================================

KIT_base_url = "https://ki-toolbox.scc.kit.edu/api/v1"
KIT_api_key = os.environ["KIT_AI_API_KEY"]

AVAILABLE_MODELS = [
    "kit.gpt-oss-120b",
    "kit.mixtral-8x22b-instruct",
    "kit.qwen3.5-397b-A17b",
    "azure.gpt-4.1-mini"
]

st.sidebar.header("Choose LLM")

selected_model = st.sidebar.selectbox(
    "Select model",
    AVAILABLE_MODELS,
    index=0
)

KIT_model=selected_model

# =============================================================================
# Choose LLM
# =============================================================================
def call_kit_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    client = OpenAI(api_key=KIT_api_key, base_url=KIT_base_url)
    try:
        # Send chat request
        resp = client.chat.completions.create(
            model=KIT_model,
            messages=messages,
            temperature=temperature)
    except KeyError as exc:
        raise RuntimeError("Error") from exc

    return resp.choices[0].message.content.strip()

# ------------------------
# Single annotation
# ------------------------

def annotate_one(task: str, text: str, parent_text: Optional[str] = None) -> Dict[str, Any]:
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task, text, parent_text)},
        ]

        raw = call_kit_chat(messages, temperature=0.0)

        obj = json.loads(raw)
        return obj
    except KeyError as exc:
        raise RuntimeError("Error") from exc

# ------------------------
# Prompt construction
# ------------------------

def build_user_prompt(task: str, text: str, parent_text: Optional[str] = None) -> str:
    """
    Compose the user message for labeling tasks that includes:
      - TEXT
      - optional PARENT_TEXT
      - a directive specifying which task to perform.
    """
    parts: Dict[str, Any] = {"TEXT": text}
    if parent_text is not None and str(parent_text).strip():
        parts["PARENT_TEXT"] = str(parent_text)
    directive = TASK_INSTRUCTION_TEMPLATE.format(task_name=task)
    return json.dumps(parts, ensure_ascii=False) + "\n\n" + directive

# =============================================================================
# Load data
# =============================================================================

p_com = Path("../data/data.ndjson")
st.write("Exists comments:", p_com.exists(), p_com.resolve())

if p_com.exists():
    df_comments = pd.read_json(p_com, lines=True)
else:
    df_comments = pd.DataFrame()

# Required columns für diesen Datensatz
required = ["comment_id", "body", "parent_id", "submission_id", "user", "created_utc", "depth"]

if not df_comments.empty:
    missing = [c for c in required if c not in df_comments.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_comments: {missing}")

    # IDs/Text sauber typisieren; numerische Felder als int lassen (besser für spätere Analysen)
    df_comments["comment_id"] = df_comments["comment_id"].astype("string")
    df_comments["parent_id"]  = df_comments["parent_id"].astype("string")
    df_comments["submission_id"] = df_comments["submission_id"].astype("string")
    df_comments["user"] = df_comments["user"].astype("string")
    df_comments["body"] = df_comments["body"].astype("string")  # kann NA enthalten
else:
    df_comments = pd.DataFrame(columns=required)

#st.write(df_comments.head())
#st.write("Columns:", list(df_comments.columns))

st.sidebar.header("Select Comment")

options = df_comments.to_dict("records")
selected_comment = st.sidebar.selectbox(
    "Choose a comment",
    options,
    format_func=lambda x: x["body"]
    #format_func=lambda x: x["body"][:80] + "..."
)

st.write("### Selected Comment")
st.write(selected_comment["body"])

st.write("### In Submission")
st.write(selected_comment["submission_title"])



# ------------------------
# System prompt (labeling tasks)
# ------------------------

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
on a 1–6 scale on this topic (1 = strongly against, 3–4 = neutral/unclear, 6 = strongly in favor).

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
Civility (ordinal 1–6)
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

# Template that tells the model which single task to execute now
TASK_INSTRUCTION_TEMPLATE = (
    "Now perform ONLY the task = {task_name}. "
    "Return strictly valid JSON for that task and nothing else (no Markdown). "
    "Ensure keys and value ranges match the schema exactly."
)

tasks = [
    "stance_intensity",
    "civility",
    "epistemic_modality",
    "justification_density",
    "responsiveness",
    "agreement",
    "sarcasm",
]

st.sidebar.header("Select Tasks")

task = st.sidebar.selectbox(
    "Choose annotation tasks",
    tasks,
    index=1  # optional default
)

text = selected_comment["body"]

result = annotate_one(task, text, parent_text=None)

st.write(result)

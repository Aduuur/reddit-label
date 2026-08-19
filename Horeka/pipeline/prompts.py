"""
prompts.py
==========
Alle Task-Spezifikationen, System-Prompts und Prompt-Builder für das
Labeling der Reddit-Diskursdaten.

Wird von label_pipeline.py importiert. Hier liegt der gesamte inhaltliche
Teil (Prompts + Schema), damit die Pipeline-Logik davon getrennt bleibt.
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional


# =============================================================================
# Task-Spezifikationen (Schema-Validierung)
# =============================================================================

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "stance_intensity": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "stance_intensity",
        "score_key": "score",
        "confidence_range": (0.0, 1.0),
        "score_range": (1, 6),
        "allow_abstain": False,
        "label_key": None,
    },
    "epistemic_modality": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "epistemic_modality",
        "score_key": "score",
        "confidence_range": (0.0, 1.0),
        "score_range": (0.0, 1.0),
        "allow_abstain": False,
        "label_key": None,
    },
    "justification_density": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "justification_density",
        "score_key": "score",
        "confidence_range": (0.0, 1.0),
        "score_range": (0.0, float("inf")),
        "allow_abstain": False,
        "label_key": None,
    },
    "responsiveness": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "responsiveness",
        "score_key": "score",
        "confidence_range": (0.0, 1.0),
        "score_range": (0.0, 1.0),
        "allow_abstain": True,
        "label_key": None,
    },
    "agreement": {
        "schema_keys": {"task", "score", "confidence"},
        "task_value": "agreement",
        "score_key": "score",
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
        "allow_abstain": False,
    },
    "sarcasm": {
        "schema_keys": {"task", "label", "confidence"},
        "task_value": "sarcasm",
        "score_key": None,
        "label_key": "label",
        "label_type": (int,),
        "score_range": (0.0, 1.0),
        "confidence_range": (0.0, 1.0),
        "allow_abstain": False,
    },
}


# =============================================================================
# System-Prompt (gemeinsam für alle Labeling-Tasks)
# =============================================================================

SYSTEM_PROMPT = """You are a careful, literal discourse annotator. Use ONLY the provided TEXT,
PARENT_TEXT (if given), and TOPIC_DEF. Do not infer beyond the text or use any external knowledge.
Output must be strictly valid JSON matching the registered schema for the task. Include a "confidence" field in [0,1].

ALWAYS COMMIT TO A VALUE. For every task you must return a concrete score/label in the
required range. Do NOT abstain on the basis of weak or ambiguous evidence — instead, give your
best judgement and express your uncertainty THROUGH the "confidence" field. The confidence is the
mechanism for signalling doubt, not the label.

CONFIDENCE MUST BE HONEST AND WELL-SPREAD. Use the FULL range [0,1], not just high values:
- 0.85-1.0  : strong, unambiguous textual evidence
- 0.55-0.85 : clear leaning but with some ambiguity
- 0.30-0.55 : genuine guess; weak or conflicting signals
- 0.0-0.30  : essentially no evidence; the value is close to a coin flip
Comments with little signal should receive a real score AND a low confidence. Do not inflate
confidence. If you are unsure, that belongs in a low confidence number, not in refusing to answer.

The ONLY case where you may return "ABSTAIN" instead of a value:
- responsiveness or agreement WHEN no PARENT_TEXT is provided. Without a parent there is no
  basis to judge these two, so "ABSTAIN" is correct there. In that case set confidence <= 0.2.
For every other task, and whenever PARENT_TEXT is present, you MUST return a numeric value.

IMPORTANT:
- Output ONLY the JSON object, no Markdown, no explanation. English only.
- Never output any string other than an allowed number (or "ABSTAIN" strictly in the one case above).

--------------------------------------------------
Stance intensity (ordinal; [1,6])
--------------------------------------------------
TASK: Extract the topic of this comment and then, based on the topic, the author's stance
on a 1-6 scale (1 = strongly against, 3-4 = neutral/unclear, 6 = strongly in favor).
Always give a value. If the stance is unclear or mixed, choose the closest point (often 3-4)
and set a low confidence.
JSON schema:
{ "task": "stance_intensity", "score": <int in [1,6]>, "confidence": <float in [0,1]> }

--------------------------------------------------
Epistemic modality (continuous; [0,1])
--------------------------------------------------
TASK: Score epistemic modality: density of hedges, modal verbs, and explicit uncertainty,
normalized by length. Higher = more expressed humility/uncertainty.
Always give a value. Short texts with no hedges are a legitimate 0.0 (not a reason to abstain).
Lower your confidence if the text is too short to judge reliably.
JSON schema:
{ "task": "epistemic_modality", "score": <float in [0.0,1.0]>, "confidence": <float in [0,1]> }

--------------------------------------------------
Justification density (per 100 words)
--------------------------------------------------
TASK: Count distinct justification units (claim + warrant), normalized per 100 words.
Always give a value. A comment with no justification is a legitimate 0.0 (not a reason to abstain).
Use confidence to signal how hard it was to judge.
JSON schema:
{ "task": "justification_density", "score": <non-negative float>, "confidence": <float in [0,1]> }

--------------------------------------------------
Responsiveness (continuous; [0,1])
--------------------------------------------------
TASK: Rate how directly this reply addresses its PARENT_TEXT (semantic overlap/engagement).
If PARENT_TEXT IS provided, always give a value (use low confidence when the relation is unclear).
ONLY if PARENT_TEXT is absent, return "ABSTAIN" with confidence <= 0.2.
JSON schema:
{ "task": "responsiveness", "score": <float in [0.0,1.0] OR "ABSTAIN" only if no PARENT_TEXT>, "confidence": <float in [0,1]> }

--------------------------------------------------
Agreement (continuous; [-1,1])
--------------------------------------------------
TASK: Rate agreement with PARENT_TEXT. -1 = contradicts, 0 = neutral/unrelated, 1 = fully agrees.
If PARENT_TEXT IS provided, always give a value (use low confidence when polarity is unclear).
ONLY if PARENT_TEXT is absent, return "ABSTAIN" with confidence <= 0.2.
JSON schema:
{ "task": "agreement", "score": <float in [-1.0,1.0] OR "ABSTAIN" only if no PARENT_TEXT>, "confidence": <float in [0,1]> }

--------------------------------------------------
Civility (ordinal 1-6)
--------------------------------------------------
TASK: Rate civility on 1 (highly uncivil/insulting) to 6 (highly civil).
Always give a value. Neutral/factual comments are typically civil (5-6). If cues are mixed,
pick the closest point and lower your confidence.
JSON schema:
{ "task": "civility", "label": <integer 1..6>, "confidence": <float in [0,1]> }

--------------------------------------------------
Sarcasm (binary 0/1)
--------------------------------------------------
TASK: Detect whether the TEXT is sarcastic or clearly ironic regarding its main topic.
- Use 1 if there is clear sarcastic or ironic intent.
- Use 0 otherwise (literal text). This is the default when there is no sarcasm signal.
Always give 0 or 1; express any doubt through a low confidence.
JSON schema:
{ "task": "sarcasm", "label": <0 or 1>, "confidence": <float in [0,1]> }
"""

TASK_INSTRUCTION_TEMPLATE = (
    "Now perform ONLY the task = {task_name}. "
    "Return strictly valid JSON for that task and nothing else (no Markdown). "
    "Ensure keys and value ranges match the schema exactly."
)


# =============================================================================
# Argument-Extraktion (eigener Task)
# =============================================================================

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
  "arguments": [ { "claim": "<string>", "proof": "<string>" } ],
  "confidence": <float in [0,1]>
}

Guidelines:
- "arguments" must be a list of objects, each with keys "claim" and "proof".
- If TEXT contains no clear arguments, use an empty list [].
- Claims and proofs must be grounded in the text; do NOT hallucinate.
- Prefer under-detection over over-detection.
- Output ONLY the JSON, no Markdown.
"""

ARG_USER_TEMPLATE = (
    "TEXT: {text}\n\n"
    "If available, PARENT_TEXT (context for replies): {parent_text}\n\n"
    "Now perform ONLY argument_extraction as described in the system prompt. "
    "Return strictly valid JSON and nothing else."
)


# =============================================================================
# Prompt-Builder
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

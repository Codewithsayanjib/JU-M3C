"""Versioned prompts for T1-T4. Identical across models.

v0: initial prompts (baseline, kept for comparison).
v1: format-only fixes. T1 no combined/invented labels; T2 label list separated from
    definitions; T4 raw JSON without fences, unique entities, <=10 per list. T3 unchanged.
v1.1: REJECTED (kept for the record). v1 with ONE change to T2: each definition next to its label, plus an explicit
    instruction to output only the label text. T1/T3/T4 identical to v1.
"""
from .data import T2_LABELS, labels_for

# FROZEN 2026-09-29 (tag prompts-v1). Chosen over v1.1 by mean T2 lenient macro-F1 on full
# val across phi4_mini + qwen35_4b: v1 0.516 vs v1.1 0.471. See results/promptdev/t2_decision.md.
PROMPT_VERSION = "v1"
VERSIONS = ("v0", "v1", "v1.1")

SYSTEM = "You are a clinical documentation assistant. Follow the instructions exactly."

MAX_NEW_TOKENS = {"T1": 20, "T2": 20, "T3": 80, "T4": 400}

T2_DEFINITIONS = {
    "Operative note": "procedure/surgery report",
    "Consult / H&P": "initial evaluation: history, exam, assessment",
    "Imaging report": "radiology/diagnostic test report with findings and impression",
    "Progress note": "follow-up/SOAP",
    "Discharge summary": "hospital course at discharge",
    "ER report": "emergency department visit",
}
assert list(T2_DEFINITIONS) == T2_LABELS


def _bullets(items) -> str:
    return "\n".join(f"- {x}" for x in items)


# ---------------- v0 ----------------
def _t1_v0(text: str) -> str:
    return (
        "Classify the medical specialty of the clinical note below.\n"
        f"Choose exactly one label from this list:\n{_bullets(labels_for('T1'))}\n\n"
        "Answer with the label only, exactly as written in the list. No explanation.\n\n"
        f"Clinical note:\n{text}\n\nLabel:"
    )


def _t2_v0(text: str) -> str:
    labels = "\n".join(f"- {k}: {v}" for k, v in T2_DEFINITIONS.items())
    return (
        "Classify the document type of the clinical note below.\n"
        f"Choose exactly one label from this list:\n{labels}\n\n"
        "Answer with the label only (the text before the colon), exactly as written. "
        "No explanation.\n\n"
        f"Clinical note:\n{text}\n\nLabel:"
    )


def _t3_v0(text: str) -> str:
    return (
        "Summarize the clinical note below in ONE sentence of at most 30 words.\n"
        "Output only the sentence, with no preamble or labels.\n\n"
        f"Clinical note:\n{text}\n\nSummary:"
    )


def _t4_v0(text: str) -> str:
    return (
        "Extract the clinical entities mentioned in the note below.\n"
        "Return JSON only, with exactly these keys:\n"
        '{"diagnoses": [...], "procedures": [...], "medications": [...]}\n'
        "Each value is a list of short strings copied from the note. Use [] if none. "
        "No text outside the JSON.\n\n"
        f"Clinical note:\n{text}\n\nJSON:"
    )


# ---------------- v1 ----------------
def _t1_v1(text: str) -> str:
    return (
        "Classify the medical specialty of the clinical note below.\n"
        f"Labels:\n{_bullets(labels_for('T1'))}\n\n"
        "Copy exactly one label from the list; do not combine or invent labels. "
        "Answer with the label only. No explanation.\n\n"
        f"Clinical note:\n{text}\n\nLabel:"
    )


def _t2_v1(text: str) -> str:
    defs = "\n".join(f"{k} = {v}" for k, v in T2_DEFINITIONS.items())
    return (
        "Classify the document type of the clinical note below.\n"
        f"Labels:\n{_bullets(T2_LABELS)}\n\n"
        f"Definitions (for reference only, do not output them):\n{defs}\n\n"
        "Copy exactly one label from the Labels list. Answer with the label only. "
        "No explanation.\n\n"
        f"Clinical note:\n{text}\n\nLabel:"
    )


def _t4_v1(text: str) -> str:
    return (
        "Extract the clinical entities mentioned in the note below.\n"
        "Return a JSON object with exactly these keys:\n"
        '{"diagnoses": [...], "procedures": [...], "medications": [...]}\n'
        "Each value is a list of short strings copied from the note. Use [] if none.\n"
        "Output raw JSON only, no code fences. List each entity once, at most 10 per list. "
        "No text outside the JSON.\n\n"
        f"Clinical note:\n{text}\n\nJSON:"
    )


# ---------------- v1.1 ----------------
def _t2_v1_1(text: str) -> str:
    labels = "\n".join(f"- {k} \u2014 {v}" for k, v in T2_DEFINITIONS.items())
    return (
        "Classify the document type of the clinical note below.\n"
        f"Labels (each followed by its definition):\n{labels}\n\n"
        "Output only the label text (the part before the dash), exactly as written. "
        "Do not output the definition. No explanation.\n\n"
        f"Clinical note:\n{text}\n\nLabel:"
    )


_BUILDERS = {
    "v0": {"T1": _t1_v0, "T2": _t2_v0, "T3": _t3_v0, "T4": _t4_v0},
    "v1": {"T1": _t1_v1, "T2": _t2_v1, "T3": _t3_v0, "T4": _t4_v1},
    "v1.1": {"T1": _t1_v1, "T2": _t2_v1_1, "T3": _t3_v0, "T4": _t4_v1},
}


def build(task: str, note: dict, version: str = PROMPT_VERSION) -> tuple[str, str]:
    """Return (system, user) messages for a note."""
    return SYSTEM, _BUILDERS[version][task](note["text"].strip())

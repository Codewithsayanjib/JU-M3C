"""Zero-shot prompts for the corpus benchmark (Paper 1). Identical across models.

Versions are per task. b0 = baseline zero-shot style for every task. b1 (2026-10-03, decided on
the dev review) changes ONLY diagnosis_mcq: the prompt asks for the single correct option in a fixed
"Answer: <letter>" format, and the token limit is raised from 8 to 128 so models that explain first
are not cut off. Summarization keeps prompt b0; only its token limit rises from 320 to 450.
Older versions are kept for the record and never edited. The three MTSamples note tasks reuse the
frozen v1 prompts of src/prompts.py unchanged (T1 specialty, T2 document type, T3 summary).
Frozen on 2026-10-06 (tag bench-v1): prompts, versions and token limits below are final for the
test evaluation runs.
"""
from .prompts import SYSTEM, build as build_v1

BENCH_PROMPT_VERSION = "b0"  # default for tasks not listed in TASK_PROMPT_VERSION
TASK_PROMPT_VERSION = {"diagnosis_mcq": "b1"}

MAX_NEW_TOKENS = {"summarization": 450, "title_generation": 48, "diagnosis_mcq": 128,
                  "diagnosis_open": 32, "notes_specialty": 20, "notes_doctype": 20,
                  "notes_summary": 80, "notes_section": 12}
MAX_NEW_TOKENS_DIALOGUE = {"note_section": 200, "full_note": 900}

# MTS-Dialog section headers (the 20 native labels) with their meaning.
SECTION_HEADERS = {
    "FAM/SOCHX": "family history / social history", "GENHX": "history of present illness",
    "PASTMEDICALHX": "past medical history", "CC": "chief complaint", "PASTSURGICAL": "past surgical history",
    "ALLERGY": "allergies", "ROS": "review of systems", "MEDICATIONS": "medications",
    "ASSESSMENT": "assessment", "EXAM": "physical examination", "DIAGNOSIS": "diagnosis",
    "DISPOSITION": "disposition", "PLAN": "plan", "EDCOURSE": "emergency department course",
    "IMMUNIZATIONS": "immunizations", "IMAGING": "imaging results", "GYNHX": "gynecologic history",
    "PROCEDURES": "procedures", "OTHER_HISTORY": "other history", "LABS": "laboratory results",
}
SECTION_LABELS = list(SECTION_HEADERS)
_V1_TASK = {"notes_specialty": "T1", "notes_doctype": "T2", "notes_summary": "T3"}


def _summarization(ex):
    return ("Summarize the clinical case below in one paragraph of about 100 to 150 words. Cover the "
            "presentation, key findings, diagnosis, treatment and outcome.\n"
            "Output only the summary, with no preamble or headings.\n\n"
            f"Clinical case:\n{ex['input']}\n\nSummary:")


def _title(ex):
    return ("Write the title of the published case report that describes the clinical case below.\n"
            "One line, at most 25 words. Output only the title.\n\n"
            f"Clinical case:\n{ex['input']}\n\nTitle:")


def _mcq_b0(ex):  # kept for the record (dev run of 2026-10-02/03)
    opts = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", ex["options"]))
    return ("Read the clinical case below and choose the most likely final diagnosis.\n\n"
            f"Clinical case:\n{ex['input']}\n\nOptions:\n{opts}\n\n"
            "Answer with one letter only (A, B, C or D).\n\nAnswer:")


def _mcq(ex):  # b1
    opts = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", ex["options"]))
    return ("Read the clinical case below and choose the most likely final diagnosis.\n"
            "Exactly one of the four options is correct.\n\n"
            f"Clinical case:\n{ex['input']}\n\nOptions:\n{opts}\n\n"
            "Reply with the letter of the single correct option, in exactly this format:\n"
            "Answer: <letter>\n"
            "where <letter> is A, B, C or D. Do not explain your choice.")


def _diagnosis_open(ex):
    return ("Read the clinical case below and state the final diagnosis.\n"
            "Give the single most specific diagnosis in a few words. Output only the diagnosis, "
            "with no explanation.\n\n"
            f"Clinical case:\n{ex['input']}\n\nDiagnosis:")


def _section(ex):
    labels = "\n".join(f"- {k}" for k in SECTION_LABELS)
    defs = "\n".join(f"{k} = {v}" for k, v in SECTION_HEADERS.items())
    return ("Classify which section of a clinical note the text below belongs to.\n"
            f"Labels:\n{labels}\n\nDefinitions (for reference only, do not output them):\n{defs}\n\n"
            "Copy exactly one label from the Labels list. Answer with the label only. No explanation.\n\n"
            f"Text:\n{ex['input']}\n\nLabel:")


def _dialogue(ex):
    if ex.get("target_kind") == "full_note":
        return ("Write the clinical note for the doctor-patient conversation below. Use the standard "
                "sections of a visit note (history, examination, results, assessment and plan). "
                "Include only information stated in the conversation.\n"
                "Output only the note.\n\n"
                f"Conversation:\n{ex['input']}\n\nClinical note:")
    name = SECTION_HEADERS.get(ex.get("section_header"), "the relevant section")
    return (f"Write the '{name}' section of a clinical note from the doctor-patient conversation below. "
            "Include only information stated in the conversation. Be concise.\n"
            "Output only the section text, with no heading.\n\n"
            f"Conversation:\n{ex['input']}\n\nSection text:")


_BUILDERS = {"summarization": _summarization, "title_generation": _title, "diagnosis_mcq": _mcq,
             "diagnosis_open": _diagnosis_open, "notes_section": _section, "dialogue2note": _dialogue}


def build(task: str, ex: dict) -> tuple[str, str]:
    """(system, user) for one task example."""
    if task in _V1_TASK:
        return build_v1(_V1_TASK[task], {"text": ex["input"]})
    return SYSTEM, _BUILDERS[task](ex)


def prompt_version(task: str) -> str:
    return TASK_PROMPT_VERSION.get(task, BENCH_PROMPT_VERSION)


def max_new_tokens(task: str, ex: dict) -> int:
    if task == "dialogue2note":
        return MAX_NEW_TOKENS_DIALOGUE[ex.get("target_kind", "note_section")]
    return MAX_NEW_TOKENS[task]

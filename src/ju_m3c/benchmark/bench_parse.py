"""Output parsing for the corpus benchmark. Strict and lenient, raw output always kept.

Reuses the frozen parsers of src/parse.py (strip_thinking, parse_label, parse_summary).

bench-v1.1 (2026-10-12, after the test outputs were scored once under bench-v1): every output is cut
at the first end-of-turn token before parsing. Phi-3.5-mini's end-of-turn token "<|end|>" was not a
stop token in the runner, so it was written into the output and generation sometimes continued
after it. The rule is the same for all models; raw outputs are never modified.
"""
import re

from .data import labels_for
from .metrics import strip_markdown
from .parse import parse_label, parse_summary, strip_thinking
from .bench_prompts import SECTION_LABELS

_LETTER = re.compile(r"\b([ABCD])\b")
_ANSWER = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\(?\**([ABCD])\b", re.I)
_PREFIX = re.compile(r"^(\*\*)?(answer|title|diagnosis|final diagnosis|summary|section text|label)(\*\*)?\s*[:\-]\s*", re.I)
CLASS_LABELS = {"notes_specialty": lambda: labels_for("T1"), "notes_doctype": lambda: labels_for("T2"),
                "notes_section": lambda: SECTION_LABELS}


END_TOKENS = ("<|end|>", "<|endoftext|>", "<|end_of_text|>", "<|im_end|>", "<|eot_id|>", "<end_of_turn>",
              "<turn|>", "</s>")
_END = re.compile("|".join(re.escape(t) for t in END_TOKENS))


def cut_at_end_token(raw: str) -> str:
    """The output up to the first end-of-turn token (unchanged if there is none)."""
    m = _END.search(raw)
    return raw[:m.start()] if m else raw


def has_end_token(raw: str) -> bool:
    return _END.search(raw) is not None


def clean_text(raw: str) -> str:
    text, _ = strip_thinking(raw)
    return strip_markdown(_PREFIX.sub("", text.strip())).strip()


def first_line(raw: str) -> str:
    lines = [l.strip() for l in clean_text(raw).splitlines() if l.strip()]
    return strip_markdown(_PREFIX.sub("", lines[0])).strip().strip('"').strip() if lines else ""


def parse_mcq(raw: str, options: list[str]) -> dict:
    """strict = the output is exactly "Answer: X" (prompt b1) or exactly one letter.
    lenient, in order: the first "Answer: X" anywhere; a single distinct letter A-D in the first
    line, else in the whole output; else exactly one option's text appears in the output."""
    exact, _ = strip_thinking(raw)
    if exact in ("A", "B", "C", "D"):
        return {"parsed": exact, "strict": True, "lenient": True}
    m = re.fullmatch(r"Answer:\s*([ABCD])", exact)
    if m:
        return {"parsed": m.group(1), "strict": True, "lenient": True}
    m = _ANSWER.search(exact)
    if m:
        return {"parsed": m.group(1).upper(), "strict": False, "lenient": True}
    text = clean_text(raw)
    if text in ("A", "B", "C", "D"):
        return {"parsed": text, "strict": False, "lenient": True}
    head = text.split("\n")[0]
    letters = set(_LETTER.findall(head)) or set(_LETTER.findall(text))
    if len(letters) == 1:
        return {"parsed": letters.pop(), "strict": False, "lenient": True}
    hits = [l for l, o in zip("ABCD", options) if o.strip().lower() in text.lower()]
    if len(hits) == 1:
        return {"parsed": hits[0], "strict": False, "lenient": True}
    return {"parsed": None, "strict": False, "lenient": False}


def parse(task: str, raw: str, ex: dict) -> dict:
    """Returns {"parsed", "strict", "lenient"} for any benchmark task."""
    raw = cut_at_end_token(raw)
    if task == "diagnosis_mcq":
        return parse_mcq(raw, ex["options"])
    if task in CLASS_LABELS:
        labels = CLASS_LABELS[task]()
        r = parse_label(raw, labels)
        if r["strict"]:
            return {"parsed": r["label"], "strict": True, "lenient": True}
        # Models that add an explanation after the label ("CC\n\nExplanation: ...") mention other
        # labels later on; the label on the first line is the answer.
        head = first_line(raw)
        h = parse_label(head, labels) if head else {"lenient": False}
        if h["lenient"]:
            return {"parsed": h["label"], "strict": False, "lenient": True}
        return {"parsed": r["label"], "strict": False, "lenient": r["lenient"]}
    if task == "notes_summary":
        r = parse_summary(raw)
        return {"parsed": strip_markdown(r["summary"]), "strict": r["strict"], "lenient": r["lenient"]}
    text, _ = strip_thinking(raw)
    if task in ("title_generation", "diagnosis_open"):
        out = first_line(raw)
        one_line = len([l for l in text.strip().splitlines() if l.strip()]) == 1
        return {"parsed": out, "strict": bool(out) and one_line and out == text.strip(), "lenient": bool(out)}
    out = clean_text(raw)  # summarization, dialogue2note: the whole cleaned output
    return {"parsed": out, "strict": bool(out) and out == text.strip(), "lenient": bool(out)}

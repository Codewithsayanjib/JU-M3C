"""Strict and lenient parsers. Raw output is always kept by the caller."""
import json
import re

ENTITY_KEYS = ("diagnoses", "procedures", "medications")

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
# Gemma 4 reasoning channel markers.
_CHANNEL_RE = re.compile(r"<\|channel>.*?(<channel\|>|$)", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> and <|channel>...<channel|> blocks (unclosed = to end)."""
    new = _CHANNEL_RE.sub("", _THINK_RE.sub("", text))
    new = new.replace("</think>", "")  # stray closer from an empty-think template
    return new.strip(), new.strip() != text.strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def parse_label(raw: str, labels: list[str]) -> dict:
    """strict = exact label; lenient = normalized exact, else a single label as substring.

    When several labels match as substrings, labels contained in another matched label are
    dropped (e.g. 'Urology' inside 'Neurology'); the match counts only if one remains.
    """
    text, _ = strip_thinking(raw)
    strict = text if text in labels else None
    if strict is not None:
        return {"label": strict, "strict": True, "lenient": True}

    norm = _norm(text)
    by_norm = {_norm(l): l for l in labels}
    if norm in by_norm:
        return {"label": by_norm[norm], "strict": False, "lenient": True}

    hits = [l for l in labels if _norm(l) and _norm(l) in norm]
    hits = [h for h in hits if not any(h != o and _norm(h) in _norm(o) for o in hits)]
    if len(hits) == 1:
        return {"label": hits[0], "strict": False, "lenient": True}
    return {"label": None, "strict": False, "lenient": False}


def _first_json_object(text: str) -> str | None:
    """First balanced {...} block, respecting strings."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _dedup(items: list[str]) -> list[str]:
    """Drop repeats (case/whitespace-insensitive), keep first occurrence."""
    seen, out = set(), []
    for x in items:
        k = " ".join(x.lower().split())
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _normalize_entities(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in ENTITY_KEYS:
        v = obj.get(k, [])
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            v = []
        out[k] = _dedup([str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()])
    return out


_KEY_RE = re.compile(r'"(diagnoses|procedures|medications)"\s*:\s*\[')
_STR_RE = re.compile(r'\s*"((?:[^"\\]|\\.)*)"\s*(,|\])?')


def salvage_entities(text: str) -> dict | None:
    """Recover complete string items from truncated JSON (e.g. cut off at max tokens).

    For each key whose list starts, keep every fully closed "..." item before the cut.
    Returns None if no complete item was recovered.
    """
    out = {k: [] for k in ENTITY_KEYS}
    for m in _KEY_RE.finditer(text):
        pos, items = m.end(), []
        while True:
            sm = _STR_RE.match(text, pos)
            if not sm:
                break
            try:
                items.append(json.loads(f'"{sm.group(1)}"'))
            except json.JSONDecodeError:
                break
            pos = sm.end()
            if sm.group(2) != ",":
                break
        out[m.group(1)] = items
    return _normalize_entities(out) if any(out.values()) else None


def parse_entities(raw: str) -> dict:
    """strict = json.loads on full (thinking-stripped) output; lenient = first {...} block;
    salvaged = neither worked, but complete items were recovered from truncated JSON.
    salvaged is reported separately and never counted as strict or lenient."""
    text, _ = strip_thinking(raw)
    try:
        parsed = _normalize_entities(json.loads(text))
        if parsed is not None:
            return {"entities": parsed, "strict": True, "lenient": True, "salvaged": False}
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_RE.search(text)
    candidate = _first_json_object(fenced.group(1) if fenced else text)
    if candidate is None and fenced:
        candidate = _first_json_object(text)
    if candidate is not None:
        try:
            parsed = _normalize_entities(json.loads(candidate))
            if parsed is not None:
                return {"entities": parsed, "strict": False, "lenient": True, "salvaged": False}
        except json.JSONDecodeError:
            pass

    salvaged = salvage_entities(text)
    if salvaged is not None:
        return {"entities": salvaged, "strict": False, "lenient": False, "salvaged": True}
    return {"entities": {k: [] for k in ENTITY_KEYS}, "strict": False, "lenient": False,
            "salvaged": False}


def parse_summary(raw: str) -> dict:
    """T3: strict = non-empty single line; lenient = first non-empty line, 'Summary:' removed."""
    text, _ = strip_thinking(raw)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return {"summary": "", "strict": False, "lenient": False}
    first = re.sub(r"^(\*\*)?summary(\*\*)?\s*:\s*", "", lines[0], flags=re.I).strip()
    return {"summary": first, "strict": len(lines) == 1 and first == lines[0], "lenient": bool(first)}


def parse(task: str, raw: str, labels: list[str] | None = None) -> dict:
    if task in ("T1", "T2"):
        r = parse_label(raw, labels)
        return {"parsed": r["label"], "strict": r["strict"], "lenient": r["lenient"], "salvaged": False}
    if task == "T3":
        r = parse_summary(raw)
        return {"parsed": r["summary"], "strict": r["strict"], "lenient": r["lenient"], "salvaged": False}
    if task == "T4":
        r = parse_entities(raw)
        return {"parsed": r["entities"], "strict": r["strict"], "lenient": r["lenient"],
                "salvaged": r["salvaged"]}
    raise ValueError(task)

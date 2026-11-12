"""Task metrics. Scoring re-parses raw_output with the CURRENT parser, so runs made with
different prompt versions are compared under the same parser."""
import re
import statistics as st
from difflib import SequenceMatcher

from sklearn.metrics import accuracy_score, f1_score

from .parse import parse

NONE_LABEL = "<unparsed>"
MIN_CLASS_N = 10  # T2 "frequent classes" macro-F1 threshold

_MD_EMPH = re.compile(r"(\*\*|__)")


def strip_markdown(s: str) -> str:
    """Scoring-time cleanup for T3 (all models): drop **, __ and leading/trailing *."""
    return _MD_EMPH.sub("", s).strip().strip("*").strip()


def classification(rows: list[dict], gold: dict, labels: list[str], mode: str = "lenient") -> dict:
    """mode 'strict': only exact-label outputs count; 'lenient': lenient parse. Unparsed = wrong."""
    y_true, y_pred = [], []
    for r in rows:
        p = parse(r["task"], r["raw_output"], labels)
        ok = p["strict"] if mode == "strict" else p["lenient"]
        y_true.append(gold[r["note_id"]])
        y_pred.append(p["parsed"] if ok else NONE_LABEL)
    present = sorted(set(y_true))  # macro-F1 over gold classes present in this subset
    return {
        "n": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "per_class_f1": dict(zip(present, f1_score(y_true, y_pred, labels=present, average=None,
                                                    zero_division=0).round(3).tolist())),
        "y_true": y_true, "y_pred": y_pred,
    }


def parse_rates(rows: list[dict], labels=None) -> dict:
    ps = [parse(r["task"], r["raw_output"], labels) for r in rows]
    n = len(ps)
    return {
        "strict_fail": sum(not p["strict"] for p in ps) / n,
        "lenient_fail": sum(not p["lenient"] for p in ps) / n,
        "salvaged": sum(p["salvaged"] for p in ps) / n,
        "finish_length": sum(r.get("finish_reason") == "length" for r in rows) / n,
    }


_ROUGE = None


def summary(rows: list[dict], gold: dict) -> dict:
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    preds = [strip_markdown(parse("T3", r["raw_output"])["parsed"]) for r in rows]
    rl = [_ROUGE.score(gold[r["note_id"]], p)["rougeL"].fmeasure for r, p in zip(rows, preds)]
    lens = [len(p.split()) for p in preds]
    return {
        "n": len(rows),
        "rougeL": st.mean(rl),
        "pct_le_30_words": sum(0 < l <= 30 for l in lens) / len(lens),
        "mean_words": st.mean(lens),
    }


def bertscore(preds: list[str], refs: list[str]) -> float:
    from bert_score import score
    _, _, f = score(preds, refs, lang="en", verbose=False)
    return float(f.mean())


def entities(rows: list[dict]) -> dict:
    ps = [parse("T4", r["raw_output"]) for r in rows]
    n = len(ps)
    return {
        "n": n,
        "json_strict": sum(p["strict"] for p in ps) / n,
        "json_lenient": sum(p["lenient"] for p in ps) / n,
        "json_salvaged": sum(p["salvaged"] for p in ps) / n,
        "finish_length": sum(r.get("finish_reason") == "length" for r in rows) / n,
    }


def latency(rows: list[dict]) -> dict:
    lat = [r["latency_s"] for r in rows]
    tps = [r["gen_tokens"] / r["latency_s"] for r in rows if r["latency_s"] > 0]
    return {"median_latency_s": st.median(lat), "median_tok_per_s": st.median(tps)}


def rouge_l_per_note(preds: list[str], refs: list[str]) -> list[float]:
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [_ROUGE.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


# ---- T4 silver matching ----
def _norm_ent(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def silver_keywords(note: dict) -> list[str]:
    """Keywords minus the specialty/doc-type category words MTSamples mixes in."""
    cats = {_norm_ent(c) for c in note.get("raw_categories", [])}
    kws = {_norm_ent(k) for k in note["keywords"]}
    return sorted(k for k in kws if k and k not in cats)


def _match(a: str, b: str) -> bool:
    """Normalized fuzzy match: containment (min 3 chars) or similarity >= 0.9."""
    if a == b:
        return True
    if min(len(a), len(b)) >= 3 and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.9


def entity_recall_per_note(entities: dict, note: dict) -> tuple[float, float | None]:
    """(recall of silver keywords, precision of predictions). Precision None if no predictions."""
    preds = [_norm_ent(x) for v in entities.values() for x in v]
    preds = [p for p in preds if p]
    kws = silver_keywords(note)
    rec = sum(any(_match(k, p) for p in preds) for k in kws) / len(kws) if kws else 0.0
    prec = sum(any(_match(p, k) for k in kws) for p in preds) / len(preds) if preds else None
    return rec, prec

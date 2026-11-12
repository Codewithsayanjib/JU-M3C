"""Scoring for the corpus benchmark. Frozen on 2026-10-06 after the dev output review (tag bench-v1).

  python scripts/benchmark/score_bench.py --subset dev [--no-bertscore] [--models a,b]

Reads results/bench/{subset}/{model}/{task}.jsonl and the task files under $CORPUS_ROOT (for references).
Writes results/bench/{subset}/summary.csv (model, task, metric, value, ci_lo, ci_hi, n).
Outputs are re-parsed with the current parser. Bootstrap 95% CIs: 1,000 resamples, seed 42.
Scoring the test evaluation subset needs --confirm-test.

Outputs are cut at the first end-of-turn token before parsing (bench-v1.1, see src/bench_parse.py).

Main metric per task (frozen):
  summarization, title_generation, notes_summary, dialogue2note : ROUGE-L (+ BERTScore F1, length)
  diagnosis_mcq                                                  : accuracy (chance 0.25) + position stats
  diagnosis_open                                                 : token F1 (+ exact, contains,
                                                                   SapBERT cosine as secondary;
                                                                   spelling-normalised exact/contains
                                                                   and SapBERT match rate: appendix)
  notes_specialty, notes_doctype, notes_section                  : macro-F1 (+ accuracy)
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import f1_score  # noqa: E402

from ju_m3c.benchmark import stats  # noqa: E402
from ju_m3c.benchmark.bench_parse import has_end_token, parse  # noqa: E402
from ju_m3c.benchmark.metrics import rouge_l_per_note  # noqa: E402

CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
GEN = ("summarization", "title_generation", "notes_summary", "dialogue2note")
CLS = ("notes_specialty", "notes_doctype", "notes_section")
MAIN = {**{t: "rougeL" for t in GEN}, **{t: "macro_f1" for t in CLS},
        "diagnosis_mcq": "accuracy", "diagnosis_open": "token_f1"}
_ART = re.compile(r"\b(a|an|the)\b")
# British -> American spelling and simple plural folding, for the synonym-normalised match.
_SPELL = [("haem", "hem"), ("oedema", "edema"), ("oesophag", "esophag"), ("tumour", "tumor"),
          ("aemia", "emia"), ("paediatr", "pediatr"), ("foetal", "fetal"), ("leukaem", "leukem"),
          ("ischaem", "ischem"), ("anaem", "anem"), ("orthopaed", "orthoped"), ("gynaecolog", "gynecolog"),
          ("oestrogen", "estrogen"), ("coeliac", "celiac"), ("diarrhoea", "diarrhea"), ("faecal", "fecal")]
SAPBERT = ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "090663c3ae57bf35ffe4d0d468a2a88d03051a4d")
SAPBERT_MATCH = 0.85  # not validated by hand; appendix only, the mean cosine is the secondary metric
_sap = None


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())
    return " ".join(_ART.sub(" ", s).split())


def syn_norm(s: str) -> str:
    out = []
    for w in norm(s).split():
        for a, b in _SPELL:
            w = w.replace(a, b)
        if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
REFS_CACHE = {}


def get_refs(task: str, subset: str) -> dict:
    key = (task, subset)
    if key not in REFS_CACHE:
        path = CORPUS_ROOT / "processed" / "tasks" / task / f"{subset}.jsonl"
        with open(path) as f:
            REFS_CACHE[key] = {json.loads(l)["example_id"]: json.loads(l) for l in f}
    return REFS_CACHE[key]


def sapbert_cosine(preds: list[str], refs: list[str]) -> list[float]:
    """Cosine of SapBERT [CLS] embeddings (a model trained to place synonymous medical terms close)."""
    global _sap
    from transformers import AutoModel, AutoTokenizer
    if _sap is None:
        tok = AutoTokenizer.from_pretrained(SAPBERT[0], revision=SAPBERT[1])
        mod = AutoModel.from_pretrained(SAPBERT[0], revision=SAPBERT[1]).to(DEVICE).eval()
        _sap = (tok, mod)
    tok, mod = _sap

    def emb(xs):
        out = []
        for i in range(0, len(xs), 64):
            b = tok([x or "" for x in xs[i:i + 64]], padding=True, truncation=True, max_length=25,
                    return_tensors="pt")
            b = {k: v.to(DEVICE) for k, v in b.items()}
            with torch.no_grad():
                v = mod(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out)
    return (emb(preds) * emb(refs)).sum(-1).tolist()


def token_f1(pred: str, ref: str) -> float:
    p, r = norm(pred).split(), norm(ref).split()
    common = sum((Counter(p) & Counter(r)).values())
    return 0.0 if not common else 2 * common / (len(p) + len(r))


def add(out, model, task, metric, point, lo=None, hi=None, n=None):
    out.append({"model": model, "task": task, "metric": metric, "value": round(float(point), 4),
                "ci_lo": None if lo is None else round(lo, 4), "ci_hi": None if hi is None else round(hi, 4),
                "n": n, "main": metric == MAIN.get(task)})


def score_task(model, task, rows, refs, out, bert):
    ps = [parse(task, r["raw_output"], refs[r["example_id"]]) for r in rows]
    exs = [refs[r["example_id"]] for r in rows]
    n = len(rows)
    add(out, model, task, "parse_fail_strict", *stats.mean_ci([not p["strict"] for p in ps]), n)
    add(out, model, task, "parse_fail_lenient", *stats.mean_ci([not p["lenient"] for p in ps]), n)
    # an output that contains an end-of-turn token was finished by the model, whatever came after it
    add(out, model, task, "hit_token_limit", *stats.mean_ci(
        [r["finish_reason"] == "length" and not has_end_token(r["raw_output"]) for r in rows]), n)
    add(out, model, task, "latency_s_median", *stats.median_ci([r["latency_s"] for r in rows]), n)
    add(out, model, task, "ttft_s_median", *stats.median_ci([r["ttft_s"] for r in rows]), n)
    dec = [r["decode_tps"] for r in rows if r["gen_tokens"] > 1]
    if dec:
        add(out, model, task, "decode_tps_median", *stats.median_ci(dec), len(dec))

    if task in GEN:
        preds = [p["parsed"] or "" for p in ps]
        targets = [e["target"] for e in exs]
        add(out, model, task, "rougeL", *stats.mean_ci(rouge_l_per_note(preds, targets)), n)
        add(out, model, task, "mean_words", *stats.mean_ci([len(p.split()) for p in preds]), n)
        add(out, model, task, "length_ratio", *stats.mean_ci(
            [len(p.split()) / max(1, len(t.split())) for p, t in zip(preds, targets)]), n)
        if bert:
            from bert_score import score
            _, _, f = score(preds, targets, lang="en", device=DEVICE, verbose=False)
            add(out, model, task, "bertscore_f1", *stats.mean_ci(f.numpy()), n)
    elif task in CLS:
        y = np.array([e["target"] for e in exs], dtype=object)
        pred = np.array([p["parsed"] if p["lenient"] else "<unparsed>" for p in ps], dtype=object)
        present = sorted(set(y))
        add(out, model, task, "accuracy", *stats.mean_ci(pred == y), n)
        add(out, model, task, "macro_f1", *stats.bootstrap_ci(
            lambda i: f1_score(y[i], pred[i], labels=present, average="macro", zero_division=0), n), n)
        top, c = Counter(pred).most_common(1)[0]
        add(out, model, task, "most_common_prediction_share", c / n, n=n)
    elif task == "diagnosis_mcq":
        gold = [e["target"] for e in exs]
        pred = [p["parsed"] if p["lenient"] else None for p in ps]
        add(out, model, task, "accuracy", *stats.mean_ci([a == b for a, b in zip(pred, gold)]), n)
        for letter in "ABCD":  # position bias: share of predictions and accuracy when gold is that letter
            add(out, model, task, f"predicted_share[{letter}]", sum(p == letter for p in pred) / n, n=n)
            idx = [i for i, g in enumerate(gold) if g == letter]
            if idx:
                add(out, model, task, f"accuracy_when_gold[{letter}]",
                    sum(pred[i] == letter for i in idx) / len(idx), n=len(idx))
    elif task == "diagnosis_open":
        preds = [p["parsed"] or "" for p in ps]
        targets = [e["target"] for e in exs]
        add(out, model, task, "token_f1", *stats.mean_ci([token_f1(p, t) for p, t in zip(preds, targets)]), n)
        add(out, model, task, "exact_match", *stats.mean_ci([norm(p) == norm(t) for p, t in zip(preds, targets)]), n)
        add(out, model, task, "contains_match", *stats.mean_ci(
            [bool(norm(t)) and (norm(t) in norm(p) or (bool(norm(p)) and norm(p) in norm(t)))
             for p, t in zip(preds, targets)]), n)
        sp, st_ = [syn_norm(p) for p in preds], [syn_norm(t) for t in targets]
        add(out, model, task, "syn_exact_match", *stats.mean_ci([a == b for a, b in zip(sp, st_)]), n)
        add(out, model, task, "syn_contains_match", *stats.mean_ci(
            [bool(b) and (b in a or (bool(a) and a in b)) for a, b in zip(sp, st_)]), n)
        if bert:  # same switch as BERTScore: needs a model download on first use
            cos = sapbert_cosine(preds, targets)
            add(out, model, task, "sapbert_cosine", *stats.mean_ci(cos), n)
            add(out, model, task, f"sapbert_match@{SAPBERT_MATCH}", *stats.mean_ci([c >= SAPBERT_MATCH for c in cos]), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev", choices=["dev", "test_eval"])
    ap.add_argument("--models", default=None)
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    if a.subset == "test_eval" and not a.confirm_test:
        sys.exit("Scoring the test evaluation subset needs --confirm-test (after scoring is frozen).")
    root = ROOT / "results" / "bench" / ("sealed_test_eval" if a.subset == "test_eval" else a.subset)
    summary_path = root / "summary.csv"
    fieldnames = ["model", "task", "metric", "value", "ci_lo", "ci_hi", "n", "main"]

    # Pre-cache all task reference sets so external drive is read once upfront
    print(f"Pre-caching reference datasets from {CORPUS_ROOT}...", flush=True)
    for task_dir in (CORPUS_ROOT / "processed" / "tasks").iterdir():
        if (task_dir / f"{a.subset}.jsonl").exists():
            get_refs(task_dir.name, a.subset)
    print(f"Cached {len(REFS_CACHE)} task references into memory.", flush=True)

    out = []
    if a.models and summary_path.exists():  # re-scoring some models: keep the rows of the others
        keep = [r for r in csv.DictReader(open(summary_path)) if r["model"] not in a.models.split(",")]
        out = [{k: r[k] for k in fieldnames} for r in keep]
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "analysis"):
        if a.models and d.name not in a.models.split(","):
            continue
        for f in sorted(d.glob("*.jsonl")):
            task = f.stem
            refs = get_refs(task, a.subset)
            rows = [json.loads(l) for l in open(f)]
            print(f"scoring {d.name} {task} (n={len(rows)})", flush=True)
            score_task(d.name, task, rows, refs, out, not a.no_bertscore)
        meta = d / "run_meta.json"
        if meta.exists():
            add(out, d.name, "all", "peak_memory_gb", json.loads(meta.read_text())["peak_memory_gb_max"])
        # Incrementally update summary.csv after each model completes
        with open(summary_path, "w", newline="") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out)

    print("wrote", summary_path)


if __name__ == "__main__":
    main()

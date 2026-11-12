"""Score the prompt-variant runs (paraphrased instruction, clinical role) against the zero-shot outputs
for the same examples. Same method as scripts/experiments/score_fewshot.py.

  python scripts/experiments/score_prompt_variants.py

For every model and task: main metric zero-shot and few-shot on the same examples, the difference
with a paired bootstrap 95% interval (1,000 resamples, seed 42), and strict parse-failure rates.
Writes results/bench/prompt_variants/summary_prompt_variants.csv and docs/bench_prompt_variants.md.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
os.environ.setdefault("CORPUS_ROOT", "corpus")

from analyze_bench import md as table  # noqa: E402
from score_bench import CLS, GEN, MAIN, get_refs, token_f1  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from ju_m3c.benchmark.bench_parse import parse  # noqa: E402
from ju_m3c.benchmark.metrics import rouge_l_per_note  # noqa: E402

PV = ROOT / "results" / "bench" / "prompt_variants"
ZS = ROOT / "results" / "bench" / "sealed_test_eval"
TASKS = ["diagnosis_mcq", "diagnosis_open", "notes_specialty", "notes_doctype", "notes_section",
         "summarization", "notes_summary", "title_generation", "dialogue2note"]
SEED, N_BOOT = 42, 1000


def scores(task, raws, exs):
    ps = [parse(task, r, e) for r, e in zip(raws, exs)]
    if task in GEN:
        return np.array(rouge_l_per_note([p["parsed"] or "" for p in ps], [e["target"] for e in exs])), ps
    if task == "diagnosis_open":
        return np.array([token_f1(p["parsed"] or "", e["target"]) for p, e in zip(ps, exs)]), ps
    pred = np.array([p["parsed"] if p["lenient"] else "<unparsed>" for p in ps], dtype=object)
    return pred, ps


def main():
    out = []
    for style in ("paraphrase", "role"):
        out += score_style(style, PV / style)
    write(pd.DataFrame(out))


def score_style(style, FS):
    rng = np.random.default_rng(SEED)
    out = []
    models = sorted(p.name for p in FS.iterdir() if p.is_dir())
    for t in TASKS:
        refs = get_refs(t, "test_eval")
        idx_cache = None
        for m in models:
            fs = {r["example_id"]: r for r in map(json.loads, open(FS / m / f"{t}.jsonl"))}
            zs = {r["example_id"]: r for r in map(json.loads, open(ZS / m / f"{t}.jsonl"))}
            ids = sorted(fs)
            exs = [refs[i] for i in ids]
            a, pa = scores(t, [zs[i]["raw_output"] for i in ids], exs)
            b, pb = scores(t, [fs[i]["raw_output"] for i in ids], exs)
            n = len(ids)
            if idx_cache is None:
                idx_cache = [rng.integers(0, n, n) for _ in range(N_BOOT)]
            if t in CLS or t == "diagnosis_mcq":
                y = np.array([e["target"] for e in exs], dtype=object)
                if t in CLS:
                    labs = sorted(set(y))
                    f = lambda p, k: f1_score(y[k], p[k], labels=labs, average="macro", zero_division=0)
                else:
                    f = lambda p, k: float(np.mean(p[k] == y[k]))
                full = np.arange(n)
                va, vb = f(a, full), f(b, full)
                diffs = np.array([f(b, k) - f(a, k) for k in idx_cache])
            else:
                va, vb = float(a.mean()), float(b.mean())
                diffs = np.array([b[k].mean() - a[k].mean() for k in idx_cache])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            row = {"style": style, "task": t, "model": m, "metric": MAIN[t], "n": n, "zero_shot": va, "few_shot": vb, "diff": vb - va,
                   "ci_lo": lo, "ci_hi": hi, "significant": bool(lo > 0 or hi < 0),
                   "strict_fail_zero": 1 - np.mean([p["strict"] for p in pa]), "strict_fail_few": 1 - np.mean([p["strict"] for p in pb])}
            if t == "diagnosis_mcq":
                for l in "ABCD":
                    row[f"share_{l}_zero"] = float(np.mean(a == l)); row[f"share_{l}_few"] = float(np.mean(b == l))
            out.append(row)
        print("scored", style, t, flush=True)
    return out


def write(df):
    df.to_csv(PV / "summary_prompt_variants.csv", index=False)
    md = ["# Prompt variants against zero-shot, same examples\n",
          "Difference = variant minus zero-shot, paired bootstrap 95% interval. * = interval excludes zero.\n"]
    for style, d in df.groupby("style"):
        md.append(f"\n## {style}\n")
        md.append(table(d.assign(cell=[f"{x:+.3f}{'*' if s else ''}" for x, s in zip(d["diff"], d.significant)]).pivot(index="model", columns="task", values="cell")[TASKS]))
    (ROOT / "docs" / "bench_prompt_variants.md").write_text("\n".join(md) + "\n")
    print("wrote", PV / "summary_prompt_variants.csv", "and docs/bench_prompt_variants.md")


if __name__ == "__main__":
    main()

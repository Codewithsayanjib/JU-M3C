"""Score the larger reference model (Qwen3.5-9B) against the twelve small models on the same
300 examples per task (zero-shot, frozen prompts and scoring).

  python scripts/experiments/score_reference.py

Writes results/bench/prompt_variants/zeroshot/summary_reference.csv and docs/bench_reference.md.
Difference: reference minus the best small model on the same examples, paired bootstrap 95% interval.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
os.environ.setdefault("CORPUS_ROOT", "corpus")

from analyze_bench import md as table  # noqa: E402
from score_bench import CLS, MAIN, get_refs  # noqa: E402
from score_fewshot import TASKS, scores  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

REF = "qwen35_9b_ref"
RD = ROOT / "results" / "bench" / "prompt_variants" / "zeroshot" / REF
ZS = ROOT / "results" / "bench" / "sealed_test_eval"
SEED, N_BOOT = 42, 1000


def main():
    rng = np.random.default_rng(SEED)
    models = sorted(p.name for p in ZS.iterdir() if p.is_dir() and p.name != "analysis")
    rows = []
    for t in TASKS:
        refs = get_refs(t, "test_eval")
        rr = {r["example_id"]: r for r in map(json.loads, open(RD / f"{t}.jsonl"))}
        ids = sorted(rr); exs = [refs[i] for i in ids]; n = len(ids)
        y = np.array([e["target"] for e in exs], dtype=object)
        boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]

        def metric(p, k):
            if t in CLS:
                return f1_score(y[k], p[k], labels=sorted(set(y)), average="macro", zero_division=0)
            if t == "diagnosis_mcq":
                return float(np.mean(p[k] == y[k]))
            return float(p[k].mean())
        vals = {}
        for m in models + [REF]:
            src = rr if m == REF else {r["example_id"]: r for r in map(json.loads, open(ZS / m / f"{t}.jsonl"))}
            vals[m], _ = scores(t, [src[i]["raw_output"] for i in ids], exs)
        full = np.arange(n)
        sc = {m: metric(vals[m], full) for m in vals}
        best = max(models, key=lambda m: sc[m])
        d = [metric(vals[REF], k) - metric(vals[best], k) for k in boots]
        rank = 1 + sum(sc[m] > sc[REF] for m in models)
        rows.append({"task": t, "metric": MAIN[t], "n": n, "reference": sc[REF], "best_small": sc[best], "best_small_model": best,
                     "median_small": float(np.median([sc[m] for m in models])), "diff_vs_best": sc[REF] - sc[best],
                     "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "rank_among_13": rank})
        print("done", t, flush=True)
    df = pd.DataFrame(rows); df.to_csv(RD.parent / "summary_reference.csv", index=False)
    (ROOT / "docs" / "bench_reference.md").write_text("# Qwen3.5-9B reference against the twelve small models (same 300 examples per task)\n\n"
                                                     + table(df.set_index("task")) + "\n")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()

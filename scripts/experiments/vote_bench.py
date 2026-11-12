"""Majority-vote baselines from the saved test outputs (no generation).

  python scripts/experiments/vote_bench.py

For multiple-choice diagnosis and the three classification tasks, each model's lenient-parsed answer
is a vote. Ensembles: all 12 models, and the top 3 and top 5 models chosen by their DEVELOPMENT score
on the task (so that test results do not decide the members). Ties are broken by the member with the
best development score. Compared with the best single model of the test set and with the oracle
(any member correct). Paired bootstrap 95% intervals (1,000 resamples, seed 42) for ensemble minus
best single model. Writes results/bench/sealed_test_eval/analysis/vote.csv and docs/bench_vote.md.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
os.environ.setdefault("CORPUS_ROOT", "corpus")

from analyze_bench import md as table  # noqa: E402
from score_bench import get_refs  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from ju_m3c.benchmark.bench_parse import parse  # noqa: E402

RES = ROOT / "results" / "bench" / "sealed_test_eval"
TASKS = ["diagnosis_mcq", "notes_specialty", "notes_doctype", "notes_section"]
SEED, N_BOOT = 42, 1000


def main():
    dev = pd.read_csv(ROOT / "results" / "bench" / "dev" / "summary.csv")
    test = pd.read_csv(RES / "summary.csv")
    models = sorted(p.name for p in RES.iterdir() if p.is_dir() and p.name != "analysis")
    rng = np.random.default_rng(SEED)
    rows = []
    for t in TASKS:
        refs = get_refs(t, "test_eval"); ids = sorted(refs)
        y = np.array([refs[i]["target"] for i in ids], dtype=object)
        pred = {}
        for m in models:
            out = {r["example_id"]: r for r in map(json.loads, open(RES / m / f"{t}.jsonl"))}
            ps = [parse(t, out[i]["raw_output"], refs[i]) for i in ids]
            pred[m] = np.array([p["parsed"] if p["lenient"] else None for p in ps], dtype=object)
        devrank = dev[(dev.task == t) & (dev.main == True)].set_index("model").value.reindex(models).sort_values(ascending=False).index.tolist()
        best = test[(test.task == t) & (test.main == True)].set_index("model").value.idxmax()

        def vote(members):
            out = []
            for k in range(len(ids)):
                c = Counter(pred[m][k] for m in members if pred[m][k] is not None)
                if not c:
                    out.append(None); continue
                top = max(c.values())
                tied = {a for a, v in c.items() if v == top}
                out.append(next(pred[m][k] for m in members if pred[m][k] in tied))  # tie: best dev member
            return np.array(out, dtype=object)

        def score(p, idx):
            if t == "diagnosis_mcq":
                return float(np.mean(p[idx] == y[idx]))
            return f1_score(y[idx], np.where(p[idx] == None, "<none>", p[idx]).astype(object), labels=sorted(set(y)), average="macro", zero_division=0)  # noqa: E711

        full = np.arange(len(ids)); boots = [rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT)]
        base = pred[best]
        cands = {f"best single ({best})": base, "vote, all 12": vote(devrank), "vote, top 5 (dev)": vote(devrank[:5]),
                 "vote, top 3 (dev)": vote(devrank[:3])}
        for name, p in cands.items():
            s = score(p, full)
            if name.startswith("best"):
                lo = hi = d = None
            else:
                diffs = [score(p, b) - score(base, b) for b in boots]
                d = s - score(base, full); lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append({"task": t, "method": name, "score": s, "diff_vs_best": d, "ci_lo": lo, "ci_hi": hi})
        for name, mem in (("oracle, top 3 (dev)", devrank[:3]), ("oracle, all 12", models)):
            ok = np.array([any(pred[m][k] == y[k] for m in mem) for k in range(len(ids))])
            rows.append({"task": t, "method": name, "score": float(ok.mean()), "diff_vs_best": None, "ci_lo": None, "ci_hi": None})
        rows.append({"task": t, "method": "members top 3 (dev)", "score": None, "diff_vs_best": None, "ci_lo": None, "ci_hi": None,
                     "note": ", ".join(devrank[:3])})
        print("done", t, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "analysis" / "vote.csv", index=False)
    md = ["# Majority-vote baselines (test evaluation subsets)\n",
          "Main metric: accuracy for multiple choice, macro-F1 for classification. Ensemble members chosen by development "
          "score; difference against the best single test model with a paired bootstrap 95% interval. Oracle rows give "
          "accuracy (any member correct).\n"]
    md.append(table(df.drop(columns=[c for c in ["note"] if c in df]).dropna(subset=["score"]).set_index(["task", "method"])))
    md.append("\n\nTop-3 members by development score: " + "; ".join(f"{r.task}: {r.note}" for r in df[df.method == 'members top 3 (dev)'].itertuples()))
    (ROOT / "docs" / "bench_vote.md").write_text("\n".join(md) + "\n")
    print("wrote", RES / "analysis" / "vote.csv", "and docs/bench_vote.md")


if __name__ == "__main__":
    main()

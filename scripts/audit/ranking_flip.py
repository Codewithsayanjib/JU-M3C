"""Does leakage change which system looks best? Ranks a model-free 'copy the nearest training case'
baseline among the 12 benchmarked models, on exactly the same test items, with the baseline's training
pool taken from (a) the sources' own published splits (leaky) or (b) our article-level split (clean).

  python scripts/experiments/ranking_flip.py --data $JUMC_DATA \
      --items reports/leakage_inflation.items.jsonl --out reports/ranking_flip.json
"""
import argparse, csv, gzip, json
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = {"diagnosis_mcq", "title_generation"}  # same metric for models and baseline


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "items", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args(); data = Path(a.data).expanduser()
    base = {}
    for l in open(a.items):
        r = json.loads(l)
        if r["task"] in TASKS:
            base[(r["task"], r["doc_id"])] = (r["native"], r["ours"])
    sc = defaultdict(dict)
    for r in csv.DictReader(gzip.open(data / "outputs/sealed_test_eval/analysis/per_example.csv.gz", "rt")):
        if r["task"] in TASKS and r["score"] != "":
            doc = r["example_id"].split(":", 1)[1]
            if (r["task"], doc) in base:
                sc[r["task"]].setdefault(r["model"], {})[doc] = float(r["score"])
    res = {}
    rng = np.random.default_rng(42)
    for t, models in sc.items():
        common = sorted(set.intersection(*(set(v) for v in models.values())))
        M = {m: np.array([v[d] for d in common]) for m, v in models.items()}
        for lab, k in (("copy baseline, leaky pool", 0), ("copy baseline, clean pool", 1)):
            M[lab] = np.array([base[(t, d)][k] for d in common])
        means = {m: round(100 * float(x.mean()), 1) for m, x in M.items()}
        out = {"n_items": len(common), "scores": dict(sorted(means.items(), key=lambda kv: -kv[1]))}
        for lab in ("copy baseline, leaky pool", "copy baseline, clean pool"):
            others = [m for m in models]
            out[lab] = {"rank": 1 + sum(means[m] > means[lab] for m in others), "of": len(others) + 1,
                        "beats_models": sum(means[lab] > means[m] for m in others)}
        best = max(models, key=lambda m: means[m])
        d = M["copy baseline, leaky pool"] - M[best]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out["leaky_baseline_minus_best_model"] = {"best": best, "diff": round(100 * float(d.mean()), 1),
                                                   "ci95": [round(100 * float(np.percentile(bs, q)), 1) for q in (2.5, 97.5)]}
        res[t] = out
        print(t, json.dumps({k: v for k, v in out.items() if k != "scores"}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

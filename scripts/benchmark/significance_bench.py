"""Pairwise significance tests between models on the corpus benchmark (main metric per task).

  python scripts/benchmark/significance_bench.py --subset test_eval --confirm-test

No generation: re-parses the saved outputs with the frozen parser and compares models on the same
examples. Per task, all model pairs:
  diagnosis_mcq                       : exact McNemar on correct/incorrect
  ROUGE-L tasks, diagnosis_open       : paired bootstrap of the mean per-example difference
  macro-F1 tasks                      : paired bootstrap of the macro-F1 difference
Bootstrap: 10,000 resamples, seed 42, the same resamples for every model of a task (paired).
p = (count + 1) / (resamples + 1), two-sided. Holm correction within each task over all pairs.
(10,000 rather than the 1,000 used for confidence intervals: with 66 pairs per task the Holm
threshold is below the smallest p-value 1,000 resamples can give.)

Writes results/bench/{subset dir}/significance.csv and docs/bench_significance.md.
"""
import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_bench import CLS, GEN, MAIN, get_refs, token_f1  # noqa: E402
from ju_m3c.benchmark import stats  # noqa: E402
from ju_m3c.benchmark.bench_parse import parse  # noqa: E402
from ju_m3c.benchmark.metrics import rouge_l_per_note  # noqa: E402

N_BOOT, SEED, ALPHA = 10_000, 42, 0.05


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def boot_p(boots: np.ndarray) -> float:
    k = min((boots <= 0).sum(), (boots >= 0).sum())
    return min(1.0, 2 * (k + 1) / (len(boots) + 1))


def macro_f1_boot(y, pred, labels, W):
    """Macro-F1 on the full sample and on every resample (W: resamples x n multiplicities).
    Same definition as sklearn with fixed labels and zero_division=0."""
    full, boots = [], []
    for c in labels:
        g, p = (y == c).astype(float), (pred == c).astype(float)
        tp, den = g * p, g + p
        full.append(2 * tp.sum() / den.sum() if den.sum() else 0.0)
        d = W @ den
        boots.append(np.divide(2 * (W @ tp), d, out=np.zeros(len(d)), where=d > 0))
    return float(np.mean(full)), np.mean(boots, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev", choices=["dev", "test_eval"])
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    if a.subset == "test_eval" and not a.confirm_test:
        sys.exit("The test evaluation subset needs --confirm-test.")
    root = ROOT / "results" / "bench" / ("sealed_test_eval" if a.subset == "test_eval" else a.subset)
    models = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "analysis")
    rows_out, md = [], [f"# Pairwise significance tests ({a.subset}, {len(models)} models)\n",
                        "Main metric per task. Exact McNemar for multiple-choice accuracy; paired bootstrap "
                        f"({N_BOOT:,} resamples, seed {SEED}) for the other metrics. p-values are Holm-adjusted "
                        f"within each task over all {len(models) * (len(models) - 1) // 2} model pairs; "
                        f"significance level {ALPHA}.\n"]
    for task in sorted(MAIN):
        refs = get_refs(task, a.subset)
        ids = sorted(refs)
        exs = [refs[i] for i in ids]
        n = len(ids)
        rng = np.random.default_rng(SEED)
        W = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(float)
        point, per_ex, boots = {}, {}, {}
        for m in models:
            by_id = {r["example_id"]: r for r in map(json.loads, open(root / m / f"{task}.jsonl"))}
            if set(by_id) != set(ids):
                sys.exit(f"{m} {task}: examples do not match the task file")
            ps = [parse(task, by_id[i]["raw_output"], refs[i]) for i in ids]
            if task in CLS:
                y = np.array([e["target"] for e in exs], dtype=object)
                pred = np.array([p["parsed"] if p["lenient"] else "<unparsed>" for p in ps], dtype=object)
                point[m], boots[m] = macro_f1_boot(y, pred, sorted(set(y)), W)
                continue
            if task == "diagnosis_mcq":
                s = [(p["parsed"] if p["lenient"] else None) == e["target"] for p, e in zip(ps, exs)]
            elif task == "diagnosis_open":
                s = [token_f1(p["parsed"] or "", e["target"]) for p, e in zip(ps, exs)]
            else:
                s = rouge_l_per_note([p["parsed"] or "" for p in ps], [e["target"] for e in exs])
            per_ex[m] = np.asarray(s, float)
            point[m] = float(per_ex[m].mean())
            boots[m] = W @ per_ex[m] / n
        pairs = []
        for x, y_ in combinations(models, 2):
            d = boots[x] - boots[y_]
            lo, hi = np.percentile(d, [2.5, 97.5])
            if task == "diagnosis_mcq":
                test, p = "mcnemar_exact", stats.mcnemar(per_ex[x] > 0, per_ex[y_] > 0)["p_value"]
            else:
                test, p = "paired_bootstrap", boot_p(d)
            pairs.append({"task": task, "metric": MAIN[task], "model_a": x, "model_b": y_, "test": test,
                          "value_a": round(point[x], 4), "value_b": round(point[y_], 4),
                          "diff": round(point[x] - point[y_], 4), "ci_lo": round(float(lo), 4),
                          "ci_hi": round(float(hi), 4), "p_value": p, "n": n})
        for r, adj in zip(pairs, holm([r["p_value"] for r in pairs])):
            r["p_holm"], r["significant"] = adj, adj < ALPHA
        rows_out += pairs
        ranked = sorted(models, key=lambda m: -point[m])
        best = ranked[0]
        sig = {frozenset((r["model_a"], r["model_b"])): r["significant"] for r in pairs}
        tied = [m for m in ranked[1:] if not sig[frozenset((best, m))]]
        md += [f"\n## {task} ({MAIN[task]}, n={n})\n",
               f"- Best: **{best}** ({point[best]:.3f})",
               "- Not significantly different from the best: "
               + (", ".join(f"{m} ({point[m]:.3f})" for m in tied) if tied else "none"),
               f"- Significant pairs: {sum(r['significant'] for r in pairs)} of {len(pairs)}\n",
               "| rank | model | score | significantly better than |", "|---|---|---|---|"]
        for i, m in enumerate(ranked, 1):
            below = [o for o in ranked[i:] if sig[frozenset((m, o))]]
            md.append(f"| {i} | {m} | {point[m]:.3f} | {len(below)} of {len(ranked) - i} lower-ranked |")
        print(f"{task}: best {best} {point[best]:.3f}; tied with best: {tied or 'none'}", flush=True)

    out = root / "significance.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    name = "bench_significance.md" if a.subset == "test_eval" else f"bench_significance_{a.subset}.md"
    (ROOT / "docs" / name).write_text("\n".join(md) + "\n")
    print("wrote", out, "and docs/" + name)


if __name__ == "__main__":
    main()

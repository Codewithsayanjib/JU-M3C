"""Score the order-sensitivity runs (checklist A2) against the original test outputs.

  python scripts/experiments/score_reliability.py

Multiple choice: the 500 questions in four orders (the original run = rotation 0, plus rotations 1-3),
so each question is asked once with its correct option in each position.
  acc_rot0..3        accuracy per order
  acc_pos_A..D       accuracy when the correct option is in that position (pooled over the four orders)
  rstd               standard deviation of acc_pos_A..D, in percentage points
  same_answer        share of questions where the model picks the same option text in all four orders
  all_four_correct   share of questions answered correctly in all four orders
Classification: original label order against one shuffled order, on the same examples.
  macro_f1 / accuracy for both, their difference with a paired bootstrap 95% interval, and
  same_label = share of examples given the same label under both orders.
Writes results/bench/reliability/summary_reliability.csv and docs/bench_reliability.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from run_reliability import CLS_TASKS, options_of  # noqa: E402
from score_bench import get_refs  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from analyze_bench import md as table  # noqa: E402
from ju_m3c.benchmark.bench_parse import parse  # noqa: E402

REL = ROOT / "results" / "bench" / "reliability"
TEST = ROOT / "results" / "bench" / "sealed_test_eval"
SEED, N_BOOT = 42, 1000


def rows(path):
    return {r["example_id"]: r for r in map(json.loads, open(path))}


def chosen(task, raw, ex):
    p = parse(task, raw, ex)
    return p["parsed"] if p["lenient"] else None


def main():
    mcq = get_refs("diagnosis_mcq", "test_eval")
    models = sorted(p.name for p in REL.iterdir() if p.is_dir())
    out, md = [], ["# Order sensitivity (test evaluation subsets)\n"]
    rng = np.random.default_rng(SEED)
    for m in models:
        rec = {"model": m}
        # ---- multiple choice
        rot = {r: rows(REL / m / f"mcq_rot{r}.jsonl") for r in (1, 2, 3)}
        ids = sorted(rot[1])
        orig = rows(TEST / m / "diagnosis_mcq.jsonl")
        text = {i: [] for i in ids}
        correct = {i: [] for i in ids}
        bypos = {l: [] for l in "ABCD"}
        for r in (0, 1, 2, 3):
            acc = []
            for i in ids:
                if r == 0:
                    ex = {**mcq[i], "options": options_of(mcq[i])}; raw = orig[i]["raw_output"]
                else:
                    ex = {**mcq[i], "options": rot[r][i]["options"], "target": rot[r][i]["target"]}; raw = rot[r][i]["raw_output"]
                c = chosen("diagnosis_mcq", raw, ex)
                ok = c == ex["target"]
                acc.append(ok); bypos[ex["target"]].append(ok); correct[i].append(ok)
                text[i].append(ex["options"]["ABCD".index(c)] if c in ("A", "B", "C", "D") else None)
            rec[f"acc_rot{r}"] = float(np.mean(acc))
        for l in "ABCD":
            rec[f"acc_pos_{l}"] = float(np.mean(bypos[l]))
        rec["rstd"] = float(np.std([100 * rec[f"acc_pos_{l}"] for l in "ABCD"]))
        rec["acc_mean"] = float(np.mean([rec[f"acc_rot{r}"] for r in range(4)]))
        rec["same_answer"] = float(np.mean([len(set(t)) == 1 and t[0] is not None for t in text.values()]))
        rec["all_four_correct"] = float(np.mean([all(c) for c in correct.values()]))
        # ---- classification: original against shuffled label order
        for t in CLS_TASKS:
            refs = get_refs(t, "test_eval")
            sh = rows(REL / m / f"{t}_shuffled.jsonl")
            o = rows(TEST / m / f"{t}.jsonl")
            ids_t = sorted(sh)
            y = np.array([refs[i]["target"] for i in ids_t], dtype=object)
            a = np.array([chosen(t, o[i]["raw_output"], refs[i]) or "<unparsed>" for i in ids_t], dtype=object)
            b = np.array([chosen(t, sh[i]["raw_output"], refs[i]) or "<unparsed>" for i in ids_t], dtype=object)
            labels = sorted(set(y))
            f = lambda idx, p: f1_score(y[idx], p[idx], labels=labels, average="macro", zero_division=0)
            full = np.arange(len(y))
            fa, fb = f(full, a), f(full, b)
            diffs = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(y), len(y))
                diffs.append(f(idx, b) - f(idx, a))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rec.update({f"{t}_f1_orig": fa, f"{t}_f1_shuffled": fb, f"{t}_f1_diff": fb - fa, f"{t}_diff_lo": lo,
                        f"{t}_diff_hi": hi, f"{t}_acc_orig": float(np.mean(a == y)), f"{t}_acc_shuffled": float(np.mean(b == y)),
                        f"{t}_same_label": float(np.mean(a == b))})
        out.append(rec)
        print("scored", m, flush=True)
    df = pd.DataFrame(out).set_index("model")
    df.to_csv(REL / "summary_reliability.csv")
    order = df.rstd.sort_values().index
    mc = df.loc[order, ["acc_rot0", "acc_rot1", "acc_rot2", "acc_rot3", "acc_pos_A", "acc_pos_B", "acc_pos_C", "acc_pos_D",
                         "rstd", "same_answer", "all_four_correct"]]
    md.append("## Multiple-choice diagnosis, 500 questions in four orders\n")
    md.append(table(mc))
    for t in CLS_TASKS:
        x = df.loc[order, [f"{t}_f1_orig", f"{t}_f1_shuffled", f"{t}_f1_diff", f"{t}_diff_lo", f"{t}_diff_hi", f"{t}_same_label"]]
        x.columns = ["f1_original", "f1_shuffled", "difference", "ci_lo", "ci_hi", "same_label"]
        md.append(f"\n\n## {t}: original against shuffled label order\n")
        md.append(table(x))
    (ROOT / "docs" / "bench_reliability.md").write_text("\n".join(md) + "\n")
    print("wrote", REL / "summary_reliability.csv", "and docs/bench_reliability.md")


if __name__ == "__main__":
    main()

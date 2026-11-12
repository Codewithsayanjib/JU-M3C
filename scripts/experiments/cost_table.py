"""Cost of running the benchmark on a laptop (2e): quality against memory and time, no new runs.

Text models: sealed test-eval summary (main metric per task, median latency per task, peak memory).
Vision models: per-item MM test outputs (accuracy, latency_s, peak_memory_gb).
Writes reports/cost.json and reports/cost.csv; marks the Pareto-optimal models (no other model is
both smaller in peak memory and better in mean score, or both faster and better).

  python scripts/experiments/cost_table.py --data $JUMC_DATA \
      --mm $JUMC_MV/full/results --out $JUMC_MV/reports/cost.json
"""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path


def pareto(rows, x, y):
    return {r["model"] for r in rows
            if not any(o[x] <= r[x] and o[y] >= r[y] and (o[x] < r[x] or o[y] > r[y]) for o in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--mm", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summ = list(csv.DictReader(open(Path(a.data).expanduser() / "outputs/sealed_test_eval/summary.csv")))
    by = defaultdict(lambda: {"main": {}, "lat": {}, "n": {}})
    for r in summ:
        m = by[r["model"]]
        if r["main"].strip() == "True":
            m["main"][r["task"]] = float(r["value"]); m["n"][r["task"]] = int(float(r["n"]))
        elif r["metric"] == "latency_s_median":
            m["lat"][r["task"]] = float(r["value"])
        elif r["metric"] == "peak_memory_gb":
            m["mem"] = float(r["value"])
    tasks = sorted({t for m in by.values() for t in m["main"]})
    # mean rank per task (1 = best)
    ranks = defaultdict(list)
    for t in tasks:
        order = sorted(by, key=lambda k: -by[k]["main"].get(t, -1))
        for i, k in enumerate(order):
            ranks[k].append(i + 1)
    text = []
    for k, m in by.items():
        sc = 100 * statistics.mean(m["main"][t] for t in tasks)
        lat = statistics.mean(m["lat"][t] for t in tasks)
        hours = sum(m["lat"][t] * m["n"][t] for t in tasks) / 3600
        text.append({"model": k, "mean_main_score": round(sc, 1), "mean_rank": round(statistics.mean(ranks[k]), 2),
                     "peak_memory_gb": round(m["mem"], 2), "mean_median_latency_s": round(lat, 2),
                     "test_eval_hours_est": round(hours, 1), "score_per_gb": round(sc / m["mem"], 1),
                     "items_per_hour": round(3600 / lat)})
    pm, pl = pareto(text, "peak_memory_gb", "mean_main_score"), pareto(text, "mean_median_latency_s", "mean_main_score")
    for r in text:
        r["pareto_memory"] = r["model"] in pm; r["pareto_latency"] = r["model"] in pl
    text.sort(key=lambda r: -r["mean_main_score"])

    vision = []
    for d in sorted(Path(a.mm).iterdir()):
        row = {"model": d.name}
        for task, f in [("mm1", "mm1_all_test.jsonl"), ("mm2", "mm2_both_test.jsonl")]:
            p = d / f
            if not p.exists():
                continue
            rs = [json.loads(l) for l in open(p)]
            if len(rs) < 100:  # unfinished run (qwen25vl_7b paused)
                continue
            row[f"{task}_acc"] = round(100 * statistics.mean(r["correct"] for r in rs), 1)
            lat = [r["latency_s"] for r in rs if "latency_s" in r]
            if lat:
                row[f"{task}_median_latency_s"] = round(statistics.median(lat), 2)
                row["peak_memory_gb"] = round(max(row.get("peak_memory_gb", 0), max(r["peak_memory_gb"] for r in rs)), 2)
        if "mm1_acc" in row:
            if "peak_memory_gb" in row:
                row["mm1_acc_per_gb"] = round(row["mm1_acc"] / row["peak_memory_gb"], 1)
            vision.append(row)

    res = {"note": "Text: mean of the main metric over %d tasks (0-100), sealed test-eval, MacBook Air M4 16 GB, MLX 4-bit. "
                   "Vision: MM test, uncleaned accuracy (cleaned figures are in mm_analysis.json). BiomedCLIP latency not logged." % len(tasks),
           "tasks": tasks, "text": text, "vision": vision}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(text[0])); w.writeheader(); w.writerows(text)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

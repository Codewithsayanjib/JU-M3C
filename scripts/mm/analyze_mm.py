"""MM-1 / MM-2 test analysis: accuracy with 95% CIs, paired McNemar tests (Holm), per-modality
accuracy, letter bias, BiomedCLIP before/after its pretraining cutoff, latency and memory.

Main numbers exclude items that use an image from a cross-split duplicate pair (L4);
the full-set numbers are reported next to them.

Usage:
  python scripts/mm/analyze_mm.py --mm $JUMC_MV/full \
      --pairs $JUMC_MV/reports/mm_leakage_full.pairs.jsonl --out reports/mm_analysis.json
"""
import argparse, json, random, statistics
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import binomtest, fisher_exact

MODELS = ["biomedclip", "qwen25vl_3b", "internvl3_2b"]
CONDS = ["text", "image", "both", "options"]


def ci(xs, n_boot=2000, seed=42):
    rng = random.Random(seed)
    m = [sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot)]
    m.sort()
    return [round(100 * m[int(.025 * n_boot)], 1), round(100 * m[int(.975 * n_boot)], 1)]


def acc(xs):
    return {"n": len(xs), "acc": round(100 * sum(xs) / len(xs), 1), "ci95": ci(xs)}


def mcnemar(a, b):
    b01 = sum(x and not y for x, y in zip(a, b)); b10 = sum(y and not x for x, y in zip(a, b))
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"a_only": b01, "b_only": b10, "diff_pts": round(100 * (sum(a) - sum(b)) / len(a), 1), "p": p}


def holm(tests):
    order = sorted(tests, key=lambda k: tests[k]["p"])
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * tests[k]["p"]))
        tests[k]["p_holm"] = run
        tests[k]["p"] = float(f"{tests[k]['p']:.3g}"); tests[k]["p_holm"] = float(f"{run:.3g}")
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mm = Path(a.mm)
    bad = set()
    for l in open(a.pairs):
        p = json.loads(l)
        if p["split_a"] != p["split_b"]:
            bad |= {p["image_a"], p["image_b"]}

    items = {t: {json.loads(l)["item_id"]: json.loads(l) for l in open(mm / f"items/{t}_test.jsonl")} for t in ("mm1", "mm2")}
    excl = {t: {i for i, it in items[t].items() if bad & set(it["images"])} for t in items}
    res = {"excluded_items": {t: len(v) for t, v in excl.items()}, "mm1": {}, "mm2": {}, "tests": {}}

    runs = {}
    for m in MODELS:
        for t, conds in (("mm1", ["all"]), ("mm2", CONDS)):
            for c in conds:
                f = mm / "results" / m / f"{t}_{c}_test.jsonl"
                if f.exists():
                    runs[(m, t, c)] = {r["item_id"]: r for r in map(json.loads, open(f))}

    for (m, t, c), rs in runs.items():
        ids = sorted(i for i in rs if i not in excl[t])
        xs = [bool(rs[i]["correct"]) for i in ids]
        e = {"clean": acc(xs), "all": acc([bool(r["correct"]) for r in rs.values()]),
             "parse_fail": sum(rs[i].get("parsed_lenient") is None for i in ids),
             "pred_letters": dict(Counter(rs[i].get("parsed_lenient") for i in ids)),
             "gold_letters": dict(Counter(rs[i]["answer"] for i in ids))}
        lat = [rs[i]["latency_s"] for i in ids if "latency_s" in rs[i]]
        if lat:
            e["median_latency_s"] = round(statistics.median(lat), 2)
            e["peak_memory_gb"] = round(max(rs[i]["peak_memory_gb"] for i in ids), 2)
        if t == "mm1":
            by = defaultdict(list)
            for i in ids:
                by[items[t][i]["modality"]].append(bool(rs[i]["correct"]))
            e["by_modality"] = {k: {"n": len(v), "acc": round(100 * sum(v) / len(v), 1)} for k, v in sorted(by.items())}
            yr = lambda i: items[t][i].get("year") or 0
            old = [bool(rs[i]["correct"]) for i in ids if 0 < yr(i) <= 2021]
            new = [bool(rs[i]["correct"]) for i in ids if yr(i) >= 2023]
            e["year_le2021"], e["year_ge2023"] = acc(old), acc(new)
            e["year_fisher_p"] = float(f"{fisher_exact([[sum(old), len(old) - sum(old)], [sum(new), len(new) - sum(new)]])[1]:.3g}")
        res[t][f"{m}/{c}"] = e

    tests = {}
    for m in ("qwen25vl_3b", "internvl3_2b"):
        R = {c: runs[(m, "mm2", c)] for c in CONDS}
        ids = sorted(i for i in R["text"] if i not in excl["mm2"])
        v = {c: [bool(R[c][i]["correct"]) for i in ids] for c in CONDS}
        for x, y in (("both", "text"), ("image", "options"), ("text", "options")):
            tests[f"{m}: {x} vs {y}"] = mcnemar(v[x], v[y])
        r1 = runs[(m, "mm1", "all")]; rc = runs[("biomedclip", "mm1", "all")]
        ids1 = sorted(i for i in r1 if i not in excl["mm1"])
        tests[f"mm1: biomedclip vs {m}"] = mcnemar([bool(rc[i]["correct"]) for i in ids1], [bool(r1[i]["correct"]) for i in ids1])
        tests[f"mm1: {m} vs chance"] = {"p": binomtest(sum(bool(r1[i]["correct"]) for i in ids1), len(ids1), 0.25).pvalue}
    res["tests"] = holm(tests)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print("excluded", res["excluded_items"])
    for t in ("mm1", "mm2"):
        for k, e in res[t].items():
            print(t, k, e["clean"], "all", e["all"]["acc"], "pf", e["parse_fail"], e.get("median_latency_s", ""), e.get("peak_memory_gb", ""))
    for k, e in res["mm1"].items():
        print(k, "<=2021", e["year_le2021"], ">=2023", e["year_ge2023"], "p", e["year_fisher_p"])
        print("  ", {m: v["acc"] for m, v in e["by_modality"].items()})
    for k, v in res["tests"].items():
        print(k, v)


if __name__ == "__main__":
    main()

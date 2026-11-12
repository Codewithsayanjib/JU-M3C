"""Generalization over time (2b): are models worse on cases published after their training data?

Re-scores existing sealed test outputs (no new runs). Publication year comes from the article's PMCID
(full/pubyear.jsonl); documents without a PMCID (MTSamples, dialogue sources) have no date and drop out.
Groups: older = published <= 2021, newer = >= 2023 (2022 left out as a buffer; same cut as the
BiomedCLIP check). The cut is fixed for all models: their knowledge cutoffs are not all public.
Per model x task: mean per-example score in each group, bootstrap 95% CI of (newer - older),
permutation p-value, Holm correction across all comparisons.

  python scripts/experiments/temporal.py --data $JUMC_DATA \
      --mm $JUMC_MV/full --out $JUMC_MV/reports/temporal.json
"""
import argparse, csv, gzip, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OLD, NEW = 2021, 2023


def diff_stats(old, new, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    old, new = np.asarray(old, float), np.asarray(new, float)
    d = new.mean() - old.mean()
    bo = old[rng.integers(0, len(old), (n, len(old)))].mean(1); bn = new[rng.integers(0, len(new), (n, len(new)))].mean(1)
    ci = np.percentile(bn - bo, [2.5, 97.5])
    allv = np.concatenate([old, new]); k = 0
    for _ in range(n):
        rng.shuffle(allv)
        k += abs(allv[len(old):].mean() - allv[:len(old)].mean()) >= abs(d) - 1e-12
    return round(100 * d, 1), [round(100 * ci[0], 1), round(100 * ci[1], 1)], (k + 1) / (n + 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i])); out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--mm", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, mm = Path(a.data).expanduser(), Path(a.mm)
    year = {json.loads(l)["pmcid"]: json.loads(l)["year"] for l in open(mm / "pubyear.jsonl")}
    pmc = {}
    for l in gzip.open(data / "links/doc_index.jsonl.gz", "rt"):
        d = json.loads(l); pmc[d["doc_id"]] = d["pmcid"]

    def yr_of_doc(doc_id):
        p = pmc.get(doc_id)
        if p and not str(p).startswith("PMC"):
            p = "PMC" + str(p)
        return year.get(p)

    rows = defaultdict(list); cov = Counter()
    for r in csv.DictReader(gzip.open(data / "outputs/sealed_test_eval/analysis/per_example.csv.gz", "rt")):
        if r["score"] == "":
            continue
        doc = r["example_id"].split(":", 1)[1]
        y = yr_of_doc(doc)
        cov[(r["task"], y is not None)] += 1
        if y:
            rows[(r["model"], r["task"])].append((y, float(r["score"])))
    items = {t: {json.loads(l)["item_id"]: json.loads(l) for l in open(mm / f"items/{t}_test.jsonl")} for t in ("mm1", "mm2")}
    for m in sorted(p.name for p in (mm / "results").iterdir()):
        for t, f in (("mm1", "mm1_all_test.jsonl"), ("mm2", "mm2_both_test.jsonl")):
            p = mm / "results" / m / f
            if not p.exists():
                continue
            rs = [json.loads(l) for l in open(p)]
            if len(rs) < 100:
                continue
            for r in rs:
                y = items[t].get(r["item_id"], {}).get("year")
                if y:
                    rows[(m, t)].append((y, float(bool(r["correct"]))))

    res, keys, ps = {}, [], []
    for (m, t), v in sorted(rows.items()):
        old = [s for y, s in v if y <= OLD]; new = [s for y, s in v if y >= NEW]
        if len(old) < 30 or len(new) < 30:
            res[f"{m}/{t}"] = {"n_old": len(old), "n_new": len(new), "skipped": "fewer than 30 in a group"}
            continue
        d, ci, p = diff_stats(old, new)
        res[f"{m}/{t}"] = {"n_old": len(old), "n_new": len(new), "old": round(100 * np.mean(old), 1),
                           "new": round(100 * np.mean(new), 1), "diff_new_minus_old": d, "ci95": ci, "p": round(p, 4)}
        keys.append(f"{m}/{t}"); ps.append(p)
    for k, q in zip(keys, holm(ps)):
        res[k]["p_holm"] = round(q, 4)
    out = {"cut": {"older": f"<= {OLD}", "newer": f">= {NEW}"},
           "text_year_coverage": {f"{t}|{'dated' if h else 'undated'}": n for (t, h), n in sorted(cov.items())},
           "results": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, v)
    print(out["text_year_coverage"])


if __name__ == "__main__":
    main()

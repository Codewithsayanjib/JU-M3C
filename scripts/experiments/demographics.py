"""Patient subgroups (2d): who the corpus covers by sex and age, and whether models are less accurate
for some groups. No new model runs.

Sex and age per document:
  PMC-Patients documents  -> the dataset's own metadata (age, gender);
  all other documents     -> a rule on the opening of the case text ("a 45-year-old woman", "8-month-old boy").
The rule is validated against the PMC-Patients metadata (agreement reported), and 50 rule outputs from
other sources are written out for hand-checking.
Age bands: child < 18, adult 18-64, older >= 65.
Subgroup scores: sealed test-eval per-example scores (text) and MM test items (vision, via case_text).
Gaps: female - male, older - adult, child - adult; bootstrap 95% CI, permutation p, Holm over all gaps.

  python scripts/experiments/demographics.py --data $JUMC_DATA \
      --mm $JUMC_MV/full \
      --pmcp $JUMC_MV/external/pmc_patients/PMC-Patients.csv \
      --out $JUMC_MV/reports/demographics.json
"""
import argparse, ast, csv, gzip, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(10 ** 9)
FEM = r"woman|women|female|girl|lady|mother|pregnant|gravida|primigravida|multigravida|she|her"
MAL = r"man|men|male|boy|gentleman|father|he|his"
SEX_RE = re.compile(rf"\b(?:(?P<f>{FEM})|(?P<m>{MAL}))\b", re.I)
AGE_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)[\s-]*(year|yr|month|week|day)s?[\s-]*old\b|\baged?\s+(\d{1,3})\b|\b(\d{1,3})\s*(?:yo|y/o|y\.o\.)\b", re.I)


def rule(text):
    head = (text or "")[:600]
    sex = None
    m = SEX_RE.search(head)
    if m:
        sex = "F" if m.group("f") else "M"
    age = None
    a = AGE_RE.search(head)
    if a:
        if a.group(1):
            v = float(a.group(1)); unit = a.group(2).lower()
            age = v if unit in ("year", "yr") else v / 12 if unit == "month" else v / 52 if unit == "week" else v / 365
        else:
            age = float(a.group(3) or a.group(4))
        if age > 110:
            age = None
    return sex, age


def band(age):
    if age is None:
        return None
    return "child" if age < 18 else "adult" if age < 65 else "older"


def gap(x, y, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x.mean() - y.mean()
    bx = x[rng.integers(0, len(x), (n, len(x)))].mean(1); by = y[rng.integers(0, len(y), (n, len(y)))].mean(1)
    ci = np.percentile(bx - by, [2.5, 97.5])
    allv = np.concatenate([x, y]); k = 0
    for _ in range(n):
        rng.shuffle(allv)
        k += abs(allv[:len(x)].mean() - allv[len(x):].mean()) >= abs(d) - 1e-12
    return round(100 * d, 1), [round(100 * ci[0], 1), round(100 * ci[1], 1)], (k + 1) / (n + 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i])); out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "mm", "pmcp", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    data, mm = Path(a.data).expanduser(), Path(a.mm)

    meta = {}
    for r in csv.DictReader(open(a.pmcp)):
        try:
            ag = ast.literal_eval(r["age"]) if r["age"] else []
        except (ValueError, SyntaxError):
            ag = []
        age = None
        if ag:
            v, u = ag[0]
            age = v if u == "year" else v / 12 if u == "month" else v / 52 if u == "week" else v / 365 if u == "day" else None
        meta[r["patient_uid"]] = (r["gender"] if r["gender"] in ("F", "M") else None, age)

    demo, src, agree, sample_pool = {}, {}, Counter(), []
    for l in gzip.open(data / "corpus/records.jsonl.gz", "rt"):
        r = json.loads(l)
        s, ag = rule(r["text"])
        if r["source"] == "pmc_patients":
            ms, mg = meta.get(r["source_id"], (None, None))
            if ms and s:
                agree["sex_n"] += 1; agree["sex_ok"] += ms == s
            if mg is not None and ag is not None:
                agree["age_n"] += 1; agree["age_band_ok"] += band(mg) == band(ag)
            if ms or mg is not None:
                s, ag = ms, mg
        else:
            sample_pool.append((r["doc_id"], r["source"], s, ag, r["text"][:250]))
        demo[r["doc_id"]] = (s, band(ag)); src[r["doc_id"]] = r["source"]

    cov = defaultdict(Counter)
    for d, (s, b) in demo.items():
        for key in (src[d], "all"):
            cov[key][f"sex_{s or 'unknown'}"] += 1; cov[key][f"age_{b or 'unknown'}"] += 1; cov[key]["n"] += 1

    # per-example scores
    rows = defaultdict(list)
    for r in csv.DictReader(gzip.open(data / "outputs/sealed_test_eval/analysis/per_example.csv.gz", "rt")):
        if r["score"] == "":
            continue
        doc = r["example_id"].split(":", 1)[1]
        if doc in demo:
            rows[(r["model"], r["task"])].append((*demo[doc], float(r["score"])))
    items = {}
    for t in ("mm1", "mm2"):
        for l in open(mm / f"items/{t}_test.jsonl"):
            it = json.loads(l)
            d = demo.get(it.get("doc_id")) if it.get("doc_id") else None
            if d is None:
                s, ag = rule(it["case_text"]); d = (s, band(ag))
            items[it["item_id"]] = d
    for p in sorted((mm / "results").iterdir()):
        for t, f in (("mm1", "mm1_all_test.jsonl"), ("mm2", "mm2_both_test.jsonl")):
            fp = p / f
            if fp.exists():
                rs = [json.loads(l) for l in open(fp)]
                if len(rs) >= 100:
                    for r in rs:
                        rows[(p.name, t)].append((*items[r["item_id"]], float(bool(r["correct"]))))

    res, keys, ps = {}, [], []
    for (m, t), v in sorted(rows.items()):
        e = {}
        for name, xs, ys in (("female_minus_male", [z for s, b, z in v if s == "F"], [z for s, b, z in v if s == "M"]),
                             ("older_minus_adult", [z for s, b, z in v if b == "older"], [z for s, b, z in v if b == "adult"]),
                             ("child_minus_adult", [z for s, b, z in v if b == "child"], [z for s, b, z in v if b == "adult"])):
            if len(xs) >= 30 and len(ys) >= 30:
                d, ci, pv = gap(xs, ys)
                e[name] = {"n": [len(xs), len(ys)], "gap": d, "ci95": ci, "p": round(pv, 4)}
                keys.append((f"{m}/{t}", name)); ps.append(pv)
        if e:
            res[f"{m}/{t}"] = e
    for (k, name), q in zip(keys, holm(ps)):
        res[k][name]["p_holm"] = round(q, 4)

    random.seed(42)
    sample = random.sample(sample_pool, 50)
    out = {"rule_vs_pmc_patients_metadata": {"sex_agreement": round(agree["sex_ok"] / agree["sex_n"], 4), "sex_n": agree["sex_n"],
                                             "age_band_agreement": round(agree["age_band_ok"] / agree["age_n"], 4), "age_n": agree["age_n"]},
           "coverage": {k: dict(v) for k, v in cov.items()}, "gaps": res, "n_tests": len(ps),
           "n_significant_holm": sum(q < 0.05 for q in holm(ps))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    with open(Path(a.out).with_suffix(".handcheck.jsonl"), "w") as f:
        for d, s_, sx, ag, tx in sample:
            f.write(json.dumps({"doc_id": d, "source": s_, "sex": sx, "age": ag, "band": band(ag), "text": tx}) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "gaps"}, indent=1))
    for k, v in res.items():
        for name, g in v.items():
            if g["p_holm"] < 0.05:
                print("SIG", k, name, g)


if __name__ == "__main__":
    main()

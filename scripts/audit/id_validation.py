"""Validate the text near-duplicate detector against ground truth that needs no annotators.

Several sources come from PubMed Central and carry the article ID (PMCID). Two documents from
different sources whose articles contain a single case each, and which share a PMCID, are the same
patient case (known positives). Pairs from different PMCIDs are, with rare exceptions, different cases.

Reports
  recall    : share of known same-case cross-source pairs that the near-duplicate detector flags
              (per source pair), and that the article layer links (100% by construction)
  precision : share of detector pairs between PMCID-bearing documents whose PMCIDs match
              (a lower bound: a case republished in another article counts as a false positive)
  threshold sensitivity of both on the Jaccard score

  python scripts/experiments/id_validation.py --data $JUMC_DATA --out reports/id_validation.json
"""
import argparse, gzip, json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def norm(p):
    if p is None:
        return None
    p = str(p)
    return p if p.startswith("PMC") else "PMC" + p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); data = Path(a.data).expanduser()
    pm, src = {}, {}
    for l in gzip.open(data / "links/doc_index.jsonl.gz", "rt"):
        d = json.loads(l); pm[d["doc_id"]] = norm(d["pmcid"]); src[d["doc_id"]] = d["source"]
    by_art = defaultdict(lambda: defaultdict(list))
    for d, p in pm.items():
        if p:
            by_art[p][src[d]].append(d)
    # known positives: article has exactly one document in each of two sources
    pos = set()
    for p, ss in by_art.items():
        singles = [v[0] for s, v in ss.items() if len(v) == 1]
        for x, y in combinations(sorted(singles), 2):
            pos.add((x, y))
    pairs = []
    for l in gzip.open(data / "links/near_dup_pairs.jsonl.gz", "rt"):
        r = json.loads(l); x, y = sorted((r["a"], r["b"])); pairs.append((x, y, r["jaccard"]))
    res = {"known_same_case_pairs": len(pos), "thresholds": {}}
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        det = {(x, y) for x, y, j in pairs if j >= t}
        rec = Counter(); tot = Counter()
        for x, y in pos:
            k = "|".join(sorted((src[x], src[y]))); tot[k] += 1; rec[k] += (x, y) in det
        both = [(x, y) for x, y in det if pm.get(x) and pm.get(y) and src[x] != src[y]]
        same = sum(pm[x] == pm[y] for x, y in both)
        res["thresholds"][t] = {
            "detected_pairs": len(det),
            "recall_by_source_pair": {k: {"n": tot[k], "recall": round(rec[k] / tot[k], 4)} for k in sorted(tot)},
            "recall_overall": round(sum(rec.values()) / max(1, sum(tot.values())), 4),
            "precision_lower_bound": {"n": len(both), "same_pmcid": same, "precision": round(same / max(1, len(both)), 4)}}
        print(t, res["thresholds"][t]["recall_overall"], res["thresholds"][t]["precision_lower_bound"], flush=True)
    res["min_jaccard_in_file"] = min(j for _, _, j in pairs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["thresholds"][min(res["thresholds"])]["recall_by_source_pair"], indent=1))


if __name__ == "__main__":
    main()

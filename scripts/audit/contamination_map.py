"""Cross-benchmark contamination map: for every published benchmark A with a native test split, the share
of A's test documents that also appear (same article or near-duplicate text) in the training data of
benchmark B, using each benchmark's own published splits. Sources without a split count as training data.

Also reports, per cell, how many of the links are found only by the article layer (reworded copies that a
text-duplicate check misses).

  python scripts/experiments/contamination_map.py --data $JUMC_DATA --out reports/contamination_map.json
"""
import argparse, gzip, json
from collections import defaultdict
from pathlib import Path


def native(ns):
    n = (ns or "").lower()
    return "test" if n.startswith("test") else "val" if n.startswith(("val", "dev", "valid")) else "train"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--image-pairs", default=None)
    a = ap.parse_args(); data = Path(a.data).expanduser()
    src, split, pmc = {}, {}, {}
    for l in gzip.open(data / "links/doc_index.jsonl.gz", "rt"):
        d = json.loads(l)
        src[d["doc_id"]] = d["source"]; split[d["doc_id"]] = native(d["native_split"]); pmc[d["doc_id"]] = d["pmcid"]
    art = defaultdict(set)
    for d, p in pmc.items():
        if p:
            art[str(p).removeprefix("PMC")].add(d)
    txt = defaultdict(set)
    for l in gzip.open(data / "links/near_dup_pairs.jsonl.gz", "rt"):
        r = json.loads(l); txt[r["a"]].add(r["b"]); txt[r["b"]].add(r["a"])
    sources = sorted(set(src.values()))
    tests = {s: [d for d in src if src[d] == s and split[d] == "test"] for s in sources}
    tests = {s: v for s, v in tests.items() if v}
    res = {}
    for A, docs in tests.items():
        row = {}
        for B in sources:
            hit = hit_txt = hit_art_only = 0
            for d in docs:
                ta = {x for x in txt[d] if src[x] == B and split[x] != "test"}
                aa = {x for x in (art[str(pmc[d]).removeprefix("PMC")] if pmc[d] else ()) if x != d and src[x] == B and split[x] != "test"}
                if ta or aa:
                    hit += 1; hit_txt += bool(ta); hit_art_only += bool(aa) and not ta
            if hit:
                row[B] = {"n": hit, "pct": round(100 * hit / len(docs), 1), "text_layer": hit_txt, "article_layer_only": hit_art_only}
        anyhit = 0
        for d in docs:
            nb = {x for x in txt[d] if split[x] != "test"}
            if pmc[d]:
                nb |= {x for x in art[str(pmc[d]).removeprefix("PMC")] if x != d and split[x] != "test"}
            anyhit += bool(nb)
        res[A] = {"n_test": len(docs), "any_benchmark": {"n": anyhit, "pct": round(100 * anyhit / len(docs), 1)}, "by_training_source": row}
        print(A, res[A]["n_test"], res[A]["any_benchmark"], {k: v["pct"] for k, v in row.items()}, flush=True)
    # multi-hop leakage: test documents connected to native training data only through a chain of links
    # that switches layer (article, text, image), with no direct link in any single layer
    par = {}
    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    def u(x, y):
        par[f(x)] = f(y)
    for g in art.values():
        g = list(g)
        for x in g[1:]:
            u(g[0], x)
    for d, nb in txt.items():
        for x in nb:
            u(d, x)
    by_art_img = defaultdict(set)
    if a.image_pairs:
        for l in open(a.image_pairs):
            r = json.loads(l)
            x, y = r["image_a"].split("_")[0].removeprefix("PMC"), r["image_b"].split("_")[0].removeprefix("PMC")
            if x != y and art.get(x) and art.get(y):
                u(next(iter(art[x])), next(iter(art[y]))); by_art_img[x].add(y); by_art_img[y].add(x)
    comp_train = defaultdict(int)
    for d in src:
        if split[d] != "test":
            comp_train[f(d)] += 1
    mh = {}
    for A, docs in tests.items():
        direct = 0; chain = 0
        for d in docs:
            p = str(pmc[d]).removeprefix("PMC") if pmc[d] else None
            nb = {x for x in txt[d] if split[x] != "test"}
            if p:
                nb |= {x for x in art[p] if x != d and split[x] != "test"}
                nb |= {x for q in by_art_img[p] for x in art[q] if split[x] != "test"}
            if nb:
                direct += 1
            elif comp_train[f(d)]:
                chain += 1
        mh[A] = {"direct": direct, "chain_only": chain}
    res["_multi_hop"] = mh
    print("multi-hop", mh)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Experiment A: the multi-layer link graph of JU-M3C.

Nodes are documents. Edge layers:
  article  - documents from different sources that share a PMCID (same article)
  text     - text near-duplicate pairs (links/near_dup_pairs.jsonl.gz)
  image    - confirmed cross-article duplicate images (L4 pairs), mapped to the articles' documents
Reports per-layer edge counts, connected components of the union graph, the source-overlap
network, and how many edges each split strategy cuts (cross-split edges = leakage paths):
  ours (article_cluster split), each source's native split (sources without one count as train), random document split, random PMCID split.

Usage:
  python scripts/mm/network_links.py --data $JUMC_DATA \
      --pairs $JUMC_MV/reports/mm_leakage_full.pairs.jsonl --out reports/network.json
"""
import argparse, gzip, json, random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = Path(a.data).expanduser()
    docs = [json.loads(l) for l in gzip.open(data / "links/doc_index.jsonl.gz", "rt")]
    idx = {d["doc_id"]: i for i, d in enumerate(docs)}
    split = {json.loads(l)["doc_id"]: json.loads(l)["split"] for l in gzip.open(data / "links/splits.jsonl.gz", "rt")}
    sp = [split.get(d["doc_id"]) for d in docs]
    by_pmc = defaultdict(list)
    for i, d in enumerate(docs):
        if d["pmcid"]:
            by_pmc[d["pmcid"]].append(i)

    layers = {"article": set(), "text": set(), "image": set()}
    for ids in by_pmc.values():
        for x, y in combinations(ids, 2):
            if docs[x]["source"] != docs[y]["source"]:
                layers["article"].add((min(x, y), max(x, y)))
    for l in gzip.open(data / "links/near_dup_pairs.jsonl.gz", "rt"):
        p = json.loads(l)
        if p["a"] in idx and p["b"] in idx:
            x, y = idx[p["a"]], idx[p["b"]]
            layers["text"].add((min(x, y), max(x, y)))
    for l in open(a.pairs):
        p = json.loads(l)
        pa, pb = p["image_a"].split("_")[0], p["image_b"].split("_")[0]
        for x in by_pmc.get(pa, []):
            for y in by_pmc.get(pb, []):
                layers["image"].add((min(x, y), max(x, y)))

    dsu = DSU(len(docs))
    for E in layers.values():
        for x, y in E:
            dsu.union(x, y)
    comp = defaultdict(list)
    for i in range(len(docs)):
        comp[dsu.find(i)].append(i)
    sizes = Counter(len(v) for v in comp.values())
    multi = [v for v in comp.values() if len(v) > 1]

    src_nodes = Counter(d["source"] for d in docs)
    overlap = Counter()
    for v in multi:
        ss = sorted({docs[i]["source"] for i in v})
        for s, t in combinations(ss, 2):
            overlap[(s, t)] += 1
    n_src_in_comp = Counter(len({docs[i]["source"] for i in v}) for v in comp.values())

    # split strategies: same split proportions as ours
    props = Counter(s for s in sp if s)
    labels = list(props); weights = [props[k] for k in labels]
    rng = random.Random(42)
    rand_doc = [rng.choices(labels, weights)[0] for _ in docs]
    pmc_lab = {p: rng.choices(labels, weights)[0] for p in by_pmc}
    rand_pmc = [pmc_lab[d["pmcid"]] if d["pmcid"] else rand_doc[i] for i, d in enumerate(docs)]
    def native(d):
        n = (d.get("native_split") or "").lower()
        return "test" if n.startswith("test") else "val" if n.startswith(("val", "dev", "valid")) else "train"
    strategies = {"ours_article_cluster": sp, "native_source_splits": [native(d) for d in docs], "random_document": rand_doc, "random_pmcid": rand_pmc}
    cuts = {}
    for name, S in strategies.items():
        cuts[name] = {}
        for ln, E in layers.items():
            c = [e for e in E if S[e[0]] and S[e[1]] and S[e[0]] != S[e[1]]]
            test_docs = {i for e in c for i in e if S[i] == "test"}
            cuts[name][ln] = {"edges": len(E), "cross_split": len(c), "pct": round(100 * len(c) / max(1, len(E)), 2),
                              "test_docs_with_leak": len(test_docs)}

    res = {"nodes": len(docs), "nodes_by_source": dict(src_nodes),
           "edges": {k: len(v) for k, v in layers.items()},
           "components": len(comp), "multi_doc_components": len(multi),
           "largest_component": max(len(v) for v in comp.values()),
           "component_size_hist": {str(k): v for k, v in sorted(sizes.items())},
           "sources_per_component": {str(k): v for k, v in sorted(n_src_in_comp.items())},
           "source_overlap": {f"{s}|{t}": n for (s, t), n in overlap.most_common()},
           "split_cuts": cuts}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "component_size_hist"}, indent=1))


if __name__ == "__main__":
    main()

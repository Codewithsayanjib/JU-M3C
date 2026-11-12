"""Multi-layer network analysis (Kivela et al. 2014 style): one node set (case documents), four
edge layers, plus case-image and case-concept interconnections.

Layers (intra-layer edges between documents of the case pool):
  article  - same article in two sources (shared PMCID)
  text     - text near-duplicates
  image    - confirmed cross-article duplicate images
  concept  - documents of different articles sharing >= --min-shared major MeSH topics
Reports per-layer statistics, pairwise edge overlap (Jaccard), union components that only form
when layers are combined, the multiplex participation coefficient, leakage per layer under our
split, and the size of the case-image and case-MeSH interconnections.

Usage:
  python scripts/mm/multilayer.py --data $JUMC_DATA --mm $JUMC_MV/full \
      --pairs $JUMC_MV/reports/mm_leakage_full.pairs.jsonl --out reports/multilayer.json
"""
import argparse, gzip, json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import sparse


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def components(n, E):
    d = DSU(n)
    for x, y in E:
        d.union(x, y)
    return d


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "mm", "pairs", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--min-shared", type=int, default=2)
    a = ap.parse_args()
    data, mm = Path(a.data).expanduser(), Path(a.mm)
    docs = [json.loads(l) for l in gzip.open(data / "links/doc_index.jsonl.gz", "rt")]
    n = len(docs); idx = {d["doc_id"]: i for i, d in enumerate(docs)}
    split = {}
    for l in gzip.open(data / "links/splits.jsonl.gz", "rt"):
        r = json.loads(l); split[r["doc_id"]] = r["split"]
    sp = [split.get(d["doc_id"]) for d in docs]
    by_pmc = defaultdict(list)
    for i, d in enumerate(docs):
        if d["pmcid"]:
            by_pmc[d["pmcid"]].append(i)

    L = {k: set() for k in ("article", "text", "image", "concept")}
    for ids in by_pmc.values():
        for x, y in combinations(ids, 2):
            if docs[x]["source"] != docs[y]["source"]:
                L["article"].add((min(x, y), max(x, y)))
    for l in gzip.open(data / "links/near_dup_pairs.jsonl.gz", "rt"):
        p = json.loads(l)
        if p["a"] in idx and p["b"] in idx:
            x, y = idx[p["a"]], idx[p["b"]]; L["text"].add((min(x, y), max(x, y)))
    for l in open(a.pairs):
        p = json.loads(l)
        for x in by_pmc.get(p["image_a"].split("_")[0], []):
            for y in by_pmc.get(p["image_b"].split("_")[0], []):
                L["image"].add((min(x, y), max(x, y)))

    mesh = {r["pmcid"]: r for r in map(json.loads, open(mm / "mesh.jsonl"))}
    arts = sorted(p for p in by_pmc if p in mesh and any(m["major"] for m in mesh[p]["mesh"]))
    voc = {}; rows, cols = [], []
    for i, p in enumerate(arts):
        for u in {m["ui"] for m in mesh[p]["mesh"] if m["major"]}:
            rows.append(i); cols.append(voc.setdefault(u, len(voc)))
    M = sparse.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(arts), len(voc)))
    C = sparse.triu(M @ M.T, k=1).tocoo()
    keep = C.data >= a.min_shared
    for i, j in zip(C.row[keep], C.col[keep]):
        for x in by_pmc[arts[i]]:
            for y in by_pmc[arts[j]]:
                L["concept"].add((min(x, y), max(x, y)))
    print({k: len(v) for k, v in L.items()}, flush=True)

    per = {}
    deg = {k: Counter() for k in L}
    for k, E in L.items():
        for x, y in E:
            deg[k][x] += 1; deg[k][y] += 1
        d = components(n, E)
        cs = Counter(d.find(i) for i in deg[k])
        cross = sum(1 for x, y in E if sp[x] and sp[y] and sp[x] != sp[y])
        per[k] = {"edges": len(E), "active_nodes": len(deg[k]),
                  "mean_degree_active": round(2 * len(E) / max(1, len(deg[k])), 2),
                  "components_active": len(cs), "largest_component": max(cs.values()) if cs else 0,
                  "cross_split_edges_ours": cross,
                  "test_docs_exposed_ours": len({i for x, y in E if sp[x] != sp[y] for i in (x, y) if sp[i] == "test"})}

    overlap = {}
    for p, q in combinations(L, 2):
        inter = len(L[p] & L[q]); uni = len(L[p] | L[q])
        overlap[f"{p}|{q}"] = {"shared_edges": inter, "jaccard": round(inter / max(1, uni), 4)}
    multi_layer_edges = Counter()
    allE = Counter()
    for E in L.values():
        allE.update(E)
    multi_layer_edges = Counter(allE.values())

    # components of the leakage-relevant union (article+text+image) that no single layer connects
    leak_layers = ["article", "text", "image"]
    U = components(n, set().union(*(L[k] for k in leak_layers)))
    single = {k: components(n, L[k]) for k in leak_layers}
    groups = defaultdict(list)
    for i in range(n):
        groups[U.find(i)].append(i)
    multi = [g for g in groups.values() if len(g) > 1]
    only_combined = sum(1 for g in multi if all(len({single[k].find(i) for i in g}) > 1 for k in leak_layers))
    docs_only_combined = sum(len(g) for g in multi if all(len({single[k].find(i) for i in g}) > 1 for k in leak_layers))

    # multiplex participation coefficient over the 4 layers
    Lc = len(L); P = []
    for i in set().union(*(deg[k] for k in L)):
        o = sum(deg[k][i] for k in L)
        P.append(Lc / (Lc - 1) * (1 - sum((deg[k][i] / o) ** 2 for k in L)))
    P = np.array(P)
    nl = Counter(sum(1 for k in L if deg[k][i]) for i in set().union(*(deg[k] for k in L)))

    imgs_per_pmc = Counter(json.loads(l)["pmcid"] for l in open(mm / "figures.jsonl"))
    case_img = sum(imgs_per_pmc.get(d["pmcid"], 0) for d in docs if d["pmcid"])
    case_mesh = sum(len(mesh[d["pmcid"]]["mesh"]) for d in docs if d["pmcid"] in mesh)
    res = {"nodes": n, "layers": per, "edge_overlap": overlap,
           "edges_by_number_of_layers": {str(k): v for k, v in sorted(multi_layer_edges.items())},
           "leakage_union": {"multi_doc_components": len(multi), "components_only_when_layers_combined": only_combined,
                             "docs_in_those_components": docs_only_combined},
           "participation": {"nodes_with_any_edge": int(len(P)), "mean": round(float(P.mean()), 4),
                             "share_P0_single_layer": round(float((P == 0).mean()), 4),
                             "nodes_by_number_of_layers": {str(k): v for k, v in sorted(nl.items())}},
           "interlayer": {"case_image_links": case_img, "image_nodes": sum(imgs_per_pmc.values()),
                          "case_concept_links": case_mesh, "concept_nodes": len({m["ui"] for r in mesh.values() for m in r["mesh"]})},
           "concept_min_shared_major": a.min_shared}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

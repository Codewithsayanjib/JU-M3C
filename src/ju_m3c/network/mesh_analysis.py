"""Experiment B: the MeSH knowledge layer.

MeSH headings are assigned by NLM indexers, so they are an existing expert annotation.
Reports coverage (MeSH, MeSH or author keywords) overall, by source and by year; the most common
descriptors; publication types; a co-occurrence network of major-topic descriptors with Louvain
communities; and topic balance across our splits (Jensen-Shannon divergence of descriptor
distributions, train vs val vs test, against random splits of the same sizes).

Usage:
  python scripts/mm/mesh_analysis.py --data $JUMC_DATA \
      --mesh $JUMC_MV/full/mesh.jsonl \
      --pubyear $JUMC_MV/full/pubyear.jsonl --out reports/mesh.json
"""
import argparse, gzip, json, math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import networkx as nx


def jsd(p, q):
    keys = set(p) | set(q); sp, sq = sum(p.values()), sum(q.values()); d = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq; m = (a + b) / 2
        d += (a * math.log2(a / m) if a else 0) + (b * math.log2(b / m) if b else 0)
    return d / 2


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "mesh", "pubyear", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--min-cooc", type=int, default=20)
    a = ap.parse_args()
    data = Path(a.data).expanduser()
    mesh = {r["pmcid"]: r for r in map(json.loads, open(a.mesh))}
    year = {r["pmcid"]: r["year"] for r in map(json.loads, open(a.pubyear))}
    srcs, split = defaultdict(set), {}
    for l in gzip.open(data / "links/doc_index.jsonl.gz", "rt"):
        d = json.loads(l)
        if d["pmcid"]:
            srcs[d["pmcid"]].add(d["source"])
    for l in gzip.open(data / "links/splits.jsonl.gz", "rt"):
        d = json.loads(l)
        split.setdefault(d["doc_id"], d["split"])
    art_split = {}
    for l in gzip.open(data / "links/doc_index.jsonl.gz", "rt"):
        d = json.loads(l)
        if d["pmcid"] and d["doc_id"] in split:
            art_split[d["pmcid"]] = split[d["doc_id"]]

    has = lambda r: bool(r["mesh"]); has_any = lambda r: bool(r["mesh"] or r["keywords"])
    n = len(mesh)
    cov = {"articles": n, "with_pmid": sum(bool(r["pmid"]) for r in mesh.values()),
           "mesh": sum(map(has, mesh.values())), "mesh_or_keywords": sum(map(has_any, mesh.values()))}
    by_src = {}
    for s in sorted({s for v in srcs.values() for s in v}):
        rs = [mesh[p] for p, v in srcs.items() if s in v and p in mesh]
        by_src[s] = {"articles": len(rs), "mesh_pct": round(100 * sum(map(has, rs)) / max(1, len(rs)), 1),
                     "mesh_or_kw_pct": round(100 * sum(map(has_any, rs)) / max(1, len(rs)), 1)}
    by_year = defaultdict(lambda: [0, 0])
    for p, r in mesh.items():
        y = year.get(p)
        if y:
            by_year[y][0] += 1; by_year[y][1] += has(r)
    by_year = {y: round(100 * v[1] / v[0], 1) for y, v in sorted(by_year.items()) if v[0] >= 200}

    desc, major, ptypes = Counter(), Counter(), Counter()
    for r in mesh.values():
        desc.update({m["name"] for m in r["mesh"]})
        major.update({m["name"] for m in r["mesh"] if m["major"]})
        ptypes.update(set(r["pub_types"]))
    CHECK = {"Humans", "Male", "Female", "Adult", "Middle Aged", "Aged", "Child", "Adolescent", "Young Adult",
             "Infant", "Child, Preschool", "Aged, 80 and over", "Infant, Newborn"}

    cooc = Counter()
    for r in mesh.values():
        ms = sorted({m["name"] for m in r["mesh"] if m["major"]})
        cooc.update(combinations(ms, 2))
    G = nx.Graph()
    for (x, y), w in cooc.items():
        if w >= a.min_cooc:
            G.add_edge(x, y, weight=w)
    comms = nx.community.louvain_communities(G, weight="weight", seed=42) if G.number_of_nodes() else []
    comms = sorted(comms, key=len, reverse=True)
    deg = sorted(G.degree(weight="weight"), key=lambda x: -x[1])[:15]
    top_comms = [{"size": len(c), "top": [m for m, _ in sorted(((m, major[m]) for m in c), key=lambda x: -x[1])[:8]]}
                 for c in comms[:10]]

    per_split = defaultdict(Counter)
    for p, r in mesh.items():
        s = art_split.get(p)
        if s and r["mesh"]:
            per_split[s].update({m["name"] for m in r["mesh"] if m["name"] not in CHECK})
    balance = {f"{x}|{y}": round(jsd(per_split[x], per_split[y]), 4)
               for x, y in combinations(sorted(per_split), 2)}
    # null: same split sizes, labels shuffled across articles (5 seeds)
    import random
    arts = [(p, art_split[p]) for p, r in mesh.items() if p in art_split and r["mesh"]]
    null = defaultdict(list)
    for seed in range(5):
        lab = [s for _, s in arts]; random.Random(seed).shuffle(lab)
        ps = defaultdict(Counter)
        for (p, _), s in zip(arts, lab):
            ps[s].update({m["name"] for m in mesh[p]["mesh"] if m["name"] not in CHECK})
        for x, y in combinations(sorted(ps), 2):
            null[f"{x}|{y}"].append(jsd(ps[x], ps[y]))
    balance = {k: {"ours": v, "random_mean": round(sum(null[k]) / len(null[k]), 4)} for k, v in balance.items()}

    res = {"coverage": cov, "coverage_by_source": by_src, "mesh_pct_by_year": by_year,
           "distinct_descriptors": len(desc), "distinct_major": len(major),
           "top_descriptors": [d for d in desc.most_common(40) if d[0] not in CHECK][:25],
           "top_major": major.most_common(25), "top_pub_types": ptypes.most_common(10),
           "cooc_network": {"min_cooc": a.min_cooc, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                            "density": round(nx.density(G), 4) if G.number_of_nodes() else 0,
                            "communities": len(comms),
                            "modularity": round(nx.community.modularity(G, comms, weight="weight"), 3) if comms else None,
                            "top_weighted_degree": deg, "top_communities": top_comms},
           "split_jsd": balance, "split_articles_with_mesh": {k: sum(1 for p, s in art_split.items() if s == k and p in mesh and mesh[p]["mesh"]) for k in per_split}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    nx.write_graphml(G, str(Path(a.out).with_suffix(".graphml")))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()

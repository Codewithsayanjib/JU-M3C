"""Phase 5: profile label coverage and distributions (numbers only, no task decisions).

  python scripts/corpus/profile.py

Reads  $CORPUS_ROOT/interim/*.jsonl and processed/doc_index.jsonl
Writes $CORPUS_ROOT/reports/profile.json and prints a summary.

Coverage is reported per source (documents) and per article cluster (a label counts for the
cluster if any document in it carries it, i.e. after pooling labels across sources).
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
FIELDS = ["title", "journal", "year", "keywords", "mesh", "major_mesh", "diagnosis", "summary",
          "specialty", "doc_type", "section_header", "age", "gender", "extraction"]
AUX = ["abstract", "dialogue", "article_text", "diagnostic_reasoning"]
# MeSH check tags / publication types that say nothing about the case itself.
GENERIC_MESH = {"humans", "male", "female", "adult", "middle aged", "aged", "aged, 80 and over",
                "young adult", "adolescent", "child", "child, preschool", "infant", "infant, newborn",
                "case reports", "animals", "treatment outcome", "follow-up studies", "pregnancy"}


def main():
    index = {}
    for line in open(CORPUS_ROOT / "processed" / "doc_index.jsonl"):
        d = json.loads(line)
        index[d["doc_id"]] = d
    per_source = defaultdict(lambda: {"docs": 0, "cov": Counter(), "words": [], "licence": Counter(),
                                      "years": Counter(), "splits": Counter()})
    cluster_cov = defaultdict(set)
    cluster_sources = defaultdict(set)
    mesh, major, keywords, journals, diag, spec, dtype, sect = (Counter() for _ in range(8))
    n_specific_mesh = []
    for f in sorted((CORPUS_ROOT / "interim").glob("*.jsonl")):
        if f.name.startswith("._"):
            continue
        for line in open(f):
            r = json.loads(line)
            s = per_source[r["source"]]
            s["docs"] += 1
            s["words"].append(r["n_words"])
            s["licence"][r["licence"]] += 1
            s["splits"][r["native_split"] or "none"] += 1
            L, c = r["labels"], index[r["doc_id"]]["article_cluster"]
            cluster_sources[c].add(r["source"])
            if L.get("year"):
                s["years"][str(L["year"])[:4]] += 1
            specific = [m for m in L.get("mesh", []) if m.lower() not in GENERIC_MESH]
            have = [k for k in FIELDS if L.get(k)] + [k for k in AUX if r["aux"].get(k)]
            if specific:
                have.append("mesh_specific")
                n_specific_mesh.append(len(specific))
            for k in have:
                s["cov"][k] += 1
                cluster_cov[k].add(c)
            mesh.update(specific)
            major.update(L.get("major_mesh", []))
            keywords.update(k.lower() for k in L.get("keywords", []))
            if L.get("journal"):
                journals[L["journal"]] += 1
            if L.get("diagnosis"):
                diag[L["diagnosis"].strip().lower()] += 1
            if L.get("specialty"):
                spec[L["specialty"]] += 1
            if L.get("doc_type"):
                dtype[L["doc_type"]] += 1
            if L.get("section_header"):
                sect[L["section_header"]] += 1

    n_clusters = len(cluster_sources)
    out = {"n_docs": len(index), "n_article_clusters": n_clusters, "per_source": {}, "cluster_coverage": {
        k: {"clusters": len(v), "pct": round(100 * len(v) / n_clusters, 1)} for k, v in sorted(cluster_cov.items())}}
    for name, s in per_source.items():
        w = np.array(s["words"])
        out["per_source"][name] = {
            "docs": s["docs"],
            "words": {"p5": int(np.percentile(w, 5)), "median": int(np.median(w)),
                      "p95": int(np.percentile(w, 95)), "max": int(w.max())},
            "coverage_pct": {k: round(100 * v / s["docs"], 1) for k, v in sorted(s["cov"].items())},
            "licence": dict(s["licence"].most_common(8)), "native_splits": dict(s["splits"]),
            "years": {"min": min(s["years"]) if s["years"] else None,
                      "max": max(s["years"]) if s["years"] else None}}
    out["distributions"] = {
        "mesh_specific": {"distinct": len(mesh), "per_doc_median": int(np.median(n_specific_mesh)) if n_specific_mesh else 0,
                          "terms_ge_500_docs": sum(v >= 500 for v in mesh.values()),
                          "terms_ge_100_docs": sum(v >= 100 for v in mesh.values()), "top": mesh.most_common(30)},
        "major_mesh": {"distinct": len(major), "top": major.most_common(20)},
        "keywords": {"distinct": len(keywords), "top": keywords.most_common(20)},
        "journals": {"distinct": len(journals), "top": journals.most_common(20)},
        "diagnosis": {"distinct": len(diag), "docs": sum(diag.values()),
                      "repeated_ge_5": sum(v >= 5 for v in diag.values()), "top": diag.most_common(15)},
        "specialty": dict(spec.most_common()), "doc_type": dict(dtype.most_common()),
        "section_header": dict(sect.most_common())}
    (CORPUS_ROOT / "reports" / "profile.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"docs": out["n_docs"], "article_clusters": n_clusters}))


if __name__ == "__main__":
    main()

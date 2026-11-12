"""Phase 4: de-duplication and cross-source overlap.

  python scripts/corpus/dedup.py

Reads  $CORPUS_ROOT/interim/*.jsonl
Writes $CORPUS_ROOT/processed/doc_index.jsonl   one line per document (no text):
          doc_id, source, n_words, pmcid, exact_cluster, dup_cluster, article_cluster, flags
       $CORPUS_ROOT/reports/dedup_report.json   counts, overlap matrices, thresholds
       $CORPUS_ROOT/reports/near_dup_pairs.jsonl verified near-duplicate / containment pairs

Three kinds of link (documents under MIN_WORDS words are only matched exactly):
  exact       identical text after normalisation (lowercase, alphanumerics only)
  near-dup    Jaccard >= NEAR_J on 5-word shingles
  contained   >= CONTAIN of the smaller document's shingles appear in the larger one
Clusters (union-find):
  dup_cluster      exact + near-dup                      -> "the same document"
  article_cluster  dup_cluster + contained + same PMCID  -> "the same published article";
                   leak-free splits must keep an article_cluster on one side.
Candidates come from bottom-K sketches of the shingle hashes; every candidate is then verified
exactly on the full shingle sets. All hashing is deterministic (crc32-based).
"""
import json
import os
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
SHINGLE, K, MIN_SHARED, MAX_POSTING = 5, 32, 6, 200
NEAR_J, CONTAIN, MIN_WORDS = 0.9, 0.9, 30
_TOK = re.compile(r"[a-z0-9]+")
_P = np.array([pow(1000003, i, 1 << 64) for i in range(SHINGLE)], dtype=np.uint64)
_cache: dict = {}


def tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def exact_key(toks: list[str]) -> int:
    return zlib.crc32(" ".join(toks).encode()) | (len(toks) << 32)


def shingles(toks: list[str]) -> np.ndarray:
    """Sorted unique 64-bit hashes of all SHINGLE-word windows."""
    if len(toks) < SHINGLE:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((_cache.get(t) or _cache.setdefault(t, zlib.crc32(t.encode()) + 1) for t in toks),
                      dtype=np.uint64, count=len(toks))
    win = np.lib.stride_tricks.sliding_window_view(ids, SHINGLE)
    with np.errstate(over="ignore"):
        return np.unique((win * _P).sum(axis=1, dtype=np.uint64))


def jaccard_containment(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter), inter / min(a.size, b.size)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def find_links(texts: list[str]) -> tuple[list[int], list[tuple]]:
    """Return (exact keys, verified pairs [(i, j, jaccard, containment)]).
    Only a bottom-K sketch is kept per document; candidates are re-tokenised for verification."""
    keys, first, postings = [], {}, defaultdict(list)
    for i, t in enumerate(texts):
        toks = tokens(t)
        k = exact_key(toks)
        keys.append(k)
        if k in first:
            continue  # exact duplicates are represented by their first occurrence
        first[k] = i
        if len(toks) >= MIN_WORDS:
            for v in shingles(toks)[:K].tolist():  # sorted -> K smallest hashes = bottom-K sketch
                postings[v].append(i)
    shared = Counter()
    for docs in postings.values():
        if 1 < len(docs) <= MAX_POSTING:
            for x in range(len(docs)):
                for y in range(x + 1, len(docs)):
                    shared[(docs[x], docs[y])] += 1
    del postings
    pairs = []
    for (i, j), c in shared.items():
        if c >= MIN_SHARED:
            jac, con = jaccard_containment(shingles(tokens(texts[i])), shingles(tokens(texts[j])))
            if jac >= NEAR_J or con >= CONTAIN:
                pairs.append((i, j, round(jac, 4), round(con, 4)))
    return keys, pairs


def main():
    docs, texts = [], []
    for f in sorted((CORPUS_ROOT / "interim").glob("*.jsonl")):
        if f.name.startswith("._"):
            continue
        for line in open(f):
            r = json.loads(line)
            docs.append({"doc_id": r["doc_id"], "source": r["source"], "n_words": r["n_words"],
                         "pmcid": r["pmcid"], "native_split": r["native_split"]})
            texts.append(r["text"])
    n = len(docs)
    print(f"{n:,} documents loaded", flush=True)
    keys, pairs = find_links(texts)
    del texts
    print(f"{len(pairs):,} verified near-duplicate / containment pairs", flush=True)

    exact, dup, art = UF(n), UF(n), UF(n)
    first = {}
    for i, k in enumerate(keys):
        if k in first:
            for uf in (exact, dup, art):
                uf.union(first[k], i)
        else:
            first[k] = i
    for i, j, jac, con in pairs:
        if jac >= NEAR_J:
            dup.union(i, j)
        art.union(i, j)
    by_pmc = {}
    for i, d in enumerate(docs):
        if d["pmcid"]:
            art.union(by_pmc.setdefault(d["pmcid"], i), i)

    sources = sorted({d["source"] for d in docs})
    out_dir = CORPUS_ROOT / "processed"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "doc_index.jsonl", "w") as f:
        for i, d in enumerate(docs):
            d.update(exact_cluster=exact.find(i), dup_cluster=dup.find(i), article_cluster=art.find(i),
                     short=d["n_words"] < MIN_WORDS)
            f.write(json.dumps(d) + "\n")
    with open(CORPUS_ROOT / "reports" / "near_dup_pairs.jsonl", "w") as f:
        for i, j, jac, con in pairs:
            f.write(json.dumps({"a": docs[i]["doc_id"], "b": docs[j]["doc_id"], "jaccard": jac,
                                "containment": con}) + "\n")

    def overlap(field):
        members = defaultdict(set)
        for d in docs:
            members[d[field]].add(d["source"])
        return {a: {b: sum(1 for d in docs if d["source"] == a and b in members[d[field]])
                    for b in sources} for a in sources}

    per_source = {}
    for s in sources:
        ds = [d for d in docs if d["source"] == s]
        per_source[s] = {"docs": len(ds), "unique_exact": len({d["exact_cluster"] for d in ds}),
                         "unique_dup": len({d["dup_cluster"] for d in ds}),
                         "articles": len({d["article_cluster"] for d in ds}),
                         "short_docs": sum(d["short"] for d in ds)}
    report = {"params": {"shingle_words": SHINGLE, "sketch_k": K, "min_shared": MIN_SHARED,
                         "near_jaccard": NEAR_J, "containment": CONTAIN, "min_words": MIN_WORDS},
              "total": {"docs": n, "unique_exact": len({d["exact_cluster"] for d in docs}),
                        "unique_dup": len({d["dup_cluster"] for d in docs}),
                        "article_clusters": len({d["article_cluster"] for d in docs})},
              "per_source": per_source,
              "overlap_docs_sharing_dup_cluster": overlap("dup_cluster"),
              "overlap_docs_sharing_article_cluster": overlap("article_cluster")}
    (CORPUS_ROOT / "reports" / "dedup_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["total"]), flush=True)


if __name__ == "__main__":
    main()

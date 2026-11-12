"""Text-based near-duplicate / containment detector for clinical case documents.

Self-contained: standard library only. Two documents are judged the same case when, after
normalisation, their five-word shingle sets have Jaccard similarity >= ``jaccard`` OR the smaller
set is >= ``containment`` contained in the larger. Candidate pairs are found with a lightweight
shared-shingle blocking index so the detector scales beyond all-pairs comparison.

This is the reusable core of the JU-M3C leakage audit; it makes no assumption about the nine
sources used in the paper and runs on any two sets of documents.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

_KEEP = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> list[str]:
    """Lowercase, drop non-alphanumeric characters, split on whitespace."""
    text = _KEEP.sub(" ", text.lower())
    return _WS.sub(" ", text).strip().split()


def shingles(text: str, k: int = 5) -> frozenset[str]:
    """Set of k-word shingles (k=5 in the paper). Short texts yield their single joined token."""
    words = normalise(text)
    if len(words) < k:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: frozenset, b: frozenset) -> float:
    """Fraction of the smaller set contained in the larger."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return len(small & large) / len(small)


@dataclass
class DetectorConfig:
    k: int = 5
    jaccard: float = 0.9
    containment: float = 0.9
    min_words: int = 30          # documents shorter than this are matched by exact shingle only
    max_doc_freq: int = 200      # shingles in more than this many documents are ignored as boilerplate
    sketch_size: int = 32        # smallest shingle hashes kept per document for blocking
    min_shared_sketch: int = 6   # candidate pairs must share at least this many sketch hashes


def _sketch(sh: frozenset[str], size: int) -> list[int]:
    return sorted(hash(s) for s in sh)[:size]


def candidate_pairs(docs: dict[str, frozenset], cfg: DetectorConfig):
    """Yield candidate (id_a, id_b) pairs that share enough blocking hashes to be worth verifying."""
    sketches = {i: _sketch(sh, cfg.sketch_size) for i, sh in docs.items()}
    by_hash: dict[int, list[str]] = defaultdict(list)
    for i, sk in sketches.items():
        for h in sk:
            by_hash[h].append(i)
    shared: dict[frozenset, int] = defaultdict(int)
    for ids in by_hash.values():
        if len(ids) > cfg.max_doc_freq:
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                shared[frozenset((ids[a], ids[b]))] += 1
    for pair, n in shared.items():
        if n >= cfg.min_shared_sketch:
            yield tuple(pair)


def verify(a_text: str, b_text: str, a_sh: frozenset, b_sh: frozenset, cfg: DetectorConfig) -> bool:
    """Confirm a candidate pair on full shingle sets."""
    short = len(normalise(a_text)) < cfg.min_words or len(normalise(b_text)) < cfg.min_words
    if short:
        return a_sh == b_sh and bool(a_sh)
    return jaccard(a_sh, b_sh) >= cfg.jaccard or containment(a_sh, b_sh) >= cfg.containment


def text_links(docs: dict[str, str], cfg: DetectorConfig | None = None) -> set[tuple[str, str]]:
    """All verified near-duplicate / containment pairs within one set of ``{id: text}`` documents."""
    cfg = cfg or DetectorConfig()
    sh = {i: shingles(t, cfg.k) for i, t in docs.items()}
    out: set[tuple[str, str]] = set()
    for a, b in candidate_pairs(sh, cfg):
        if verify(docs[a], docs[b], sh[a], sh[b], cfg):
            out.add((a, b) if a < b else (b, a))
    return out


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return dict(groups)

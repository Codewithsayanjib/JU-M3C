"""Cross-dataset leakage report.

Given a *training* pool and a *test* set of documents, count how many test documents are linked to
a training document through the article layer (shared identifier) or the text layer (near-duplicate
or containment), and break the count down by the source that holds the copy. This reproduces the
headline audit of the JU-M3C paper and works on any two public datasets.

A document is a dict with at least ``id`` and ``text``; optional ``pmcid`` enables the article layer
and optional ``source`` labels where the copy sits.
"""
from __future__ import annotations

from collections import defaultdict

from .detector import DetectorConfig, containment, jaccard, normalise, shingles, candidate_pairs, verify

Doc = dict


def _by_pmcid(docs: list[Doc]) -> dict[str, list[Doc]]:
    out: dict[str, list[Doc]] = defaultdict(list)
    for d in docs:
        pid = d.get("pmcid")
        if pid:
            out[str(pid)].append(d)
    return out


def article_leaks(train: list[Doc], test: list[Doc]) -> set[str]:
    """Test ids whose article identifier also appears in the training pool."""
    train_pmcids = {str(d["pmcid"]) for d in train if d.get("pmcid")}
    return {d["id"] for d in test if d.get("pmcid") and str(d["pmcid"]) in train_pmcids}


def text_leaks(train: list[Doc], test: list[Doc], cfg: DetectorConfig) -> dict[str, list[str]]:
    """Map each leaked test id to the training ids it matches by text. Cross-set only."""
    docs = {f"TRN::{d['id']}": d["text"] for d in train}
    docs.update({f"TST::{d['id']}": d["text"] for d in test})
    sh = {i: shingles(t, cfg.k) for i, t in docs.items()}
    leaks: dict[str, list[str]] = defaultdict(list)
    for a, b in candidate_pairs(sh, cfg):
        if a[:5] == b[:5]:            # both train or both test: not cross-set leakage
            continue
        if verify(docs[a], docs[b], sh[a], sh[b], cfg):
            tst, trn = (a, b) if a.startswith("TST::") else (b, a)
            leaks[tst[5:]].append(trn[5:])
    return dict(leaks)


def audit(train: list[Doc], test: list[Doc], cfg: DetectorConfig | None = None) -> dict:
    """Full contamination report for one (train, test) pair."""
    cfg = cfg or DetectorConfig()
    art = article_leaks(train, test)
    txt = text_leaks(train, test, cfg)
    txt_ids = set(txt)
    leaked = art | txt_ids
    n = len(test)
    source_of = {d["id"]: d.get("source", "unknown") for d in train}
    by_source: dict[str, int] = defaultdict(int)
    for tst, trns in txt.items():
        for trn in trns:
            by_source[source_of.get(trn, "unknown")] += 1
    return {
        "n_test": n,
        "leaked": len(leaked),
        "leaked_pct": round(100 * len(leaked) / n, 2) if n else 0.0,
        "article_layer_only": len(art - txt_ids),
        "text_layer_only": len(txt_ids - art),
        "both_layers": len(art & txt_ids),
        "copies_by_training_source": dict(by_source),
        "leaked_ids": sorted(leaked),
    }

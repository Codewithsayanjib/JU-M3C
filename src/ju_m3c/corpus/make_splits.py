"""Phase 6: leak-free train/val/test splits at article-cluster level (seed 42).

  python scripts/corpus/make_splits.py

Reads  $CORPUS_ROOT/processed/doc_index.jsonl
Writes $CORPUS_ROOT/processed/splits.jsonl     doc_id, source, article_cluster, split, native_split, reason
       $CORPUS_ROOT/reports/split_report.json

Rules, in priority order (a cluster always goes to ONE split):
  1. MTSamples keeps its existing split unchanged (the sealed zero-shot test runs depend on it).
     A cluster containing MTSamples notes takes the strictest of their splits (test > val > train).
  2. A cluster containing any document from a source's native TEST set goes to test, so every
     published test set stays inside our test split and nothing from it can leak into train.
  3. A cluster containing a native VAL document (and no test document) goes to val.
  4. All other clusters are assigned by a deterministic hash of the cluster id:
     5% test, 5% val, 90% train.
Cross-source generalisation uses the same file: hold out source X by training only on clusters
that contain no document from X (helper: held_out_source()).
"""
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
SEED, P_TEST, P_VAL = 42, 0.05, 0.05
RANK = {"train": 0, "val": 1, "test": 2}


def native_bucket(native_split: str | None) -> str | None:
    if not native_split:
        return None
    if native_split.startswith("test"):
        return "test"
    if native_split.startswith("val"):
        return "val"
    return "train"


def hash_split(cluster_key: str) -> str:
    u = int(hashlib.sha256(f"{SEED}:{cluster_key}".encode()).hexdigest()[:12], 16) / 16 ** 12
    return "test" if u < P_TEST else "val" if u < P_TEST + P_VAL else "train"


def assign(docs: list[dict]) -> tuple[dict, dict]:
    """docs need doc_id, source, article_cluster, native_split. Returns (cluster -> split, cluster -> reason)."""
    members = defaultdict(list)
    for d in docs:
        members[d["article_cluster"]].append(d)
    split, reason = {}, {}
    for c, ds in members.items():
        mts = [native_bucket(d["native_split"]) for d in ds if d["source"] == "mtsamples"]
        native = [native_bucket(d["native_split"]) for d in ds if d["source"] != "mtsamples"]
        if mts:
            strictest = max(mts + [b for b in native if b in ("test", "val")], key=RANK.get)
            split[c], reason[c] = strictest, "mtsamples_split" if strictest == max(mts, key=RANK.get) else "native_test_or_val"
        elif "test" in native:
            split[c], reason[c] = "test", "native_test"
        elif "val" in native:
            split[c], reason[c] = "val", "native_val"
        else:
            key = min(d["doc_id"] for d in ds)  # stable cluster key, independent of row order
            split[c], reason[c] = hash_split(key), "hash"
    return split, reason


def held_out_source(rows: list[dict], source: str) -> tuple[list[str], list[str]]:
    """Cross-source config: (train doc_ids from clusters without `source`, test doc_ids of `source`)."""
    tainted = {r["article_cluster"] for r in rows if r["source"] == source}
    train = [r["doc_id"] for r in rows if r["split"] == "train" and r["article_cluster"] not in tainted]
    test = [r["doc_id"] for r in rows if r["source"] == source and r["split"] == "test"]
    return train, test


def main():
    docs = [json.loads(l) for l in open(CORPUS_ROOT / "processed" / "doc_index.jsonl")]
    split, reason = assign(docs)
    rows = [{"doc_id": d["doc_id"], "source": d["source"], "article_cluster": d["article_cluster"],
             "split": split[d["article_cluster"]], "native_split": d["native_split"],
             "reason": reason[d["article_cluster"]]} for d in docs]
    with open(CORPUS_ROOT / "processed" / "splits.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- checks
    per_cluster = defaultdict(set)
    for r in rows:
        per_cluster[r["article_cluster"]].add(r["split"])
    assert all(len(v) == 1 for v in per_cluster.values()), "a cluster spans two splits"
    mts = [r for r in rows if r["source"] == "mtsamples"]
    mts_changed = [r["doc_id"] for r in mts if r["split"] != native_bucket(r["native_split"])]
    native_test_outside = sum(1 for r in rows if native_bucket(r["native_split"]) == "test" and r["split"] != "test")

    by_source = defaultdict(Counter)
    for r in rows:
        by_source[r["source"]][r["split"]] += 1
    report = {"seed": SEED, "p_test": P_TEST, "p_val": P_VAL,
              "docs": dict(Counter(r["split"] for r in rows)),
              "article_clusters": dict(Counter(split.values())),
              "reasons_clusters": dict(Counter(reason.values())),
              "by_source_docs": {s: dict(c) for s, c in sorted(by_source.items())},
              "checks": {"clusters_spanning_two_splits": 0,
                         "mtsamples_docs_moved_to_stricter_split": len(mts_changed),
                         "mtsamples_moved_ids": mts_changed,
                         "native_test_docs_outside_test": native_test_outside},
              "native_train_docs_moved_to_test_or_val": sum(
                  1 for r in rows if native_bucket(r["native_split"]) == "train" and r["split"] != "train")}
    (CORPUS_ROOT / "reports" / "split_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("docs", "article_clusters", "reasons_clusters", "checks")}, indent=1))


if __name__ == "__main__":
    main()

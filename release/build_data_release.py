"""Assemble the JU-M3C data release.

  python scripts/release/build_data_release.py --corpus-root corpus --results results --out release/data

Parts (each is skipped with a note when its inputs are missing, so the script can be run in stages):
  corpus/   records.jsonl.gz       every document with source, licence, identifiers, text, labels,
                                   duplicate cluster, article cluster and split   (needs interim/ + processed/)
  links/    doc_index.jsonl.gz     per-document cluster ids (no text)
            near_dup_pairs.jsonl.gz verified near-duplicate and containment pairs
            splits.jsonl.gz        split per document with the rule that assigned it
  tasks/    {task}/{train,val,test,test_eval,dev}.jsonl.gz
  outputs/  raw model outputs and scores behind every number in the paper
  MANIFEST.sha256                 checksum of every file

Nothing is modified in place; inputs are only read.
"""
import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

TASKS = ["summarization", "title_generation", "diagnosis_mcq", "diagnosis_open", "notes_specialty",
         "notes_doctype", "notes_summary", "notes_section", "dialogue2note"]
OUTPUT_DIRS = ["sealed_test_eval", "reliability", "prompt_variants", "fewshot", "dev"]
SKIP = shutil.ignore_patterns("._*", ".DS_Store")  # macOS metadata files on external drives


def gz_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)


def corpus_records(root: Path, out: Path, notes: list):
    interim, proc = root / "interim", root / "processed"
    if not (interim.is_dir() and (proc / "doc_index.jsonl").exists() and (proc / "splits.jsonl").exists()):
        notes.append("corpus/: skipped (interim/ or processed/doc_index.jsonl or splits.jsonl missing)")
        return {}
    index = {r["doc_id"]: r for r in map(json.loads, open(proc / "doc_index.jsonl"))}
    split = {r["doc_id"]: r["split"] for r in map(json.loads, open(proc / "splits.jsonl"))}
    stats = Counter()
    lic = Counter()
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "corpus" / "records.jsonl.gz", "wt", compresslevel=9) as fo:
        for f in sorted(f for f in interim.glob("*.jsonl") if not f.name.startswith("._")):
            for line in open(f):
                r = json.loads(line)
                ix = index.get(r["doc_id"], {})
                r["dup_cluster"] = ix.get("dup_cluster")
                r["article_cluster"] = ix.get("article_cluster")
                r["split"] = split.get(r["doc_id"])
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats[r["source"]] += 1
                lic[r.get("licence") or "unspecified"] += 1
    return {"documents": sum(stats.values()), "per_source": dict(stats), "per_licence": dict(lic.most_common())}


def links(root: Path, out: Path, notes: list):
    pairs = [(root / "processed" / "doc_index.jsonl", "doc_index.jsonl.gz"),
             (root / "reports" / "near_dup_pairs.jsonl", "near_dup_pairs.jsonl.gz"),
             (root / "processed" / "splits.jsonl", "splits.jsonl.gz")]
    for src, name in pairs:
        if src.exists():
            gz_copy(src, out / "links" / name)
        else:
            notes.append(f"links/{name}: skipped ({src.relative_to(root)} missing)")
    for rep in ("dedup_report.json", "split_report.json", "task_report.json", "normalize_stats.json"):
        if (root / "reports" / rep).exists():
            (out / "reports").mkdir(parents=True, exist_ok=True)
            shutil.copy(root / "reports" / rep, out / "reports" / rep)


def tasks(root: Path, out: Path, notes: list):
    counts = {}
    for t in TASKS:
        d = root / "processed" / "tasks" / t
        counts[t] = {}
        for sp in ("train", "val", "test", "test_eval", "dev"):
            f = d / f"{sp}.jsonl"
            if f.exists():
                gz_copy(f, out / "tasks" / t / f"{sp}.jsonl.gz")
                counts[t][sp] = sum(1 for _ in open(f))
        missing = {"train", "val", "test"} - set(counts[t])
        if missing:
            notes.append(f"tasks/{t}: {', '.join(sorted(missing))} missing (only the evaluation and development subsets were found)")
    return counts


def outputs(results: Path, out: Path, notes: list):
    bench = results / "bench"
    for d in OUTPUT_DIRS:
        if (bench / d).is_dir():
            shutil.copytree(bench / d, out / "outputs" / d, dirs_exist_ok=True, ignore=SKIP)
        else:
            notes.append(f"outputs/{d}: skipped (bench/{d} missing)")
    lk = results / "leakage"
    if lk.is_dir():  # outputs, training-set composition and summary; adapter weights are not released
        for sub in ("outputs", "data"):
            if (lk / sub).is_dir():
                shutil.copytree(lk / sub, out / "outputs" / "leakage" / sub, dirs_exist_ok=True, ignore=SKIP)
        if (lk / "summary.json").exists():
            shutil.copy(lk / "summary.json", out / "outputs" / "leakage" / "summary.json")


def manifest(out: Path):
    lines = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{h}  {f.relative_to(out)}")
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    notes = []
    stats = {"corpus": corpus_records(a.corpus_root, a.out, notes)}
    links(a.corpus_root, a.out, notes)
    stats["tasks"] = tasks(a.corpus_root, a.out, notes)
    outputs(a.results, a.out, notes)
    for f in ("README.md", "LICENSE.md", "SOURCES.md"):
        src = Path(__file__).parent / "card" / f
        if src.exists():
            shutil.copy(src, a.out / f)
    (a.out / "release_stats.json").write_text(json.dumps({**stats, "notes": notes}, indent=1))
    n = manifest(a.out)
    print(f"wrote {a.out} ({n} files)")
    for x in notes:
        print("  note:", x)


if __name__ == "__main__":
    main()

"""Build per-task example files from the unified corpus (native labels only).

  python scripts/corpus/build_tasks.py

Reads  $CORPUS_ROOT/interim/*.jsonl, processed/doc_index.jsonl, processed/splits.jsonl
Writes $CORPUS_ROOT/processed/tasks/{task}/{train,val,test}.jsonl
       $CORPUS_ROOT/processed/tasks/{task}/test_eval.jsonl   seeded evaluation subset (<= 1,000)
       $CORPUS_ROOT/reports/task_report.json

Tasks
  summarization     case text -> abstract / summary        (multicare abstract, multiclinsum summary)
  title_generation  case text -> article title              (pmc_patients, multicare, medcasereasoning)
  diagnosis_mcq     case text + 4 options -> letter         (cupcase; options shuffled, seed 42)
  diagnosis_open    case presentation -> final diagnosis    (medcasereasoning; secondary track)
  notes_specialty   note -> specialty                       (mtsamples)
  notes_doctype     note -> document type                   (mtsamples)
  notes_summary     note -> one-line description            (mtsamples)
  notes_section     note section -> section header          (mts_dialog)
  dialogue2note     dialogue -> note / note section         (aci_bench, mts_dialog)

Rules: one example per near-duplicate cluster per task (first source in SOURCE_ORDER wins);
inputs under MIN_WORDS words are dropped (except note sections); the split is the corpus split.
The evaluation subset takes one example per article cluster, stratified by source, seed 42.
"""
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
SEED, EVAL_N, MIN_WORDS, MIN_PER_SOURCE = 42, 1000, 30, 50
SOURCE_ORDER = ["mtsamples", "aci_bench", "mts_dialog", "casereportbench", "cupcase",
                "medcasereasoning", "multiclinsum", "multicare", "pmc_patients"]
TASKS = ["summarization", "title_generation", "diagnosis_mcq", "diagnosis_open", "notes_specialty",
         "notes_doctype", "notes_summary", "notes_section", "dialogue2note"]


def mcq(doc_id: str, correct: str, distractors: list[str]) -> tuple[list[str], str]:
    """Deterministic shuffle of the 4 options; returns (options, answer letter)."""
    opts = [correct] + list(distractors)
    rng = random.Random(int(hashlib.sha256(f"{SEED}:{doc_id}".encode()).hexdigest()[:8], 16))
    rng.shuffle(opts)
    return opts, "ABCD"[opts.index(correct)]


def examples(r: dict):
    """Yield (task, input, target, extra) for one unified record."""
    L, A, s, text = r["labels"], r["aux"], r["source"], r["text"]
    long_enough = r["n_words"] >= MIN_WORDS
    if s == "multicare" and long_enough:
        if A.get("abstract"):
            yield "summarization", text, A["abstract"], {"target_kind": "abstract"}
        if L.get("title"):
            yield "title_generation", text, L["title"], {}
    elif s == "multiclinsum" and long_enough and L.get("summary"):
        yield "summarization", text, L["summary"], {"target_kind": "summary", "subset": A.get("subset")}
    elif s == "pmc_patients" and long_enough and L.get("title"):
        yield "title_generation", text, L["title"], {}
    elif s == "medcasereasoning" and long_enough:
        if L.get("title"):
            yield "title_generation", text, L["title"], {}
        if L.get("diagnosis"):
            yield "diagnosis_open", text, L["diagnosis"], {}
    elif s == "cupcase" and long_enough and L.get("diagnosis") and len(L.get("distractors", [])) == 3:
        opts, ans = mcq(r["doc_id"], L["diagnosis"], L["distractors"])
        yield "diagnosis_mcq", text, ans, {"options": opts, "answer_text": L["diagnosis"]}
    elif s == "mtsamples" and long_enough:
        if L.get("specialty"):
            yield "notes_specialty", text, L["specialty"], {}
        if L.get("doc_type"):
            yield "notes_doctype", text, L["doc_type"], {}
        if L.get("summary"):
            yield "notes_summary", text, L["summary"], {}
    elif s == "mts_dialog":
        if L.get("section_header"):
            yield "notes_section", text, L["section_header"], {}
        if A.get("dialogue"):
            yield "dialogue2note", A["dialogue"], text, {"target_kind": "note_section",
                                                         "section_header": L.get("section_header")}
    elif s == "aci_bench" and A.get("dialogue"):
        yield "dialogue2note", A["dialogue"], text, {"target_kind": "full_note"}


def eval_subset(rows: list[dict], n: int = EVAL_N) -> list[dict]:
    """<= n test examples: one per article cluster, stratified by source (min MIN_PER_SOURCE each)."""
    rng = random.Random(SEED)
    by_cluster = {}
    for r in sorted(rows, key=lambda r: r["example_id"]):
        by_cluster.setdefault(r["article_cluster"], r)
    pool = defaultdict(list)
    for r in by_cluster.values():
        pool[r["source"]].append(r)
    total = sum(len(v) for v in pool.values())
    if total <= n:
        return sorted(by_cluster.values(), key=lambda r: r["example_id"])
    quota = {s: min(len(v), max(MIN_PER_SOURCE, round(n * len(v) / total))) for s, v in pool.items()}
    while sum(quota.values()) > n:  # trim the largest stratum
        s = max(quota, key=quota.get)
        quota[s] -= 1
    out = []
    for s in sorted(pool):
        out += rng.sample(pool[s], quota[s])
    return sorted(out, key=lambda r: r["example_id"])


def main():
    index = {}
    for line in open(CORPUS_ROOT / "processed" / "doc_index.jsonl"):
        d = json.loads(line)
        index[d["doc_id"]] = (d["dup_cluster"], d["article_cluster"])
    split = {}
    for line in open(CORPUS_ROOT / "processed" / "splits.jsonl"):
        d = json.loads(line)
        split[d["doc_id"]] = d["split"]

    out_root = CORPUS_ROOT / "processed" / "tasks"
    files = {}
    for t in TASKS:
        (out_root / t).mkdir(parents=True, exist_ok=True)
        files[t] = {sp: open(out_root / t / f"{sp}.jsonl", "w") for sp in ("train", "val", "test")}
    seen = set()
    counts = defaultdict(Counter)
    dropped_dup = Counter()
    test_rows = defaultdict(list)
    for source in SOURCE_ORDER:
        for line in open(CORPUS_ROOT / "interim" / f"{source}.jsonl"):
            r = json.loads(line)
            dup, art = index[r["doc_id"]]
            sp = split[r["doc_id"]]
            for task, inp, target, extra in examples(r):
                if (task, dup) in seen:
                    dropped_dup[task] += 1
                    continue
                seen.add((task, dup))
                row = {"example_id": f"{task}:{r['doc_id']}", "task": task, "doc_id": r["doc_id"],
                       "source": source, "article_cluster": art, "split": sp, "licence": r["licence"],
                       "input": inp, "target": target, "n_words_input": len(inp.split()),
                       "n_words_target": len(str(target).split()), **extra}
                files[task][sp].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[task][sp] += 1
                counts[task][f"{sp}:{source}"] += 1
                if sp == "test":
                    test_rows[task].append({k: row[k] for k in ("example_id", "source", "article_cluster")})
    for t in TASKS:
        for f in files[t].values():
            f.close()

    report = {"seed": SEED, "eval_n": EVAL_N, "tasks": {}}
    for t in TASKS:
        keep = {r["example_id"] for r in eval_subset(test_rows[t])}
        n_eval = Counter()
        with open(out_root / t / "test_eval.jsonl", "w") as f:
            for line in open(out_root / t / "test.jsonl"):
                row = json.loads(line)
                if row["example_id"] in keep:
                    f.write(line)
                    n_eval[row["source"]] += 1
        report["tasks"][t] = {"train": counts[t]["train"], "val": counts[t]["val"], "test": counts[t]["test"],
                              "test_eval": sum(n_eval.values()), "test_eval_by_source": dict(n_eval),
                              "by_split_source": {k: v for k, v in sorted(counts[t].items()) if ":" in k},
                              "dropped_as_near_duplicate": dropped_dup[t]}
    (CORPUS_ROOT / "reports" / "task_report.json").write_text(json.dumps(report, indent=1))
    for t, v in report["tasks"].items():
        print(f"{t:17} train {v['train']:>7,}  val {v['val']:>6,}  test {v['test']:>6,}  eval {v['test_eval']:>5,}  "
              f"dups dropped {v['dropped_as_near_duplicate']:>6,}  eval sources {v['test_eval_by_source']}")


if __name__ == "__main__":
    main()

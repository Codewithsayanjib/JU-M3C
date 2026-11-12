"""Order-sensitivity runs on the evaluation subsets (checklist A2). One model per process, resumable.

  python scripts/experiments/run_reliability.py --model qwen35_4b [--parts mcq,labels] [--limit 5]

Uses the frozen prompts (bench-v1) and changes only the order of what is listed in them:
  mcq     500 multiple-choice questions (fixed sample, seed 42) with the four options rotated by one,
          two and three positions, so that together with the original run the correct option
          appears once in each position.
  labels  the three classification tasks with one shuffled order of the label list (seed 42);
          for note type and note section the definitions are reordered in the same way.
Writes results/bench/reliability/{model}/mcq_rot{1,2,3}.jsonl and {task}_shuffled.jsonl.
"""
import argparse
import ast
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import run_bench  # noqa: E402  (no heavy imports at module level)

SEED, N_MCQ = 42, 500
CLS_TASKS = ("notes_specialty", "notes_doctype", "notes_section")


def options_of(ex) -> list:
    o = ex["options"]
    return ast.literal_eval(o) if isinstance(o, str) else list(o)


def rotate(ex: dict, r: int) -> dict:
    """The same question with the options moved r positions down the list (cyclically)."""
    o = options_of(ex)
    g = "ABCD".index(ex["target"])
    out = dict(ex)
    out["options"] = [o[(i - r) % 4] for i in range(4)]
    out["target"] = "ABCD"[(g + r) % 4]
    assert out["options"]["ABCD".index(out["target"])] == o[g]
    return out


def label_order(labels: list) -> list:
    """One fixed shuffled order of a label list (seed 42), different from the original."""
    out = list(labels)
    rng = random.Random(SEED)
    while out == list(labels):
        rng.shuffle(out)
    return out


def shuffle_labels(user: str) -> str:
    """Reorder the '- label' lines of the Labels block, and the 'label = ...' definition lines if
    present, into label_order(). Every other character of the prompt stays as it is."""
    lines = user.split("\n")
    i0 = lines.index("Labels:") + 1
    i1 = i0
    while i1 < len(lines) and lines[i1].startswith("- "):
        i1 += 1
    labels = [l[2:] for l in lines[i0:i1]]
    new = label_order(labels)
    lines[i0:i1] = [f"- {l}" for l in new]
    d0 = next((k for k, l in enumerate(lines) if l.startswith("Definitions")), None)
    if d0 is not None:
        d0 += 1
        d1 = d0
        while d1 < len(lines) and " = " in lines[d1]:
            d1 += 1
        by = {l.split(" = ", 1)[0]: l for l in lines[d0:d1]}
        assert set(by) == set(labels), "definitions do not match the label list"
        lines[d0:d1] = [by[l] for l in new]
    out = "\n".join(lines)
    assert sorted(out.split("\n")) == sorted(user.split("\n")) and out != user
    return out


def mcq_subset(examples: list) -> list:
    ids = sorted(e["example_id"] for e in examples)
    keep = set(random.Random(SEED).sample(ids, min(N_MCQ, len(ids))))
    return [e for e in examples if e["example_id"] in keep]


def done_ids(path: Path) -> set:
    ids = set()
    if path.exists():
        for line in open(path):
            try:
                ids.add(json.loads(line)["example_id"])
            except json.JSONDecodeError:
                continue
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", default="mcq,labels")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(ROOT / "results" / "bench" / "reliability"))
    a = ap.parse_args()
    cache = run_bench.use_ssd_cache_if_present(a.model)

    from ju_m3c.benchmark.bench_parse import parse
    from ju_m3c.benchmark.bench_prompts import build, max_new_tokens, prompt_version
    from ju_m3c.benchmark.infer import Model

    tasks_dir = run_bench.CORPUS_ROOT / "processed" / "tasks"
    load = lambda t: [json.loads(l) for l in open(tasks_dir / t / "test_eval.jsonl")]
    jobs = []  # (file name, task, variant, examples, prompt transform)
    if "mcq" in a.parts.split(","):
        base = mcq_subset(load("diagnosis_mcq"))
        for r in (1, 2, 3):
            jobs.append((f"mcq_rot{r}", "diagnosis_mcq", f"rot{r}", [rotate(e, r) for e in base], None))
    if "labels" in a.parts.split(","):
        for t in CLS_TASKS:
            jobs.append((f"{t}_shuffled", t, "labels_shuffled", load(t), shuffle_labels))

    out_dir = Path(a.out_root) / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with Model(a.model) as m:
        run_bench.pin_model_prompts(m)
        print(f"Loaded {m.repo}@{m.revision[:8]} in {m.load_s:.1f}s (cache: {cache})", flush=True)
        for name, task, variant, examples, transform in jobs:
            examples = examples[: a.limit]
            path = out_dir / f"{name}.jsonl"
            skip = done_ids(path)
            todo = [e for e in examples if e["example_id"] not in skip]
            print(f"{name}: {len(examples)} examples, {len(examples) - len(todo)} already done", flush=True)
            t0 = time.perf_counter()
            with open(path, "a") as f:
                for i, ex in enumerate(todo, 1):
                    system, user = build(task, ex)
                    if transform:
                        user = transform(user)
                    limit = max_new_tokens(task, ex)
                    g = m.generate(system, user, limit)
                    p = parse(task, g["raw_output"], ex)
                    row = {"example_id": ex["example_id"], "task": task, "variant": variant, "model_key": a.model,
                           "model_repo": m.repo, "prompt_version": prompt_version(task), "max_new_tokens": limit,
                           "target": ex["target"], "raw_output": g["raw_output"], "parsed": p["parsed"],
                           "parse_ok_strict": p["strict"], "parse_ok_lenient": p["lenient"],
                           "prompt_tokens": g["prompt_tokens"], "gen_tokens": g["gen_tokens"],
                           "finish_reason": g["finish_reason"], "latency_s": g["latency_s"],
                           "timestamp": datetime.now(timezone.utc).isoformat()}
                    if task == "diagnosis_mcq":
                        row["options"] = ex["options"]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 50 == 0 or i == len(todo):
                        print(f"  {name} {i}/{len(todo)}", flush=True)
            print(f"  {name} wall {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"Peak memory: {m.peak_memory_gb()} GB")


if __name__ == "__main__":
    main()

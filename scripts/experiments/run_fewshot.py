"""Few-shot runs on a subset of the evaluation examples (checklist A4). One model per process, resumable.

  python scripts/experiments/run_fewshot.py --model qwen35_4b [--n 300] [--limit 5]

Three solved examples are placed before the frozen zero-shot prompt (bench-v1), as input/answer pairs;
the instructions, the case and the token limits are unchanged, so each result can be compared with
the zero-shot output for the same example. The examples are the same for every model and every test
item of a task:
  - drawn from the training split (seed 42), with inputs of at most 250 words where available;
  - for the classification tasks, three different labels;
  - multiple-choice diagnosis has no training split (CUPCase is test-only), so its examples come from
    its development subset, which shares no article with the evaluation subset.
Subset: --n examples per task from the evaluation subset (seed 42); for dialogue to note only note
sections are used, because three complete visit notes as examples would not fit the prompt budget.
Writes results/bench/fewshot/{model}/{task}.jsonl and results/bench/fewshot/shots.json.
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

import run_bench  # noqa: E402

SEED, K, MAX_WORDS = 42, 3, 250
TASKS = ["diagnosis_mcq", "diagnosis_open", "notes_specialty", "notes_doctype", "notes_section",
         "summarization", "notes_summary", "title_generation", "dialogue2note"]
CLS = ("notes_specialty", "notes_doctype", "notes_section")
FS_VERSION = "fs3"


def read(task, split):
    path = run_bench.CORPUS_ROOT / "processed" / "tasks" / task / f"{split}.jsonl"
    return [json.loads(l) for l in open(path)]


def options_of(ex):
    o = ex["options"]
    return ast.literal_eval(o) if isinstance(o, str) else list(o)


def pick_shots(task: str, pool: list) -> list:
    rng = random.Random(SEED)
    pool = sorted(pool, key=lambda e: e["example_id"])
    if task == "dialogue2note":
        pool = [e for e in pool if e.get("target_kind") == "note_section"]
    short = [e for e in pool if e["n_words_input"] <= MAX_WORDS] or pool
    rng.shuffle(short)
    if task in CLS:
        out, seen = [], set()
        for e in short:
            if e["target"] not in seen:
                out.append(e); seen.add(e["target"])
            if len(out) == K:
                break
        return out
    return short[:K]


def answer_text(task: str, ex: dict) -> str:
    return f"Answer: {ex['target']}" if task == "diagnosis_mcq" else ex["target"]


def shot_input(task: str, ex: dict) -> str:
    if task == "diagnosis_mcq":
        opts = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", options_of(ex)))
        return f"{ex['input']}\n\nOptions:\n{opts}"
    return ex["input"]


def with_shots(task: str, user: str, shots: list) -> str:
    """Solved examples first, then the unchanged zero-shot prompt."""
    parts = [f"Here are {len(shots)} solved examples of the task that follows. Each shows an input and the expected answer.\n"]
    for k, s in enumerate(shots, 1):
        parts.append(f"Example {k}\nInput:\n{shot_input(task, s)}\nExpected answer:\n{answer_text(task, s)}\n")
    parts.append("Now the task.\n\n" + user)
    return "\n".join(parts)


def subset(task: str, examples: list, n: int) -> list:
    pool = sorted(examples, key=lambda e: e["example_id"])
    if task == "dialogue2note":
        pool = [e for e in pool if e.get("target_kind") == "note_section"]
    if len(pool) <= n:
        return pool
    keep = set(e["example_id"] for e in random.Random(SEED).sample(pool, n))
    return [e for e in pool if e["example_id"] in keep]


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
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(ROOT / "results" / "bench" / "fewshot"))
    a = ap.parse_args()
    cache = run_bench.use_ssd_cache_if_present(a.model)

    from ju_m3c.benchmark.bench_parse import parse
    from ju_m3c.benchmark.bench_prompts import build, max_new_tokens, prompt_version
    from ju_m3c.benchmark.infer import Model

    out_root = Path(a.out_root); out_root.mkdir(parents=True, exist_ok=True)
    shots_path = out_root / "shots.json"
    shots = json.loads(shots_path.read_text()) if shots_path.exists() else {}
    for t in a.tasks.split(","):
        if t not in shots:
            shots[t] = pick_shots(t, read(t, "dev" if t == "diagnosis_mcq" else "train"))
    shots_path.write_text(json.dumps(shots, ensure_ascii=False, indent=1))

    out_dir = out_root / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with Model(a.model) as m:
        run_bench.pin_model_prompts(m)
        print(f"Loaded {m.repo}@{m.revision[:8]} in {m.load_s:.1f}s (cache: {cache})", flush=True)
        for t in a.tasks.split(","):
            examples = subset(t, read(t, "test_eval"), a.n)[: a.limit]
            path = out_dir / f"{t}.jsonl"
            todo = [e for e in examples if e["example_id"] not in done_ids(path)]
            print(f"{t}: {len(examples)} examples, {len(examples) - len(todo)} already done", flush=True)
            t0 = time.perf_counter()
            with open(path, "a") as f:
                for i, ex in enumerate(todo, 1):
                    system, user = build(t, ex)
                    limit = max_new_tokens(t, ex)
                    g = m.generate(system, with_shots(t, user, shots[t]), limit)
                    p = parse(t, g["raw_output"], ex)
                    f.write(json.dumps({
                        "example_id": ex["example_id"], "task": t, "model_key": a.model, "model_repo": m.repo,
                        "prompt_version": f"{prompt_version(t)}+{FS_VERSION}", "shots": [s["example_id"] for s in shots[t]],
                        "max_new_tokens": limit, "raw_output": g["raw_output"], "parsed": p["parsed"],
                        "parse_ok_strict": p["strict"], "parse_ok_lenient": p["lenient"], "prompt_tokens": g["prompt_tokens"],
                        "gen_tokens": g["gen_tokens"], "finish_reason": g["finish_reason"], "latency_s": g["latency_s"],
                        "timestamp": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 50 == 0 or i == len(todo):
                        print(f"  {t} {i}/{len(todo)}", flush=True)
            print(f"  {t} wall {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"Peak memory: {m.peak_memory_gb()} GB")


if __name__ == "__main__":
    main()

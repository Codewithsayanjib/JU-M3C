"""Corpus benchmark runner (Paper 1). One model per process, resumable, flush per example.

  python scripts/benchmark/run_bench.py --model qwen35_4b --subset dev
  python scripts/benchmark/run_bench.py --model qwen35_4b --subset dev --tasks diagnosis_mcq --limit 5
  python scripts/benchmark/run_bench.py --model qwen35_4b --subset test_eval --confirm-test

Reads task files from $CORPUS_ROOT/processed/tasks/{task}/{dev|test_eval}.jsonl.
Writes results/bench/{subset}/{model}/{task}.jsonl (test_eval -> results/bench/sealed_test_eval/).
Uses the frozen inference code (src/infer.Model): greedy decoding, thinking disabled.
Model weights are read from the Hugging Face cache.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
MODEL_CACHE = CORPUS_ROOT / "models" / "hub"
TASKS = ["notes_specialty", "notes_doctype", "notes_section", "diagnosis_mcq", "diagnosis_open",
         "title_generation", "notes_summary", "summarization", "dialogue2note"]


def use_ssd_cache_if_present(model_key: str):
    """Must run before huggingface_hub is imported."""
    import yaml
    repo = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))["models"][model_key]["mlx_repo"]
    if (MODEL_CACHE / ("models--" + repo.replace("/", "--"))).is_dir():
        os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE)
        return str(MODEL_CACHE)
    return "default"


PINNED_DATE = (2026, 10, 1)  # fixed "today" for templates that print the current date


def pin_date(prompt: str) -> str:
    """Some chat templates (Llama 3.2, SmolLM3) write today's date into the system block, which
    would make prompts differ from day to day. Replace it with one fixed date, keeping each
    template's own date format (e.g. '02 Oct 2026' or '02 October 2026')."""
    import re
    from datetime import date
    d = date(*PINNED_DATE)

    def repl(m):
        long_month = re.search(r"\d{1,2} [A-Za-z]{4,} \d{4}", m.group(2)) is not None
        return m.group(1) + d.strftime("%d %B %Y" if long_month else "%d %b %Y")
    return re.sub(r"(Today Date: )(\d{1,2} [A-Za-z]+ \d{4})", repl, prompt)


def pin_model_prompts(m):
    """Wrap the loaded model's prompt formatting (instance-level; src/infer.py is not modified)."""
    original = m.format_prompt
    m.format_prompt = lambda system, user: pin_date(original(system, user))
    return m


# Token limits used before 2026-10-03, for rows written before max_new_tokens was recorded.
LEGACY_LIMITS = {"summarization": 320, "diagnosis_mcq": 8}


def file_prompt_version(path: Path) -> str | None:
    for line in open(path):
        try:
            return json.loads(line)["prompt_version"]
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def archive_path(out_root: Path, subset: str, model: str, task: str, version: str, suffix: str = "") -> Path:
    from datetime import datetime as _dt
    d = out_root / "archive" / subset / model
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{task}.{version}" + (f".{suffix}" if suffix else "") + f".{_dt.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
    return d / tag


def truncated_ids(path: Path, task: str, examples: list) -> set:
    """Rows that stopped at an older token limit lower than the current one."""
    from ju_m3c.benchmark.bench_prompts import max_new_tokens
    if not path.exists():
        return set()
    by_id = {e["example_id"]: e for e in examples}
    out = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["example_id"] not in by_id or r.get("finish_reason") != "length":
            continue
        used = r.get("max_new_tokens", LEGACY_LIMITS.get(task))
        if used is not None and used < max_new_tokens(task, by_id[r["example_id"]]):
            out.add(r["example_id"])
    return out


def done_ids(path: Path, prompt_version: str) -> set:
    ids = set()
    if path.exists():
        for line in open(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line after a crash; that example is redone
            if row["prompt_version"] != prompt_version:
                sys.exit(f"{path} has prompt version {row['prompt_version']}, not {prompt_version}.")
            ids.add(row["example_id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", default="dev", choices=["dev", "test_eval"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(ROOT / "results" / "bench"))
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--archive-old-version", action="store_true",
                    help="if a task's output file was made with an older prompt version, move it to "
                         "results/bench/archive/ and start that task fresh (nothing is deleted)")
    ap.add_argument("--redo-truncated", action="store_true",
                    help="regenerate only rows that hit an older, lower token limit; greedy decoding "
                         "makes every other row identical under the higher limit")
    a = ap.parse_args()
    if a.subset == "test_eval" and not a.confirm_test:
        sys.exit("Refusing to run the test evaluation subset without --confirm-test "
                 "(freeze prompts and scoring on dev first).")
    cache = use_ssd_cache_if_present(a.model)

    from ju_m3c.benchmark.bench_parse import parse
    from ju_m3c.benchmark.bench_prompts import build, max_new_tokens, prompt_version
    from ju_m3c.benchmark.infer import Model
    from ju_m3c.benchmark.parse import strip_thinking

    tasks = a.tasks.split(",")
    out_dir = Path(a.out_root) / ("sealed_test_eval" if a.subset == "test_eval" else a.subset) / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with Model(a.model) as m:
        pin_model_prompts(m)
        print(f"Loaded {m.repo}@{m.revision[:8]} in {m.load_s:.1f}s (cache: {cache})", flush=True)
        run = {"started": datetime.now(timezone.utc).isoformat(), "model_repo": m.repo,
               "model_revision": m.revision, "subset": a.subset, "prompt_versions": {t: prompt_version(t) for t in tasks},
               "pinned_date": "%04d-%02d-%02d" % PINNED_DATE, "load_s": round(m.load_s, 2), "platform": platform.platform(),
               "versions": {p: version(p) for p in ("mlx", "mlx-lm", "transformers")}, "tasks": {}}
        for task in tasks:
            src = CORPUS_ROOT / "processed" / "tasks" / task / f"{a.subset}.jsonl"
            examples = [json.loads(l) for l in open(src)][: a.limit]
            by_id = {e["example_id"]: e for e in examples}
            path = out_dir / f"{task}.jsonl"
            pv = prompt_version(task)
            if a.archive_old_version and path.exists():
                old_v = file_prompt_version(path)
                if old_v and old_v != pv:
                    dest = archive_path(Path(a.out_root), a.subset, a.model, task, old_v)
                    path.rename(dest)
                    print(f"{task}: archived prompt {old_v} outputs -> {dest}", flush=True)
            skip = done_ids(path, pv)
            todo = [e for e in examples if e["example_id"] not in skip]
            redo = truncated_ids(path, task, examples) if a.redo_truncated else set()
            print(f"{task}: {len(examples)} examples, {len(examples) - len(todo)} already done, "
                  f"{len(redo)} truncated rows to regenerate", flush=True)
            t0 = time.perf_counter()

            def run_one(ex):
                system, user = build(task, ex)
                limit = max_new_tokens(task, ex)
                g = m.generate(system, user, limit)
                p = parse(task, g["raw_output"], ex)
                _, stripped = strip_thinking(g["raw_output"])
                return {
                    "example_id": ex["example_id"], "doc_id": ex["doc_id"], "source": ex["source"],
                    "task": task, "subset": a.subset, "model_key": a.model, "model_repo": m.repo,
                    "prompt_version": pv, "max_new_tokens": limit, "raw_output": g["raw_output"],
                    "parsed": p["parsed"], "parse_ok_strict": p["strict"], "parse_ok_lenient": p["lenient"],
                    "stripped_thinking": stripped, "prompt_tokens": g["prompt_tokens"],
                    "gen_tokens": g["gen_tokens"], "finish_reason": g["finish_reason"],
                    "latency_s": g["latency_s"], "ttft_s": g["ttft_s"], "prefill_tps": g["prefill_tps"],
                    "decode_tps": g["decode_tps"], "timestamp": datetime.now(timezone.utc).isoformat()}

            if redo:  # rewrite the file with regenerated rows; keep a copy of the original
                original = path.read_text()
                archive_path(Path(a.out_root), a.subset, a.model, task, pv, suffix="before_redo").write_text(original)
                new_rows = {}
                for i, eid in enumerate(sorted(redo), 1):
                    new_rows[eid] = run_one(by_id[eid])
                    if i % 25 == 0 or i == len(redo):
                        print(f"  {task} redo {i}/{len(redo)}", flush=True)
                tmp = path.with_suffix(".jsonl.tmp")
                with open(tmp, "w") as f:
                    for line in original.splitlines():
                        row = json.loads(line)
                        f.write(json.dumps(new_rows.get(row["example_id"], row), ensure_ascii=False) + "\n")
                tmp.replace(path)
            with open(path, "a") as f:
                for i, ex in enumerate(todo, 1):
                    f.write(json.dumps(run_one(ex), ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 25 == 0 or i == len(todo):
                        print(f"  {task} {i}/{len(todo)}", flush=True)
            run["tasks"][task] = {"n_new": len(todo), "n_redone": len(redo), "prompt_version": pv,
                                  "wall_s": round(time.perf_counter() - t0, 1)}
        run["peak_memory_gb"] = m.peak_memory_gb()
        run["finished"] = datetime.now(timezone.utc).isoformat()
    meta_path = out_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    meta["runs"].append(run)
    meta["peak_memory_gb_max"] = max(r["peak_memory_gb"] for r in meta["runs"])
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Peak memory: {run['peak_memory_gb']} GB")


if __name__ == "__main__":
    main()

"""Prompt-variant runs (paraphrased instruction, clinical role) on the few-shot subset. Resumable.

  python scripts/experiments/run_prompt_variants.py --model qwen35_4b --style paraphrase [--limit 5]
  python scripts/experiments/run_prompt_variants.py --model qwen35_4b --style role

Both start from the frozen zero-shot prompts (bench-v1) and change one thing:
  paraphrase  the instruction sentences of the user message are reworded; the case text, label lists,
              definitions, option list, answer format, length limits and the final answer cue are
              unchanged, as is the system message.
  role        the system message gives the model a clinical role; the user message is unchanged.
Examples: the same 300 per task as the few-shot run (scripts/experiments/run_fewshot.subset, seed 42), so every
result can be paired with the zero-shot and few-shot outputs for the same example. Token limits are
the frozen ones. Task files are read from the local copy (corpus/) when present, so
model weights come from the Hugging Face Hub.
Writes results/bench/prompt_variants/{style}/{model}/{task}.jsonl.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import run_bench  # noqa: E402
from run_fewshot import TASKS, done_ids, subset  # noqa: E402

LOCAL_TASKS = Path(os.environ.get("LOCAL_TASKS", Path("corpus") / "processed" / "tasks"))
ROLE_SYSTEM = ("You are an experienced physician who writes and reviews clinical documentation. "
               "Follow the instructions exactly.")
COPY_LABEL = ("Copy exactly one label from the Labels list. Answer with the label only. No explanation.",
              "Choose one label and copy it exactly as written in the Labels list. Reply with that label alone, without explanation.")
PARAPHRASE = {
    "diagnosis_mcq": [
        ("Read the clinical case below and choose the most likely final diagnosis.\nExactly one of the four options is correct.",
         "Based on the case description that follows, pick the diagnosis that best explains the patient's presentation.\nOnly one of the four options is right."),
        ("Reply with the letter of the single correct option, in exactly this format:",
         "Give your answer as the letter of the correct option, using exactly this format:"),
        ("where <letter> is A, B, C or D. Do not explain your choice.", "with <letter> being A, B, C or D. Give no explanation.")],
    "diagnosis_open": [
        ("Read the clinical case below and state the final diagnosis.\nGive the single most specific diagnosis in a few words. "
         "Output only the diagnosis, with no explanation.",
         "What is the final diagnosis for the patient described below?\nName one diagnosis, as specific as possible, in a few words. "
         "Write only the diagnosis and nothing else.")],
    "notes_specialty": [
        ("Classify the medical specialty of the clinical note below.", "Which medical specialty does the following clinical note belong to?"),
        ("Copy exactly one label from the list; do not combine or invent labels. Answer with the label only. No explanation.",
         "Choose one label and copy it exactly as written in the list; do not merge labels or create new ones. "
         "Reply with that label alone, without explanation.")],
    "notes_doctype": [
        ("Classify the document type of the clinical note below.", "What kind of document is the following clinical note?"), COPY_LABEL],
    "notes_section": [
        ("Classify which section of a clinical note the text below belongs to.",
         "The text below was taken from one section of a clinical note. Which section is it?"), COPY_LABEL],
    "summarization": [
        ("Summarize the clinical case below in one paragraph of about 100 to 150 words. Cover the presentation, key findings, "
         "diagnosis, treatment and outcome.\nOutput only the summary, with no preamble or headings.",
         "Write a single-paragraph summary of the following clinical case, roughly 100 to 150 words long. Include how the "
         "patient presented, the main findings, the diagnosis, the treatment and the outcome.\nWrite only the summary, "
         "without any introduction or headings.")],
    "notes_summary": [
        ("Summarize the clinical note below in ONE sentence of at most 30 words.\nOutput only the sentence, with no preamble or labels.",
         "Condense the following clinical note into a SINGLE sentence of no more than 30 words.\nWrite only that sentence, "
         "without any introduction or labels.")],
    "title_generation": [
        ("Write the title of the published case report that describes the clinical case below.\nOne line, at most 25 words. "
         "Output only the title.",
         "The clinical case below was published as a case report. What was the title of that report?\nGive a single line of "
         "no more than 25 words, and write only the title.")],
}
_D2N = re.compile(r"Write the '(.+?)' section of a clinical note from the doctor-patient conversation below\. Include only "
                  r"information stated in the conversation\. Be concise\.\nOutput only the section text, with no heading\.")


def paraphrase(task: str, user: str) -> str:
    if task == "dialogue2note":
        out, n = _D2N.subn(lambda m: f"Using the doctor-patient conversation below, draft the '{m.group(1)}' section of the "
                           "clinical note. Use only what is said in the conversation and keep it brief.\nWrite only the text "
                           "of the section, without a heading.", user)
        assert n == 1, "dialogue prompt not recognised"
        return out
    out = user
    for old, new in PARAPHRASE[task]:
        assert out.count(old) == 1, f"{task}: instruction not found: {old[:50]}"
        out = out.replace(old, new)
    return out


def variant(style: str, task: str, system: str, user: str) -> tuple[str, str]:
    if style == "paraphrase":
        return system, paraphrase(task, user)
    if style == "role":
        return ROLE_SYSTEM, user
    if style == "zeroshot":  # the frozen prompt unchanged (used for the reference model on the same subsets)
        return system, user
    raise ValueError(style)


def read_eval(task):
    local = LOCAL_TASKS / task / "test_eval.jsonl"
    path = local if local.exists() else run_bench.CORPUS_ROOT / "processed" / "tasks" / task / "test_eval.jsonl"
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--style", required=True, choices=["paraphrase", "role", "zeroshot"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(ROOT / "results" / "bench" / "prompt_variants"))
    a = ap.parse_args()
    cache = run_bench.use_ssd_cache_if_present(a.model)

    from ju_m3c.benchmark.bench_parse import parse
    from ju_m3c.benchmark.bench_prompts import build, max_new_tokens, prompt_version
    from ju_m3c.benchmark.infer import Model

    out_dir = Path(a.out_root) / a.style / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(t, subset(t, read_eval(t), a.n)[: a.limit]) for t in a.tasks.split(",")]
    with Model(a.model) as m:
        run_bench.pin_model_prompts(m)
        print(f"Loaded {m.repo}@{m.revision[:8]} in {m.load_s:.1f}s (cache: {cache}) style={a.style}", flush=True)
        for t, examples in jobs:
            path = out_dir / f"{t}.jsonl"
            todo = [e for e in examples if e["example_id"] not in done_ids(path)]
            print(f"{t}: {len(examples)} examples, {len(examples) - len(todo)} already done", flush=True)
            t0 = time.perf_counter()
            with open(path, "a") as f:
                for i, ex in enumerate(todo, 1):
                    system, user = variant(a.style, t, *build(t, ex))
                    limit = max_new_tokens(t, ex)
                    g = m.generate(system, user, limit)
                    p = parse(t, g["raw_output"], ex)
                    f.write(json.dumps({
                        "example_id": ex["example_id"], "task": t, "style": a.style, "model_key": a.model,
                        "model_repo": m.repo, "prompt_version": f"{prompt_version(t)}+{a.style}", "max_new_tokens": limit,
                        "raw_output": g["raw_output"], "parsed": p["parsed"], "parse_ok_strict": p["strict"],
                        "parse_ok_lenient": p["lenient"], "prompt_tokens": g["prompt_tokens"], "gen_tokens": g["gen_tokens"],
                        "finish_reason": g["finish_reason"], "latency_s": g["latency_s"],
                        "timestamp": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 50 == 0 or i == len(todo):
                        print(f"  {t} {i}/{len(todo)}", flush=True)
            print(f"  {t} wall {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"Peak memory: {m.peak_memory_gb()} GB")


if __name__ == "__main__":
    main()

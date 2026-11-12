"""Leakage-impact experiment (checklist A1): does training on documents that share an article with
the test items inflate test scores?

  python scripts/experiments/leakage_experiment.py build          # training sets (needs $CORPUS_ROOT)
  python scripts/experiments/leakage_experiment.py train leaky    # LoRA fine-tuning, one run per condition
  python scripts/experiments/leakage_experiment.py train clean
  python scripts/experiments/leakage_experiment.py eval leaky     # generate on the 1,000 title evaluation items
  python scripts/experiments/leakage_experiment.py eval clean
  python scripts/experiments/leakage_experiment.py eval base      # the same model without fine-tuning
  python scripts/experiments/leakage_experiment.py score

Task: title generation. 232 of the 1,000 evaluation items have a document of the same article in
another source (337 partner documents in our test split; 321 of them carry the identical title). A
naive merge of the public datasets, without article-level splitting, would put such partners in the
training data. Both conditions train on the same 2,000 random training examples (our leak-free
training split) plus 337 more: the leaky condition adds the 337 partner documents, the clean
condition adds 337 further random training examples. Same model, prompt, hyperparameters and seed.
Effect = (leaky - clean) on the 232 leaked items, against (leaky - clean) on the 768 others.
Prompts are the frozen zero-shot title prompts (bench-v1). Outputs: results/leakage/.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

MODEL = "granite33_2b"
TASK = "title_generation"
SEED, N_BASE, N_VALID = 42, 2000, 100
OUT = ROOT / "results" / "leakage"
CORPUS = Path(os.environ.get("CORPUS_ROOT", "corpus"))
LORA = {"fine_tune_type": "lora", "num_layers": 16, "batch_size": 4, "learning_rate": 1e-4, "iters": 1170,
        "max_seq_length": 1536, "mask_prompt": True, "grad_checkpoint": True, "seed": SEED,
        "steps_per_report": 25, "steps_per_eval": 200, "save_every": 200, "val_batches": 25,
        "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0}}


def read(split):
    return [json.loads(l) for l in open(CORPUS / "processed" / "tasks" / TASK / f"{split}.jsonl")]


def chat(ex):
    from ju_m3c.benchmark.bench_prompts import build
    system, user = build(TASK, ex)
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                         {"role": "assistant", "content": ex["target"]}]}


def build_sets():
    ev = read("test_eval")
    ev_cl = {e["article_cluster"] for e in ev}
    ev_doc = {e["doc_id"] for e in ev}
    partners = sorted((e for e in read("test") if e["article_cluster"] in ev_cl and e["doc_id"] not in ev_doc),
                      key=lambda e: e["example_id"])
    leaked = sorted({e["example_id"] for e in ev if e["article_cluster"] in {p["article_cluster"] for p in partners}})
    rng = random.Random(SEED)
    train = sorted(read("train"), key=lambda e: e["example_id"])
    pick = rng.sample(range(len(train)), N_BASE + len(partners))
    base = [train[i] for i in pick[:N_BASE]]
    extra = [train[i] for i in pick[N_BASE:]]
    valid = rng.sample(sorted(read("val"), key=lambda e: e["example_id"]), N_VALID)
    for cond, add in (("leaky", partners), ("clean", extra)):
        d = OUT / "data" / cond
        d.mkdir(parents=True, exist_ok=True)
        rows = base + add
        random.Random(SEED).shuffle(rows)
        (d / "train.jsonl").write_text("".join(json.dumps(chat(e), ensure_ascii=False) + "\n" for e in rows))
        (d / "valid.jsonl").write_text("".join(json.dumps(chat(e), ensure_ascii=False) + "\n" for e in valid))
    (OUT / "data" / "eval.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ev))
    meta = {"task": TASK, "model": MODEL, "seed": SEED, "n_base": N_BASE, "n_partners": len(partners),
            "partners_same_title": sum(p["target"].strip().lower() == next(e["target"] for e in ev if e["article_cluster"] == p["article_cluster"]).strip().lower() for p in partners),
            "n_eval": len(ev), "leaked_eval_ids": leaked, "partner_ids": [p["example_id"] for p in partners]}
    (OUT / "data" / "meta.json").write_text(json.dumps(meta, indent=1))
    print({k: v for k, v in meta.items() if not k.endswith("_ids")}, "leaked eval items:", len(leaked))


def train(cond, iters=None):
    import yaml
    from run_bench import use_ssd_cache_if_present
    use_ssd_cache_if_present(MODEL)
    repo = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))["models"][MODEL]
    cfg = dict(LORA, model=repo["mlx_repo"], train=True, data=str(OUT / "data" / cond),
               adapter_path=str(OUT / "adapters" / cond))
    if iters:
        cfg["iters"] = iters
    d = OUT / "adapters" / cond
    d.mkdir(parents=True, exist_ok=True)
    # resume after an interruption: continue from the last saved checkpoint for the remaining steps
    ckpts = sorted(d.glob("[0-9]*_adapters.safetensors"))
    if ckpts:
        done = int(ckpts[-1].name.split("_")[0])
        cfg["resume_adapter_file"] = str(ckpts[-1])
        cfg["iters"] = cfg["iters"] - done
        cfg["adapter_path"] = str(d / f"resume_from_{done}")
        print(f"resuming {cond} from step {done}, {cfg['iters']} steps left", flush=True)
    path = OUT / "adapters" / f"{cond}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "-c", str(path)], check=True, env=os.environ.copy())
    final = Path(cfg["adapter_path"]) / "adapters.safetensors"
    if final.parent != d:  # resumed run: its final weights become the condition's adapter
        import shutil
        shutil.copy(final, d / "adapters.safetensors")
    (d / "DONE").write_text(f"steps {LORA['iters'] if not iters else iters}\n")


def evaluate(cond, limit=None):
    from run_bench import pin_model_prompts, use_ssd_cache_if_present
    use_ssd_cache_if_present(MODEL)
    from ju_m3c.benchmark.bench_parse import parse
    from ju_m3c.benchmark.bench_prompts import build, max_new_tokens
    from ju_m3c.benchmark.infer import Model
    ev = [json.loads(l) for l in open(OUT / "data" / "eval.jsonl")]
    path = OUT / "outputs" / f"{cond}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["example_id"] for l in open(path)} if path.exists() else set()
    adapter = None if cond == "base" else str(OUT / "adapters" / cond)
    with Model(MODEL, adapter_path=adapter) as m, open(path, "a") as f:
        pin_model_prompts(m)
        for i, ex in enumerate([e for e in ev if e["example_id"] not in done][:limit], 1):
            system, user = build(TASK, ex)
            g = m.generate(system, user, max_new_tokens(TASK, ex))
            p = parse(TASK, g["raw_output"], ex)
            f.write(json.dumps({"example_id": ex["example_id"], "condition": cond, "raw_output": g["raw_output"],
                                "parsed": p["parsed"], "finish_reason": g["finish_reason"], "latency_s": g["latency_s"]},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(cond, i, flush=True)


def score():
    import numpy as np
    from score_bench import norm
    from ju_m3c.benchmark.metrics import rouge_l_per_note
    meta = json.loads((OUT / "data" / "meta.json").read_text())
    leaked = set(meta["leaked_eval_ids"])
    ev = {e["example_id"]: e for e in map(json.loads, open(OUT / "data" / "eval.jsonl"))}
    res = {}
    for cond in ("base", "clean", "leaky"):
        p = OUT / "outputs" / f"{cond}.jsonl"
        if not p.exists():
            continue
        rows = {r["example_id"]: r for r in map(json.loads, open(p))}
        ids = sorted(ev)
        r = rouge_l_per_note([rows[i]["parsed"] or "" for i in ids], [ev[i]["target"] for i in ids])
        ex = [norm(rows[i]["parsed"] or "") == norm(ev[i]["target"]) for i in ids]
        res[cond] = {i: (a, b) for i, a, b in zip(ids, r, ex)}
    rng = np.random.default_rng(SEED)
    out = {}
    for grp, sel in (("leaked", lambda i: i in leaked), ("not_leaked", lambda i: i not in leaked)):
        ids = [i for i in sorted(ev) if sel(i)]
        for cond in res:
            out[f"{grp}_{cond}_rougeL"] = float(np.mean([res[cond][i][0] for i in ids]))
            out[f"{grp}_{cond}_exact"] = float(np.mean([res[cond][i][1] for i in ids]))
        if {"leaky", "clean"} <= set(res):
            d = np.array([res["leaky"][i][0] - res["clean"][i][0] for i in ids])
            b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            out[f"{grp}_leaky_minus_clean"] = float(d.mean())
            out[f"{grp}_ci"] = [float(x) for x in np.percentile(b, [2.5, 97.5])]
        out[f"{grp}_n"] = len(ids)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "build":
        build_sets()
    elif step == "train":
        train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif step == "eval":
        evaluate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif step == "score":
        score()

"""Zero-shot VLM runs on the multimodal items (MLX, mlx-vlm, greedy).

Conditions
  mm1:  case text + 4 images  -> letter A-D
  mm2:  text  (case text + options, no image)
        image (images + options, no case text)
        both  (case text + images + options)
        options (options only: no case text, no image; floor for option-only guessing)
Captions are never shown. Images are given as ONE composite picture (mm1: 2x2 grid with
large A-D labels; mm2: up to 3 figures side by side), because small VLMs place all image
tokens together and cannot tell which of several separate images carries which letter.

Resumable: item ids already in the output file are skipped; one line per item, flushed.
Output: {mm}/results/{model_key}/{task}_{condition}_{split}.jsonl

Usage:
  python scripts/mm/run_vlm.py --model qwen25vl_3b --task mm1 --split dev \
      --mm $JUMC_MV/full [--limit 20]
"""
import argparse, ast, gc, json, re, time
from datetime import datetime, timezone
from pathlib import Path

PROMPT_VERSION = "mm-v2"
MODELS = {  # exact mlx-community repos (checked on Hugging Face 2026-11-05)
    "qwen25vl_3b": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "internvl3_2b": "mlx-community/InternVL3-2B-4bit",
    "qwen25vl_7b": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",  # larger reference, same family
}

MM1 = ("Below is a clinical case description and one picture showing four medical images in a 2x2 "
       "grid, labelled A (top left), B (top right), C (bottom left) and D (bottom right). "
       "Exactly one of them comes from this patient's case report.\n\n"
       "Case:\n{case}\n\nWhich image (A, B, C or D) belongs to this case? Answer with one letter only.")
MM2_TEXT = ("Read the clinical case and choose the most likely final diagnosis.\n\nCase:\n{case}\n\n"
            "Options:\n{opts}\n\nAnswer with one letter only (A, B, C or D).")
MM2_IMAGE = ("The image(s) come from a clinical case report. Based on the image(s) only, choose the most "
             "likely final diagnosis.\n\nOptions:\n{opts}\n\nAnswer with one letter only (A, B, C or D).")
MM2_OPTIONS = ("Choose the most likely final diagnosis of a clinical case report.\n\n"
               "Options:\n{opts}\n\nAnswer with one letter only (A, B, C or D).")
MM2_BOTH = ("Read the clinical case and look at the image(s) from the same case, then choose the most "
            "likely final diagnosis.\n\nCase:\n{case}\n\nOptions:\n{opts}\n\n"
            "Answer with one letter only (A, B, C or D).")


def fmt_opts(o):
    if isinstance(o, str):
        o = ast.literal_eval(o)
    return "\n".join(f"{l}. {t}" for l, t in zip("ABCD", o))


def parse(out):
    s = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    strict = s.strip(" .*()") if s.strip(" .*()") in list("ABCD") else None
    if strict:
        return strict, strict
    m = (re.match(r"^\W*([ABCD])\b(?![ -]+[A-Za-z]{2})", s)  # leading letter, not the article "A CT ..."
         or re.search(r"(?:answer|option|image)\s*(?:is|:)?\s*\(?\**([ABCD])\b", s, re.I)
         or re.search(r"\bis\s*:?\s*\(?\*\*([ABCD])\b", s)  # "... is **C**"
         or re.search(r"\b([ABCD])[.)]?\s*$", s))
    return None, (m.group(1).upper() if m else None)


def composite(paths, grid):
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for p in paths]
    cell = 512
    if grid:
        canvas = Image.new("RGB", (2 * cell, 2 * cell), "white")
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 64)
        except OSError:
            font = ImageFont.load_default()
        for k, im in enumerate(ims):
            im.thumbnail((cell - 16, cell - 16))
            x, y = (k % 2) * cell, (k // 2) * cell
            canvas.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
            d = ImageDraw.Draw(canvas)
            d.rectangle([x + 4, y + 4, x + 84, y + 84], fill="white", outline="black", width=3)
            d.text((x + 22, y + 10), "ABCD"[k], fill="black", font=font)
            d.rectangle([x, y, x + cell - 1, y + cell - 1], outline="black", width=2)
    else:
        for im in ims:
            im.thumbnail((cell, cell))
        canvas = Image.new("RGB", (sum(i.width for i in ims) + 8 * (len(ims) - 1), max(i.height for i in ims)), "white")
        x = 0
        for im in ims:
            canvas.paste(im, (x, 0)); x += im.width + 8
    return canvas


def build(item, task, cond):
    if task == "mm1":
        return MM1.format(case=item["case_text"]), item["images"]
    opts = fmt_opts(item["options"])
    if cond == "text":
        return MM2_TEXT.format(case=item["case_text"], opts=opts), []
    if cond == "options":
        return MM2_OPTIONS.format(opts=opts), []
    if cond == "image":
        return MM2_IMAGE.format(opts=opts), item["images"]
    return MM2_BOTH.format(case=item["case_text"], opts=opts), item["images"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--task", required=True, choices=["mm1", "mm2"])
    ap.add_argument("--conditions", default=None, help="mm2: comma list of text,image,both,options")
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--mm", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    mm = Path(a.mm)
    items = [json.loads(l) for l in open(mm / "items" / f"{a.task}_{a.split}.jsonl")]
    if a.limit:
        items = items[: a.limit]
    conds = ["all"] if a.task == "mm1" else (a.conditions or "text,image,both,options").split(",")

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    repo = MODELS[a.model]
    model, proc = load(repo)
    res_dir = mm / "results" / a.model; res_dir.mkdir(parents=True, exist_ok=True)

    for cond in conds:
        f_out = res_dir / f"{a.task}_{cond}_{a.split}.jsonl"
        done = {json.loads(l)["item_id"] for l in open(f_out)} if f_out.exists() else set()
        todo = [it for it in items if it["item_id"] not in done]
        print(f"{a.model} {a.task} {cond} {a.split}: {len(todo)} to run", flush=True)
        n_ok = 0
        with open(f_out, "a") as fo:
            for it in todo:
                text, imgs = build(it, a.task, cond)
                paths = []
                if imgs:
                    cdir = mm / "items" / "composites" / a.task; cdir.mkdir(parents=True, exist_ok=True)
                    cp = cdir / f"{it['item_id'].replace(':', '_')}.jpg"
                    if not cp.exists():
                        composite([mm / "items" / "images" / i for i in imgs], a.task == "mm1").save(cp, quality=90)
                    paths = [str(cp)]
                prompt = apply_chat_template(proc, model.config, text, num_images=len(paths))
                t0 = time.time()
                r = generate(model, proc, prompt, paths or None, max_tokens=a.max_tokens,
                             temperature=0.0, verbose=False)
                strict, lenient = parse(r.text)
                n_ok += lenient == it["answer"]
                fo.write(json.dumps({
                    "item_id": it["item_id"], "model_key": a.model, "model_repo": repo,
                    "task": a.task, "condition": cond, "split": a.split,
                    "prompt_version": PROMPT_VERSION, "raw_output": r.text,
                    "parsed_strict": strict, "parsed_lenient": lenient, "answer": it["answer"],
                    "correct": lenient == it["answer"], "n_images": len(imgs),
                    "prompt_tokens": r.prompt_tokens, "gen_tokens": r.generation_tokens,
                    "latency_s": round(time.time() - t0, 3), "peak_memory_gb": r.peak_memory,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                fo.flush()
        if todo:
            print(f"  accuracy so far (lenient): {n_ok / len(todo):.3f}", flush=True)

    del model, proc; gc.collect(); mx.clear_cache()


if __name__ == "__main__":
    main()

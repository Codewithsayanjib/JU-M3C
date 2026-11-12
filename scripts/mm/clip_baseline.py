"""BiomedCLIP zero-shot baseline for MM-1 (image-case matching).

1. 4-way: same items as the VLMs; pick the image with the highest text-image similarity.
2. Retrieval: case text against the pool of all gold images in the split
   (Recall@1, Recall@5, Recall@10, MRR).

Runs in the separate .venv-clip (requirements-clip.txt), so the MLX environment is untouched.

Usage:
  .venv-clip/bin/python scripts/mm/clip_baseline.py --mm $JUMC_MV/full --split dev
"""
import argparse, json
from pathlib import Path
import open_clip, torch
from PIL import Image

REPO = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    a = ap.parse_args()
    mm = Path(a.mm)
    items = [json.loads(l) for l in open(mm / "items" / f"mm1_{a.split}.jsonl")]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(REPO)
    tok = open_clip.get_tokenizer(REPO)
    model = model.to(dev).eval()

    @torch.no_grad()
    def enc_img(names):
        x = torch.stack([pre(Image.open(mm / "items" / "images" / n).convert("RGB")) for n in names]).to(dev)
        return torch.nn.functional.normalize(model.encode_image(x), dim=-1)

    @torch.no_grad()
    def enc_txt(texts):
        return torch.nn.functional.normalize(model.encode_text(tok(texts, context_length=256).to(dev)), dim=-1)

    out_dir = mm / "results" / "biomedclip"; out_dir.mkdir(parents=True, exist_ok=True)
    rows, gold_img, txt = [], [], []
    for i in range(0, len(items), 32):
        batch = items[i:i + 32]
        t = enc_txt([it["case_text"] for it in batch])
        txt.append(t)
        for it, tv in zip(batch, t):
            iv = enc_img(it["images"])
            sims = (iv @ tv).tolist()
            pred = "ABCD"[max(range(4), key=lambda k: sims[k])]
            gi = "ABCD".index(it["answer"])
            gold_img.append(iv[gi])
            rows.append({"item_id": it["item_id"], "model_key": "biomedclip", "task": "mm1",
                         "split": a.split, "sims": [round(s, 4) for s in sims],
                         "parsed_lenient": pred, "answer": it["answer"], "correct": pred == it["answer"]})
    with open(out_dir / f"mm1_all_{a.split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    T, I = torch.cat(txt), torch.stack(gold_img)
    S = T @ I.T
    rank = (S > S.diag().unsqueeze(1)).sum(1) + 1
    res = {"split": a.split, "n": len(items),
           "acc_4way": round(sum(r["correct"] for r in rows) / len(rows), 4),
           "pool_size": len(items),
           **{f"recall@{k}": round((rank <= k).float().mean().item(), 4) for k in (1, 5, 10)},
           "mrr": round((1.0 / rank.float()).mean().item(), 4)}
    json.dump(res, open(out_dir / f"mm1_summary_{a.split}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

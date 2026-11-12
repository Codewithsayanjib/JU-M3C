"""Experiment C: similar-case retrieval with the PMC-Patients patient-patient similarity labels.

Queries: PMC-Patients documents in our TEST split with at least one similar patient from a
different article. Pool: all PMC-Patients documents in the corpus. Same-article candidates are
removed from each query's ranking (they would be trivial matches).

Methods: TF-IDF (sparse text), BiomedCLIP text (dense text), BiomedCLIP image (mean of the first
two figures of each article), text->image (query text vs candidate images), MeSH (cosine on MeSH
descriptor sets), and reciprocal-rank fusion. Metrics: MRR@10, nDCG@10, Recall@100.

Embeddings are cached in --work. Runs in .venv-clip.

Usage:
  .venv-clip/bin/python scripts/mm/similar_cases.py --data $JUMC_DATA \
      --mm $JUMC_MV/full \
      --ppr $JUMC_MV/external/pmc_patients/PMC-Patients.csv \
      --work $JUMC_MV/full/retrieval --out reports/similar_cases.json
"""
import argparse, ast, csv, gzip, io, json, math, sys, tarfile, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
K = 100


DEVICE = "cpu"  # MPS stalled on long embedding runs (blocked in .cpu() after a few thousand batches)


def load_clip():
    import open_clip, torch
    dev = DEVICE
    model, _, pre = open_clip.create_model_and_transforms(REPO)
    return model.to(dev).eval(), pre, open_clip.get_tokenizer(REPO), dev, torch


def embed_text(texts, path):
    if path.exists():
        return np.load(path)
    model, _, tok, dev, torch = load_clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            x = tok(texts[i:i + 64], context_length=256).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_text(x), dim=-1).cpu().numpy().astype(np.float16))
            if i % 6400 == 0:
                print("text", i, flush=True)
    E = np.concatenate(out); np.save(path, E); return E


def embed_images(pmcids, mm, path):
    """Mean embedding of the first two figures per article; zero vector if none."""
    if path.exists():
        return np.load(path)
    from PIL import Image
    model, pre, _, dev, torch = load_clip()
    want = set(pmcids); per_shard = defaultdict(list)
    for l in open(mm / "figures.jsonl"):
        r = json.loads(l)
        if r["pmcid"] in want and r.get("image_path") and r["fig_index"] < 2:
            per_shard[r["shard"]].append((r["pmcid"], r["image_path"]))
    acc = defaultdict(list); t0 = time.time(); done = 0
    with torch.no_grad():
        for si, (shard, rows) in enumerate(sorted(per_shard.items())):
            names = {n: p for p, n in rows}; imgs, keys = [], []
            with tarfile.open(mm / "shards" / shard) as tf:
                for m in tf:
                    if m.name in names:
                        try:
                            imgs.append(pre(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))); keys.append(names[m.name])
                        except Exception:
                            pass
            for i in range(0, len(imgs), 128):
                v = torch.nn.functional.normalize(model.encode_image(torch.stack(imgs[i:i + 128]).to(dev)), dim=-1).cpu().numpy()
                for k, e in zip(keys[i:i + 128], v):
                    acc[k].append(e)
            done += len(imgs)
            if si % 20 == 0:
                print(f"shard {si}/{len(per_shard)} images {done} {done / (time.time() - t0):.0f}/s", flush=True)
    idx = {p: i for i, p in enumerate(pmcids)}
    E = np.zeros((len(pmcids), 512), np.float16)
    for p, vs in acc.items():
        v = np.mean(vs, 0); E[idx[p]] = v / np.linalg.norm(v)
    np.save(path, E); return E


def topk_dense(Q, C, mask_fn, k=K):
    res = []
    Cm = C.astype(np.float32)
    for i in range(0, len(Q), 256):
        S = Q[i:i + 256].astype(np.float32) @ Cm.T
        for j in range(S.shape[0]):
            res.append(mask_fn(i + j, S[j], k))
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "mm", "ppr", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    data, mm, work = Path(a.data).expanduser(), Path(a.mm), Path(a.work); work.mkdir(parents=True, exist_ok=True)

    docs = []
    for l in gzip.open(data / "corpus/records.jsonl.gz", "rt"):
        r = json.loads(l)
        if r["source"] == "pmc_patients":
            docs.append((r["source_id"], r["pmcid"], r["split"], r["text"]))
    uid = [d[0] for d in docs]; pos = {u: i for i, u in enumerate(uid)}
    art = np.array([u.split("-")[0] for u in uid])
    print(len(docs), "pool docs", flush=True)

    csv.field_size_limit(sys.maxsize)
    rel = {}
    for r in csv.DictReader(open(a.ppr, newline="")):
        u = r["patient_uid"]
        if u in pos and docs[pos[u]][2] == "test":
            sim = ast.literal_eval(r["similar_patients"]) if r["similar_patients"] else {}
            R = {pos[v] for v in sim if v in pos and v.split("-")[0] != u.split("-")[0]}
            if R:
                rel[pos[u]] = R
    qs = sorted(rel)
    print(len(qs), "queries", flush=True)

    def mask(qi, s, k):
        s = s.copy(); s[art == art[qs[qi]]] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        return [t for t in top if np.isfinite(s[t])]

    T = embed_text([d[3] for d in docs], work / "text_biomedclip.npy")
    pmcs = sorted(set(d[1] for d in docs)); pidx = {p: i for i, p in enumerate(pmcs)}
    I_art = embed_images(pmcs, mm, work / "image_biomedclip.npy")
    import gc; gc.collect()

    runs = {}
    tf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.5, dtype=np.float32)
    X = tf.fit_transform([d[3] for d in docs])
    Xq = X[qs]
    runs["tfidf"] = []
    for i in range(0, len(qs), 256):
        S = (Xq[i:i + 256] @ X.T).toarray()
        for j in range(S.shape[0]):
            runs["tfidf"].append(mask(i + j, S[j], K))
    del X, Xq; gc.collect()
    print("tfidf done", flush=True)

    runs["dense_text"] = topk_dense(T[qs], T, mask)
    print("dense text done", flush=True)

    I = I_art[[pidx[d[1]] for d in docs]]
    has_img = np.abs(I.astype(np.float32)).sum(1) > 0
    def mask_img(qi, s, k):
        s = s.copy(); s[~has_img] = -np.inf
        return mask(qi, s, k)
    runs["image"] = topk_dense(I[qs], I, mask_img)
    runs["text_to_image"] = topk_dense(T[qs], I, mask_img)
    print("image done", flush=True)

    mesh = {r["pmcid"]: [m["ui"] for m in r["mesh"]] for r in map(json.loads, open(mm / "mesh.jsonl"))}
    vocab = {}
    rows, cols = [], []
    for i, d in enumerate(docs):
        for u in mesh.get(d[1], []):
            rows.append(i); cols.append(vocab.setdefault(u, len(vocab)))
    M = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(docs), len(vocab)))
    nrm = np.sqrt(np.asarray(M.sum(1)).ravel()); nrm[nrm == 0] = 1
    M = sparse.diags(1 / nrm) @ M
    has_mesh = np.asarray(M.sum(1)).ravel() > 0
    runs["mesh"] = []
    for i in range(0, len(qs), 256):
        S = (M[qs[i:i + 256]] @ M.T).toarray()
        for j in range(S.shape[0]):
            s = S[j]; s[~has_mesh] = -np.inf
            runs["mesh"].append(mask(i + j, s, K))

    def rrf(names):
        out = []
        for qi in range(len(qs)):
            sc = defaultdict(float)
            for n in names:
                for r, d in enumerate(runs[n][qi]):
                    sc[d] += 1 / (60 + r + 1)
            out.append(sorted(sc, key=lambda d: -sc[d])[:K])
        return out
    runs["fusion_text"] = rrf(["tfidf", "dense_text"])
    runs["fusion_text_image"] = rrf(["tfidf", "dense_text", "image"])
    runs["fusion_text_image_mesh"] = rrf(["tfidf", "dense_text", "image", "mesh"])

    def metrics(sub):
        mrr = ndcg = rec = 0.0
        for qi in sub:
            R, ranked = rel[qs[qi]], runs_cur[qi]
            for r, d in enumerate(ranked[:10]):
                if d in R:
                    mrr += 1 / (r + 1); break
            dcg = sum(1 / math.log2(r + 2) for r, d in enumerate(ranked[:10]) if d in R)
            ndcg += dcg / sum(1 / math.log2(r + 2) for r in range(min(10, len(R))))
            rec += len(R & set(ranked[:100])) / len(R)
        n = len(sub)
        return {"n": n, "mrr@10": round(mrr / n, 4), "ndcg@10": round(ndcg / n, 4), "recall@100": round(rec / n, 4)}

    q_img = [qi for qi, q in enumerate(qs) if has_img[q]]
    q_mesh = [qi for qi, q in enumerate(qs) if has_mesh[q]]
    res = {"pool": len(docs), "pool_with_image": int(has_img.sum()), "pool_with_mesh": int(has_mesh.sum()),
           "queries": len(qs), "mean_relevant": round(float(np.mean([len(rel[q]) for q in qs])), 2), "methods": {}}
    for n in runs:
        runs_cur = runs[n]
        res["methods"][n] = {"all": metrics(range(len(qs))), "queries_with_image": metrics(q_img),
                             "queries_with_mesh": metrics(q_mesh)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for n, v in res["methods"].items():
        print(f"{n:24s}", v["all"], "img", v["queries_with_image"]["ndcg@10"], "mesh", v["queries_with_mesh"]["ndcg@10"])
    print("SIM_DONE", flush=True)


if __name__ == "__main__":
    main()

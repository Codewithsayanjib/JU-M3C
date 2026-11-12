"""Full-corpus variant of link_prediction.py: all PMC articles from every source (PMC-Patients, MultiCaRe,
MedCaseReasoning, CaseReportBench), so the text- and image-duplicate layers are populated. BiomedCLIP
features exist only for PMC-Patients articles and are zero with an availability flag elsewhere.
Adds a leave-one-layer-out ablation.

Link prediction on the multi-layer network (2g): do the layers carry complementary information?

Nodes: PMC-Patients articles that have major MeSH topics (the pool used in Experiment C, whose BiomedCLIP
embeddings are cached). Target layer: concept links (two articles share >= 2 major MeSH topics).
Concept edges are split 70 / 10 / 20 (seed 42): the 70% forms the observed concept layer, 10% are the
classifier's training positives, 20% are held-out test positives. Each positive set gets the same number
of negatives: half random article pairs, half hard (2 hops apart in the observed concept layer, not linked).

Features per article pair:
  concept  - common neighbours, Adamic-Adar, Jaccard in the observed concept layer
  text     - TF-IDF cosine, BiomedCLIP text cosine (article means), near-duplicate-text flag
  image    - BiomedCLIP image cosine, duplicate-image flag
  cross    - paths that switch layer: article -(text or image duplicate)- k -(concept)- article
Models (logistic regression, standardised): each layer alone; "flattened" (CN / AA / Jaccard on one graph
that merges all layers' edges, layer identity discarded); multi-layer (all per-layer features).
Metrics: ROC-AUC and average precision on the held-out pairs, bootstrap 95% CI, overall and on the hard half.

  .venv-clip/bin/python scripts/experiments/link_prediction.py --data $JUMC_DATA \
      --mm $JUMC_MV/full \
      --pairs $JUMC_MV/reports/mm_leakage_full.pairs.jsonl \
      --out $JUMC_MV/reports/link_prediction.json
"""
import argparse, gzip, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

SEED = 42


def adj(n, E):
    E = np.asarray(list(E), dtype=np.int64).reshape(-1, 2)
    A = sparse.coo_matrix((np.ones(len(E), np.float32), (E[:, 0], E[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    return A


def struct(A, P):
    """common neighbours, Adamic-Adar, Jaccard for pairs P on graph A."""
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.where(deg > 1, 1 / np.log(np.maximum(deg, 2)), 0).astype(np.float32)
    Ai, Aj = A[P[:, 0]], A[P[:, 1]]
    cn = np.asarray(Ai.multiply(Aj).sum(1)).ravel()
    aa = np.asarray(Ai.multiply(Aj).multiply(inv[None, :]).sum(1)).ravel()
    un = deg[P[:, 0]] + deg[P[:, 1]] - cn
    jac = np.where(un > 0, cn / np.maximum(un, 1), 0)
    return np.c_[cn, aa, jac]


def cos(E, P):
    a, b = E[P[:, 0]].astype(np.float32), E[P[:, 1]].astype(np.float32)
    return (a * b).sum(1)


def boot(y, s, mask=None, n=1000):
    rng = np.random.default_rng(SEED)
    if mask is not None:
        y, s = y[mask], s[mask]
    au, ap = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        au.append(roc_auc_score(y[i], s[i])); ap.append(average_precision_score(y[i], s[i]))
    return {"auc": round(float(roc_auc_score(y, s)), 4), "auc_ci95": [round(float(np.percentile(au, q)), 4) for q in (2.5, 97.5)],
            "ap": round(float(average_precision_score(y, s)), 4), "ap_ci95": [round(float(np.percentile(ap, q)), 4) for q in (2.5, 97.5)]}


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "mm", "pairs", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    data, mm = Path(a.data).expanduser(), Path(a.mm)
    rng = np.random.default_rng(SEED)

    # PMC-Patients documents, in the order used for the cached embeddings (similar_cases.py)
    docs, alld = [], []
    for l in gzip.open(data / "corpus/records.jsonl.gz", "rt"):
        r = json.loads(l)
        if r["source"] == "pmc_patients":
            docs.append((r["doc_id"], r["pmcid"], r["text"] or ""))
        if r["pmcid"]:
            p = str(r["pmcid"]); alld.append((r["doc_id"], p if p.startswith("PMC") else "PMC" + p, r["text"] or ""))
    T = np.load(mm / "retrieval/text_biomedclip.npy", mmap_mode="r")
    assert len(T) == len(docs), "text embedding order mismatch"
    pmcs_all = sorted(set(d[1] for d in docs))
    I = np.load(mm / "retrieval/image_biomedclip.npy", mmap_mode="r")
    assert len(I) == len(pmcs_all), "image embedding order mismatch"
    ipos = {p: i for i, p in enumerate(pmcs_all)}

    mesh = {}
    for l in open(mm / "mesh.jsonl"):
        r = json.loads(l)
        mj = {m["ui"] for m in r["mesh"] if m["major"]}
        if mj:
            mesh[r["pmcid"]] = mj
    norm = lambda p: p if str(p).startswith("PMC") else "PMC" + str(p)
    arts = sorted(p for p in set(d[1] for d in alld) if p in mesh)
    n = len(arts); apos = {p: i for i, p in enumerate(arts)}
    print(n, "articles with major MeSH", flush=True)

    # article-level text: mean BiomedCLIP text embedding; concatenated text for TF-IDF
    by_art = defaultdict(list)
    for k, d in enumerate(docs):
        if norm(d[1]) in apos:
            by_art[apos[norm(d[1])]].append(k)
    TE = np.zeros((n, T.shape[1]), np.float32); txt = [""] * n
    for i, ks in by_art.items():
        TE[i] = np.asarray(T[ks], np.float32).mean(0)
    has_te = np.linalg.norm(TE, axis=1) > 0
    TE = normalize(TE)
    tx = defaultdict(list)
    for d in alld:
        if d[1] in apos:
            tx[apos[d[1]]].append(d[2])
    for i, v in tx.items():
        txt[i] = " ".join(v)[:20000]
    IE = np.zeros((n, I.shape[1]), np.float32)
    for p, i in apos.items():
        q = ipos.get(p) if p in ipos else ipos.get(p.removeprefix("PMC"))
        if q is not None:
            IE[i] = np.asarray(I[q], np.float32)
    has_img = np.linalg.norm(IE, axis=1) > 0
    IE = normalize(IE)
    print("tf-idf", flush=True)
    X = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.5, dtype=np.float32).fit_transform(txt)

    # concept layer
    voc = {}; rows, cols = [], []
    for i, p in enumerate(arts):
        for u in mesh[p]:
            rows.append(i); cols.append(voc.setdefault(u, len(voc)))
    M = sparse.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(n, len(voc)))
    C = sparse.triu(M @ M.T, k=1).tocoo()
    keep = C.data >= 2
    CE = np.c_[C.row[keep], C.col[keep]]
    print(len(CE), "concept edges", flush=True)
    perm = rng.permutation(len(CE)); n70, n80 = int(0.7 * len(CE)), int(0.8 * len(CE))
    obs, trp, tep = CE[perm[:n70]], CE[perm[n70:n80]], CE[perm[n80:]]
    Aobs = adj(n, obs)
    allC = set(map(tuple, CE))

    # duplicate layers at article level
    dup_txt, dup_img = set(), set()
    doc_art = {}
    for d in alld:
        if d[1] in apos:
            doc_art[d[0]] = apos[d[1]]
    for l in gzip.open(data / "links/near_dup_pairs.jsonl.gz", "rt"):
        p = json.loads(l)
        x, y = doc_art.get(p["a"]), doc_art.get(p["b"])
        if x is not None and y is not None and x != y:
            dup_txt.add((min(x, y), max(x, y)))
    for l in open(a.pairs):
        p = json.loads(l)
        x, y = apos.get(norm(p["image_a"].split("_")[0])), apos.get(norm(p["image_b"].split("_")[0]))
        if x is not None and y is not None and x != y:
            dup_img.add((min(x, y), max(x, y)))
    print(len(dup_txt), "text-dup article pairs,", len(dup_img), "image-dup article pairs", flush=True)
    Atxt, Aimg = adj(n, dup_txt), adj(n, dup_img)
    Aflat = ((Aobs + Atxt + Aimg) > 0).astype(np.float32).tocsr()
    Adup = ((Atxt + Aimg) > 0).astype(np.float32).tocsr()

    def negatives(m):
        out = set()
        while len(out) < m // 2:  # random
            x, y = rng.integers(0, n, 2)
            if x != y and (min(x, y), max(x, y)) not in allC:
                out.add((min(x, y), max(x, y)))
        hard = set(); ip, ix = Aobs.indptr, Aobs.indices
        while len(hard) < m - m // 2:  # two hops: x - k - y in the observed concept layer, x and y not linked
            e = obs[rng.integers(0, len(obs))]
            x, k = (int(e[0]), int(e[1])) if rng.random() < 0.5 else (int(e[1]), int(e[0]))
            nb = ix[ip[k]:ip[k + 1]]
            y = int(nb[rng.integers(0, len(nb))])
            pr = (min(x, y), max(x, y))
            if x != y and pr not in allC and pr not in out:
                hard.add(pr)
        return np.array(sorted(out)), np.array(sorted(hard))

    def build(pos):
        rnd, hard = negatives(len(pos))
        P = np.vstack([pos, rnd, hard]).astype(np.int64)
        y = np.r_[np.ones(len(pos)), np.zeros(len(rnd) + len(hard))]
        is_hard = np.r_[np.zeros(len(pos) + len(rnd), bool), np.ones(len(hard), bool)]
        key = lambda S: np.array([(min(x, y_), max(x, y_)) in S for x, y_ in P], np.float32)
        tf = np.asarray(X[P[:, 0]].multiply(X[P[:, 1]]).sum(1)).ravel()
        img_ok = (has_img[P[:, 0]] & has_img[P[:, 1]]).astype(np.float32)
        cross = np.asarray((Adup[P[:, 0]].multiply(Aobs[P[:, 1]])).sum(1)).ravel() + \
                np.asarray((Adup[P[:, 1]].multiply(Aobs[P[:, 0]])).sum(1)).ravel()
        F = {"concept": struct(Aobs, P),
             "text": np.c_[tf, cos(TE, P) * (has_te[P[:, 0]] & has_te[P[:, 1]]), key(dup_txt)],
             "image": np.c_[cos(IE, P) * img_ok, img_ok, key(dup_img)],
             "cross": cross[:, None],
             "flattened": struct(Aflat, P)}
        return F, y, is_hard

    print("features", flush=True)
    Ftr, ytr, _ = build(trp)
    Fte, yte, hte = build(tep)
    hard_mask = (yte == 1) | hte  # positives vs hard negatives only
    combos = {"concept only": ["concept"], "text only": ["text"], "image only": ["image"],
              "flattened (one merged graph)": ["flattened"], "text + image": ["text", "image"],
              "multi-layer (all layers, kept separate)": ["concept", "text", "image", "cross"],
              "multi-layer minus text": ["concept", "image", "cross"], "multi-layer minus image": ["concept", "text", "cross"],
              "multi-layer minus cross-layer paths": ["concept", "text", "image"], "multi-layer minus topic": ["text", "image", "cross"]}
    res = {"articles": n, "concept_edges": int(len(CE)), "split": {"observed": int(len(obs)), "train_pos": int(len(trp)), "test_pos": int(len(tep))},
           "dup_text_pairs": len(dup_txt), "dup_image_pairs": len(dup_img), "models": {}}
    for name, parts in combos.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(np.hstack([Ftr[p] for p in parts]), ytr)
        s = clf.predict_proba(np.hstack([Fte[p] for p in parts]))[:, 1]
        res["models"][name] = {"all_negatives": boot(yte, s), "hard_negatives": boot(yte, s, hard_mask)}
        print(name, res["models"][name], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

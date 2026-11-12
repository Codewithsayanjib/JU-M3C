"""Multimodal leakage audit on the fetched PMC figures.

L1  Cross-modal exposure: share of native test documents whose article also ships its
    images and captions in MultiCaRe (a public image-text training resource).
L2  Caption answer leakage: share of diagnosis items whose gold diagnosis is written in a
    figure caption of the same article (compared with the case text given to the model).
L3  Conventional vs article-level image splits: with a random 80/10/10 split over figures,
    share of test figures that have a sibling figure from the same article in train.
L4  Duplicate images across articles: candidates from a 64-bit dHash (Hamming <= 4), then
    confirmed by Pearson correlation >= 0.95 of 32x32 grayscale thumbnails (dHash alone
    flags many look-alike layouts); counts pairs whose articles fall in different
    article clusters / splits.

Runs on whatever figures are downloaded so far; rerun after the fetch finishes.

Usage:
  python scripts/mm/leakage_mm.py --data $JUMC_DATA \
      --mm $JUMC_MV/full --out reports/mm_leakage.json
"""
import argparse, collections, gzip, io, json, random, re, tarfile
from itertools import combinations
from pathlib import Path
from PIL import Image


def jl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


STOP = set("with without of and the due to in on a an or for from by type other unspecified".split())


def mentions(dx, text):
    d, t = norm(dx), norm(text)
    if not d:
        return False, False
    exact = f" {d} " in f" {t} "
    toks = [w for w in d.split() if len(w) > 3 and w not in STOP]
    tset = set(t.split())
    tokens = bool(toks) and all(w in tset for w in toks)
    return exact, tokens


def dhash(data):
    im = Image.open(io.BytesIO(data)).convert("L")
    px = list(im.resize((9, 8)).tobytes())
    h = sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])
    return h, list(im.resize((32, 32)).tobytes())


def corr(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5; sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx and sy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JU-M3C release folder")
    ap.add_argument("--mm", required=True, help="fetch_figures output folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    data, mm = Path(a.data).expanduser(), Path(a.mm)
    rep = {}

    docs = list(jl(data / "links/doc_index.jsonl.gz"))
    split = {r["doc_id"]: r["split"] for r in jl(data / "links/splits.jsonl.gz")}
    cl_pmc = collections.defaultdict(set)
    cl_src = collections.defaultdict(set)
    for d in docs:
        cl_src[d["article_cluster"]].add(d["source"])
        if d["pmcid"]:
            cl_pmc[d["article_cluster"]].add(d["pmcid"])
    pmc_cl = {p: c for c, ps in cl_pmc.items() for p in ps}
    cl_split = {}
    for d in docs:
        cl_split[d["article_cluster"]] = split.get(d["doc_id"])

    # L1
    l1 = {}
    for d in docs:
        if d["native_split"] and d["native_split"].startswith("test") and d["source"] != "multicare":
            k = d["source"]
            s = l1.setdefault(k, {"n_test": 0, "pmc_article": 0, "in_multicare": 0})
            s["n_test"] += 1
            s["pmc_article"] += bool(cl_pmc[d["article_cluster"]])
            s["in_multicare"] += "multicare" in cl_src[d["article_cluster"]]
    for s in l1.values():
        s["pct_in_multicare"] = round(100 * s["in_multicare"] / s["n_test"], 1)
    rep["L1_cross_modal_exposure"] = l1

    # figures fetched so far
    figs = [f for f in jl(mm / "figures.jsonl")]
    by_pmc = collections.defaultdict(list)
    for f in figs:
        by_pmc[f["pmcid"]].append(f)
    done_cl = {pmc_cl[p] for p in by_pmc if p in pmc_cl}
    cl_caps = collections.defaultdict(list)
    for p, fs in by_pmc.items():
        if p in pmc_cl:
            cl_caps[pmc_cl[p]] += [f["caption"] for f in fs if f["caption"]]
    rep["fetched"] = {"articles": len(by_pmc), "figures": len(figs),
                      "images": sum(bool(f["image_path"]) for f in figs)}

    # L2
    l2 = {}
    for task, files in [("diagnosis_open", ["train", "val", "test"]), ("diagnosis_mcq", ["test"])]:
        for sp in files:
            for r in jl(data / f"tasks/{task}/{sp}.jsonl.gz"):
                c = int(r["article_cluster"])
                if c not in done_cl or not cl_caps[c]:
                    continue
                dx = r.get("answer_text") or r["target"]
                cap = " ".join(cl_caps[c])
                ce, ct = mentions(dx, cap)
                ie, it = mentions(dx, r["input"])
                s = l2.setdefault(f"{task}:{r['source']}", collections.Counter())
                s["n_with_captions"] += 1
                s["caption_exact"] += ce; s["caption_all_tokens"] += ct
                s["input_exact"] += ie; s["input_all_tokens"] += it
    for s in l2.values():
        n = s["n_with_captions"]
        for k in list(s):
            if k != "n_with_captions":
                s[f"pct_{k}"] = round(100 * s[k] / n, 1)
    rep["L2_caption_answer_leakage"] = {k: dict(v) for k, v in l2.items()}

    # L3
    imgs = [f for f in figs if f["image_path"]]
    rnd = random.Random(a.seed)
    lab = {id(f): ("train" if (x := rnd.random()) < 0.8 else "val" if x < 0.9 else "test") for f in imgs}
    train_art = {f["pmcid"] for f in imgs if lab[id(f)] == "train"}
    test = [f for f in imgs if lab[id(f)] == "test"]
    rep["L3_random_image_split"] = {
        "test_images": len(test),
        "pct_test_with_same_article_image_in_train": round(100 * sum(f["pmcid"] in train_art for f in test) / max(len(test), 1), 1),
        "pct_test_from_multi_figure_articles": round(100 * sum(len(by_pmc[f["pmcid"]]) > 1 for f in test) / max(len(test), 1), 1),
        "article_level_split_crossing": 0,
    }

    # L4
    hashes, thumbs = {}, {}
    for t in sorted((mm / "shards").glob("*.tar")):
        try:
            with tarfile.open(t) as tf:
                for m in tf:
                    if m.isfile() and not m.name.startswith("._"):
                        try:
                            hashes[m.name], thumbs[m.name] = dhash(tf.extractfile(m).read())
                        except Exception:
                            pass
        except tarfile.ReadError:
            pass  # shard still being written
    name_pmc = {f["image_path"]: f["pmcid"] for f in imgs}
    exact = collections.defaultdict(list)
    for n, h in hashes.items():
        exact[h].append(n)
    bands = collections.defaultdict(list)
    for n, h in hashes.items():
        for b in range(4):
            bands[(b, (h >> (16 * b)) & 0xFFFF)].append(n)
    cand = set()
    for ns in bands.values():
        if len(ns) > 200:
            continue  # blank / generic images
        for x, y in combinations(ns, 2):
            if bin(hashes[x] ^ hashes[y]).count("1") <= 4 and name_pmc.get(x) != name_pmc.get(y):
                cand.add(tuple(sorted((x, y))))
    pairs = {pr for pr in cand if corr(thumbs[pr[0]], thumbs[pr[1]]) >= 0.95}
    cross_cl = cross_split = 0
    for x, y in pairs:
        cx, cy = pmc_cl.get(name_pmc.get(x)), pmc_cl.get(name_pmc.get(y))
        if cx != cy:
            cross_cl += 1
            cross_split += cl_split.get(cx) != cl_split.get(cy)
    rep["L4_duplicate_images"] = {
        "images_hashed": len(hashes),
        "exact_groups_gt1": sum(len(v) > 1 for v in exact.values()),
        "dhash_candidate_pairs": len(cand),
        "near_dup_pairs_cross_article": len(pairs),
        "pairs_cross_cluster": cross_cl,
        "pairs_cross_split": cross_split,
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(a.out).with_suffix(".pairs.jsonl"), "w") as fp:
        for x, y in sorted(pairs):
            cx, cy = pmc_cl.get(name_pmc.get(x)), pmc_cl.get(name_pmc.get(y))
            fp.write(json.dumps({"image_a": x, "image_b": y, "cluster_a": cx, "cluster_b": cy,
                                 "split_a": cl_split.get(cx), "split_b": cl_split.get(cy),
                                 "corr": round(corr(thumbs[x], thumbs[y]), 4)}) + "\n")
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

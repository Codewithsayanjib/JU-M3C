"""Build the multimodal benchmark items from the fetched PMC figures.

MM-1  Image-case matching. Case text (PMC-Patients summary, figure references removed)
      + 4 images (A-D): the correct one is a figure from the same article, the three
      distractors are figures from other articles in the same split with the same
      caption-derived modality (gold figures without a recognised modality are not used). Metric: accuracy.
MM-2  Does the image help? CUPCase 4-option diagnosis MCQs whose article has figures,
      asked as text-only, image-only and text+image. Captions are never shown
      (they state the diagnosis in about 1 in 5 cases, see leakage_mm.py L2).

Splits follow the article-level splits: MM-1 dev = val clusters, test = test clusters.
MM-2 dev = the diagnosis_mcq dev file; test = all other CUPCase MCQs (the full native
test set minus dev) whose article has figures.
Selected images are copied out of the tar shards into items/images/.

Usage:
  python scripts/mm/build_mm_items.py --data $JUMC_DATA \
      --mm $JUMC_MV/full --n-dev 200 --n-test 1000
"""
import argparse, collections, gzip, json, random, re, tarfile
from pathlib import Path

MODALITY_RULES = [  # first match wins; order matters
    ("histology", r"\b(h&e|hematoxylin|haematoxylin|histolog|immunohisto|biopsy specimen|stain(ing)?|magnification|cytolog|smear)\b"),
    ("ct", r"\b(ct|computed tomograph|cta|pet[- /]ct)\b"),
    ("mri", r"\b(mri|mr imag|magnetic resonance|t1[- ]weighted|t2[- ]weighted|flair|dwi)\b"),
    ("xray", r"\b(x-?ray|radiograph|chest film|plain film)\b"),
    ("ultrasound", r"\b(ultrasound|ultrasonograph|sonograph|echocardiogra|doppler)\b"),
    ("endoscopy", r"\b(endoscop|colonoscop|gastroscop|bronchoscop|laryngoscop|cystoscop)\b"),
    ("ecg", r"\b(ecg|ekg|electrocardiogra)\b"),
    ("fundus_oct", r"\b(fundus|fundoscop|optical coherence|oct)\b"),
    ("angiography", r"\b(angiogra|venogra)\b"),
    ("photo", r"\b(photograph|clinical (photo|image|picture)|lesion on|rash|skin|dermoscop|intraoperative)\b"),
]
FIG_REF = re.compile(r"\s*[\(\[]\s*(fig(ure)?s?\.?|table)\s*[^\)\]]{0,30}[\)\]]", re.I)


def modality(caption):
    c = caption.lower()
    for name, rx in MODALITY_RULES:
        if re.search(rx, c):
            return name
    return "other"


def jl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mm", required=True)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-words", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fetch-complete", action="store_true", help="fetch finished: use every shard")
    a = ap.parse_args()
    data, mm = Path(a.data).expanduser(), Path(a.mm)
    out = mm / "items"; (out / "images").mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)

    docs = list(jl(data / "links/doc_index.jsonl.gz"))
    split = {r["doc_id"]: r["split"] for r in jl(data / "links/splits.jsonl.gz")}
    pmc_cl = {d["pmcid"]: d["article_cluster"] for d in docs if d["pmcid"]}
    cl_split = {d["article_cluster"]: split.get(d["doc_id"]) for d in docs}

    retracted = {r["pmcid"] for r in jl(mm / "articles.jsonl") if r.get("is_retracted")}
    year = {r["pmcid"]: r["year"] for r in jl(mm / "pubyear.jsonl")} if (mm / "pubyear.jsonl").exists() else {}

    # figures in closed shards only, retracted articles excluded
    shards = sorted((mm / "shards").glob("*.tar"))
    open_shard = None if a.fetch_complete else (shards[-1].name if shards else None)
    figs = collections.defaultdict(list)
    for f in jl(mm / "figures.jsonl"):
        if f.get("image_path") and f.get("shard") != open_shard and f["pmcid"] not in retracted:
            f["modality"] = modality(f["caption"])
            figs[f["pmcid"]].append(f)
    print(f"{len(figs)} articles with usable figures", flush=True)

    need = {}  # image_path -> shard

    def use(f):
        need[f["image_path"]] = f["shard"]
        return f["image_path"]

    # MM-1
    texts = {}
    for r in jl(data / "corpus/records.jsonl.gz"):
        if r["source"] == "pmc_patients" and r["pmcid"] in figs and r["pmcid"] not in texts:
            texts[r["pmcid"]] = r["text"]
    pool = collections.defaultdict(list)  # (split, modality) -> figures
    for p, fs in figs.items():
        for f in fs:
            pool[(cl_split.get(pmc_cl.get(p)), f["modality"])].append(f)
    for sp, n, name in [("val", a.n_dev, "dev"), ("test", a.n_test, "test")]:
        cands = sorted(p for p in texts if cl_split.get(pmc_cl.get(p)) == sp)
        rnd.shuffle(cands)
        items = []
        for p in cands:
            typed = [f for f in figs[p] if f["modality"] != "other"]
            if not typed:
                continue  # distractors must share a known modality with the gold image
            gold = rnd.choice(typed)
            others = [f for f in pool[(sp, gold["modality"])] if f["pmcid"] != p]
            if len(others) < 3:
                continue
            opts = [gold] + rnd.sample(others, 3)
            if len({o["pmcid"] for o in opts}) < 4:
                continue
            rnd.shuffle(opts)
            words = FIG_REF.sub("", texts[p]).split()
            items.append({
                "item_id": f"mm1:{p}", "task": "mm1_image_case_matching", "split": name,
                "pmcid": p, "article_cluster": pmc_cl[p], "modality": gold["modality"], "year": year.get(p),
                "case_text": " ".join(words[: a.max_words]),
                "images": [use(o) for o in opts], "option_pmcids": [o["pmcid"] for o in opts],
                "answer": "ABCD"[opts.index(gold)], "licences": [o["licence"] for o in opts],
            })
            if len(items) == n:
                break
        with open(out / f"mm1_{name}.jsonl", "w") as fo:
            for it in items:
                fo.write(json.dumps(it) + "\n")
        print(f"MM-1 {name}: {len(items)} items", flush=True)

    # MM-2
    cl_pmcs = collections.defaultdict(set)
    for p, c in pmc_cl.items():
        cl_pmcs[c].add(p)
    dev_ids = {r["doc_id"] for r in jl(data / "tasks/diagnosis_mcq/dev.jsonl.gz")}
    for src, name in [("dev", "dev"), ("test", "test")]:
        items = []
        for r in jl(data / f"tasks/diagnosis_mcq/{src}.jsonl.gz"):
            if name == "test" and r["doc_id"] in dev_ids:
                continue
            fs = [f for p in cl_pmcs.get(int(r["article_cluster"]), ()) for f in figs.get(p, [])]
            if not fs:
                continue
            fs = sorted(fs, key=lambda f: (f["pmcid"], f["fig_index"]))[:3]  # first figures, max 3
            items.append({
                "item_id": f"mm2:{r['doc_id']}", "task": "mm2_text_image_diagnosis", "split": name,
                "doc_id": r["doc_id"], "article_cluster": r["article_cluster"],
                "case_text": r["input"], "options": r["options"], "answer": r["target"],
                "answer_text": r["answer_text"], "images": [use(f) for f in fs],
                "year": year.get(fs[0]["pmcid"]),
                "modalities": [f["modality"] for f in fs], "licences": [f["licence"] for f in fs],
            })
        with open(out / f"mm2_{name}.jsonl", "w") as fo:
            for it in items:
                fo.write(json.dumps(it) + "\n")
        print(f"MM-2 {name}: {len(items)} items (of the {src} MCQs)", flush=True)

    # copy images out of shards
    by_shard = collections.defaultdict(set)
    for img, sh in need.items():
        by_shard[sh].add(img)
    for sh, names in by_shard.items():
        with tarfile.open(mm / "shards" / sh) as tf:
            for m in tf:
                if m.name in names:
                    (out / "images" / m.name).write_bytes(tf.extractfile(m).read())
    print(f"{len(need)} images copied to {out / 'images'}", flush=True)


if __name__ == "__main__":
    main()

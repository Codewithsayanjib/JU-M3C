"""Leakage inflation (2a): how much does a source's native split overstate scores?

A model-free "copy the nearest training case" baseline. For every native-test example we find the most
similar training document (TF-IDF cosine over the corpus text) and copy its answer. The SAME test
examples are scored twice; only the retrieval pool changes:
  native  - documents whose source split is train (sources without a split count as train, as in
            network_links.py), i.e. what a naive merge of the public datasets would train on;
  ours    - documents in our article-cluster train split.
Native-test documents are all in our test split, so the difference is purely the leak.

Prediction per task (neighbour = best pool document carrying the field used):
  diagnosis_open    neighbour's diagnosis, else its article title       -> hit if gold diagnosis tokens
                                                                          are >= 80% recalled
  title_generation  neighbour's title                                  -> ROUGE-L
  diagnosis_mcq     option closest (TF-IDF) to neighbour's title+dx+text -> accuracy
  notes_specialty / notes_doctype / notes_section  neighbour's label     -> accuracy
  notes_summary / dialogue2note                    neighbour's target    -> ROUGE-L
Summarization is skipped: MultiClinSum's native-test summaries are not public.
Paired bootstrap 95% CI for (native - ours); exact McNemar for the accuracy-type tasks.

  python scripts/experiments/leakage_inflation_nn.py --data $JUMC_DATA --out reports/leakage_inflation.json
"""
import argparse, gzip, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ju_m3c.benchmark.metrics import rouge_l_per_note  # noqa: E402

TASKS = ["diagnosis_open", "title_generation", "diagnosis_mcq", "notes_specialty", "notes_doctype",
         "notes_section", "notes_summary", "dialogue2note"]
ACC_TASKS = {"diagnosis_open", "diagnosis_mcq", "notes_specialty", "notes_doctype", "notes_section"}
TOK = re.compile(r"[a-z0-9]+")


def native(ns):
    n = (ns or "").lower()
    return "test" if n.startswith("test") else "val" if n.startswith(("val", "dev", "valid")) else "train"


def toks(s):
    return TOK.findall((s or "").lower())


def dx_hit(pred, gold):
    g, p = set(toks(gold)), set(toks(pred))
    return float(bool(g) and len(g & p) / len(g) >= 0.8)


def boot_diff(a, b, n=2000, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [round(100 * float(np.percentile(d, 2.5)), 1), round(100 * float(np.percentile(d, 97.5)), 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--examples", default=None, help="optional jsonl of copied answers for hand-checking")
    ap.add_argument("--per-item", default=None, help="optional jsonl of per-example scores under both pools")
    a = ap.parse_args()
    data = Path(a.data).expanduser()

    print("loading corpus", flush=True)
    docs, text = [], []
    for l in gzip.open(data / "corpus/records.jsonl.gz", "rt"):
        r = json.loads(l)
        docs.append({"doc_id": r["doc_id"], "source": r["source"], "ours": r["split"], "native": native(r["native_split"])})
        text.append(r["text"] or "")
    di = {d["doc_id"]: i for i, d in enumerate(docs)}

    # per-document fields from every task file (targets are properties of the document)
    field = defaultdict(dict); examples = defaultdict(list)
    for t in TASKS:
        for s in ["train", "val", "dev", "test"]:
            for l in gzip.open(data / f"tasks/{t}/{s}.jsonl.gz", "rt"):
                e = json.loads(l)
                if e["doc_id"] not in di:
                    continue
                field[t][di[e["doc_id"]]] = e["target"] if t != "diagnosis_mcq" else e.get("answer_text")
                examples[t].append(e)

    print(f"tf-idf over {len(docs)} documents", flush=True)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.5, stop_words="english",
                          max_features=500_000, dtype=np.float32)
    X = vec.fit_transform(text)

    res, ex_out, item_out = {}, [], []
    for t in TASKS:
        test = [e for e in examples[t] if docs[di[e["doc_id"]]]["native"] == "test"]
        test = list({e["example_id"]: e for e in test}.values())
        if not test:
            continue
        q = np.array([di[e["doc_id"]] for e in test])
        if t == "diagnosis_open":
            src_fields = [field["diagnosis_open"], field["title_generation"]]
        elif t == "diagnosis_mcq":
            src_fields = [field["diagnosis_open"], field["title_generation"], field["summarization"] if "summarization" in field else {}]
        else:
            src_fields = [field[t]]
        cand = sorted({i for f in src_fields for i in f}) if t != "diagnosis_mcq" else list(range(len(docs)))
        out = {"n": len(test)}
        scores, sims, preds = {}, {}, {}
        for scheme in ["native", "ours"]:
            pool = np.array([i for i in cand if docs[i][scheme] == "train"])
            P = X[pool].T.tocsc()
            best, bsim = np.empty(len(q), int), np.empty(len(q))
            for s0 in range(0, len(q), 256):
                S = (X[q[s0:s0 + 256]] @ P).toarray()
                j = S.argmax(1); best[s0:s0 + 256] = pool[j]; bsim[s0:s0 + 256] = S[np.arange(len(j)), j]
            pr = []
            for e, b in zip(test, best):
                if t == "diagnosis_mcq":
                    ctx = " ".join(filter(None, [field["diagnosis_open"].get(b), field["title_generation"].get(b), text[b][:3000]]))
                    opts = e["options"] if isinstance(e["options"], list) else json.loads(e["options"].replace("'", '"'))
                    O = vec.transform(opts); c = vec.transform([ctx])
                    pr.append("ABCDEFGH"[int((O @ c.T).toarray().ravel().argmax())])
                else:
                    pr.append(next((f[b] for f in src_fields if b in f), ""))
            golds = [e["target"] for e in test]
            if t == "diagnosis_open":
                sc = [dx_hit(p, g) for p, g in zip(pr, golds)]
            elif t in ACC_TASKS:
                sc = [float(p.strip().lower() == g.strip().lower()) for p, g in zip(pr, golds)]
            else:
                sc = rouge_l_per_note(pr, golds)
            scores[scheme], sims[scheme], preds[scheme] = sc, bsim, (pr, best)
            out[scheme] = {"score": round(100 * float(np.mean(sc)), 1), "median_nn_cosine": round(float(np.median(bsim)), 3),
                           "pct_nn_cosine_ge_0.9": round(100 * float(np.mean(bsim >= 0.9)), 1),
                           "pool": int(len(pool))}
            print(t, scheme, out[scheme], flush=True)
        out["metric"] = "accuracy" if t in ACC_TASKS else "rougeL"
        out["inflation_points"] = round(out["native"]["score"] - out["ours"]["score"], 1)
        out["inflation_ci95"] = boot_diff(scores["native"], scores["ours"])
        if t in ACC_TASKS:
            b = sum(1 for x, y in zip(scores["native"], scores["ours"]) if x > y)
            c = sum(1 for x, y in zip(scores["native"], scores["ours"]) if y > x)
            out["mcnemar"] = {"native_only": b, "ours_only": c, "p": float(binomtest(b, b + c).pvalue) if b + c else 1.0}
        res[t] = out
        for k, e in enumerate(test):
            item_out.append({"task": t, "example_id": e["example_id"], "doc_id": e["doc_id"],
                             "native": scores["native"][k], "ours": scores["ours"][k]})
        if a.examples:
            pr_n, best_n = preds["native"]
            for k, e in enumerate(test):
                if sims["native"][k] >= 0.5:
                    ex_out.append({"task": t, "doc_id": e["doc_id"], "gold": e["target"], "copied": pr_n[k],
                                   "neighbour": docs[best_n[k]]["doc_id"], "cosine": round(float(sims["native"][k]), 3),
                                   "score": scores["native"][k], "query_head": text[q[k]][:300], "neighbour_head": text[best_n[k]][:300]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.examples:
        with open(a.examples, "w") as f:
            for r in ex_out:
                f.write(json.dumps(r) + "\n")
    if a.per_item:
        with open(a.per_item, "w") as f:
            for r in item_out:
                f.write(json.dumps(r) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

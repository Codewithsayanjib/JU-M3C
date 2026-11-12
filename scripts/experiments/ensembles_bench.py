"""Ways of combining the twelve models without running them again (from the saved test outputs).

  python scripts/experiments/ensembles_bench.py

Answer tasks (multiple-choice diagnosis, note specialty, note type, note section):
  plurality vote        all 12 models, one vote each (ties: best development model)
  weighted vote         each vote weighted by the model's development score on the task
  Dawid-Skene           per-model, per-label reliability estimated by EM on the test answers
                        themselves (no labels used), initialised from the weighted vote
  best with override    the best development model's answer, unless the next three models by
                        development score all agree on another answer
  agreement abstention  answer only when at least k models give the plurality answer; accuracy and
                        coverage for k = 1..12
Text tasks (case summary, note summary, title, dialogue to note, open-ended diagnosis):
  consensus output      minimum Bayes risk: per example, the output of the 12 that agrees most with
                        the other 11 (ROUGE-L; token F1 for open-ended diagnosis); the reference is not used
Model choice and weights come from the development runs, so test results never decide them.
Baseline: best single model on the test set. Paired bootstrap 95% intervals, 1,000 resamples, seed 42.
Writes results/bench/sealed_test_eval/analysis/ensembles.csv, abstention.csv and docs/bench_ensembles.md.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "benchmark")); sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
os.environ.setdefault("CORPUS_ROOT", "corpus")

from analyze_bench import md as table  # noqa: E402
from score_bench import get_refs, token_f1  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from ju_m3c.benchmark.bench_parse import parse  # noqa: E402

RES = ROOT / "results" / "bench" / "sealed_test_eval"
ANSWER = ["diagnosis_mcq", "notes_specialty", "notes_doctype", "notes_section"]
TEXT = ["summarization", "notes_summary", "title_generation", "dialogue2note", "diagnosis_open"]
SEED, N_BOOT = 42, 1000
_rouge = None


def rouge(a, b):
    global _rouge
    if _rouge is None:
        from rouge_score import rouge_scorer
        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge.score(b, a)["rougeL"].fmeasure


def load(task, models):
    refs = get_refs(task, "test_eval"); ids = sorted(refs)
    out = {}
    for m in models:
        rows = {r["example_id"]: r for r in map(json.loads, open(RES / m / f"{task}.jsonl"))}
        ps = [parse(task, rows[i]["raw_output"], refs[i]) for i in ids]
        out[m] = [(p["parsed"] if p["lenient"] else None) if task in ANSWER else (p["parsed"] or "") for p in ps]
    return refs, ids, out


def dawid_skene(P, labels, init, iters=50):
    """P: list over examples of dict model->answer (None allowed). Returns posterior argmax per example."""
    L = {l: k for k, l in enumerate(labels)}; K = len(labels); models = list(P[0])
    T = np.full((len(P), K), 1e-3)
    for i, a in enumerate(init):
        if a in L:
            T[i, L[a]] = 1.0
    T /= T.sum(1, keepdims=True)
    for _ in range(iters):
        prior = T.mean(0) + 1e-6
        conf = {m: np.ones((K, K)) * 0.5 for m in models}  # smoothing
        for i, row in enumerate(P):
            for m, a in row.items():
                if a in L:
                    conf[m][:, L[a]] += T[i]
        for m in models:
            conf[m] /= conf[m].sum(1, keepdims=True)
        logT = np.tile(np.log(prior), (len(P), 1))
        for i, row in enumerate(P):
            for m, a in row.items():
                if a in L:
                    logT[i] += np.log(conf[m][:, L[a]])
        T = np.exp(logT - logT.max(1, keepdims=True)); T /= T.sum(1, keepdims=True)
    return [labels[k] for k in T.argmax(1)]


def main():
    dev = pd.read_csv(ROOT / "results" / "bench" / "dev" / "summary.csv")
    test = pd.read_csv(RES / "summary.csv")
    models = sorted(p.name for p in RES.iterdir() if p.is_dir() and p.name != "analysis")
    rng = np.random.default_rng(SEED)
    rows, abst = [], []
    for t in ANSWER + TEXT:
        refs, ids, pred = load(t, models)
        n = len(ids); boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]
        dscore = dev[(dev.task == t) & (dev.main == True)].set_index("model").value.reindex(models)
        rank = dscore.sort_values(ascending=False).index.tolist()
        best = test[(test.task == t) & (test.main == True)].set_index("model").value.idxmax()
        y = [refs[i]["target"] for i in ids]
        if t in ANSWER:
            labels = sorted(set(y) | {a for m in models for a in pred[m] if a is not None})
            yarr = np.array(y, dtype=object)

            def score(p, idx):
                p = np.array(["<none>" if a is None else a for a in p], dtype=object)
                if t == "diagnosis_mcq":
                    return float(np.mean(p[idx] == yarr[idx]))
                return f1_score(yarr[idx], p[idx], labels=sorted(set(y)), average="macro", zero_division=0)

            def vote(weights):
                out = []
                for k in range(n):
                    c = defaultdict(float)
                    for m in rank:
                        if pred[m][k] is not None:
                            c[pred[m][k]] += weights[m]
                    out.append(max(c, key=lambda a: (c[a], -next(j for j, m in enumerate(rank) if pred[m][k] == a))) if c else None)
                return out

            plural = vote({m: 1.0 for m in models})
            weighted = vote({m: float(dscore[m]) for m in models})
            ds = dawid_skene([{m: pred[m][k] for m in models} for k in range(n)], labels, weighted)
            override = []
            for k in range(n):
                others = [pred[m][k] for m in rank[1:4]]
                override.append(others[0] if others[0] is not None and len(set(others)) == 1 and others[0] != pred[rank[0]][k]
                                else pred[rank[0]][k])
            methods = {"best single (test)": pred[best], "plurality vote": plural, "weighted vote": weighted,
                       "Dawid-Skene (no labels)": ds, "best dev model with override": override}
            base = pred[best]
            for name, p in methods.items():
                s = score(p, np.arange(n))
                if name.startswith("best single"):
                    rows.append({"task": t, "method": f"best single ({best})", "score": s}); continue
                d = [score(p, b) - score(base, b) for b in boots]
                rows.append({"task": t, "method": name, "score": s, "diff_vs_best": s - score(base, np.arange(n)),
                             "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5)})
            # agreement-based abstention (accuracy on answered questions)
            for kmin in range(1, len(models) + 1):
                ok, cov = [], 0
                for k in range(n):
                    c = Counter(pred[m][k] for m in models if pred[m][k] is not None)
                    if c and c.most_common(1)[0][1] >= kmin:
                        cov += 1; ok.append(c.most_common(1)[0][0] == y[k])
                abst.append({"task": t, "min_agree": kmin, "coverage": cov / n, "accuracy": float(np.mean(ok)) if ok else np.nan})
            abst.append({"task": t, "min_agree": 0, "coverage": 1.0,
                         "accuracy": float(np.mean(np.array(pred[best], dtype=object) == yarr)), "note": f"best single ({best})"})
        else:
            sim = token_f1 if t == "diagnosis_open" else rouge
            metric = token_f1 if t == "diagnosis_open" else rouge

            def per_ex(p):
                return np.array([metric(a, r) for a, r in zip(p, y)])

            def mbr(members):
                out = []
                for k in range(n):
                    outs = [pred[m][k] for m in members]
                    S = np.zeros((len(outs), len(outs)))
                    for a in range(len(outs)):
                        for b in range(a + 1, len(outs)):
                            S[a, b] = S[b, a] = sim(outs[a], outs[b])
                    out.append(outs[int(S.sum(1).argmax())])
                return out

            base = per_ex(pred[best])
            rows.append({"task": t, "method": f"best single ({best})", "score": float(base.mean())})
            for name, members in (("consensus output, all 12", models), ("consensus output, top 5 (dev)", rank[:5])):
                s = per_ex(mbr(members))
                d = [s[b].mean() - base[b].mean() for b in boots]
                rows.append({"task": t, "method": name, "score": float(s.mean()), "diff_vs_best": float(s.mean() - base.mean()),
                             "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5)})
            orc = np.max(np.vstack([per_ex(pred[m]) for m in models]), axis=0)
            rows.append({"task": t, "method": "oracle, all 12", "score": float(orc.mean())})
        print("done", t, flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "analysis" / "ensembles.csv", index=False)
    ab = pd.DataFrame(abst); ab.to_csv(RES / "analysis" / "abstention.csv", index=False)
    md = ["# Combining models without new runs (test evaluation subsets)\n",
          "Main metric per task (accuracy, macro-F1, ROUGE-L or token F1). Differences against the best single test model "
          "with paired bootstrap 95% intervals. Weights and model choice from development runs.\n"]
    md.append(table(df.set_index(["task", "method"])))
    md.append("\n\n## Agreement-based abstention (accuracy on answered items)\n")
    md.append(table(ab[ab.min_agree > 0].pivot(index="min_agree", columns="task", values="coverage").add_prefix("coverage ").join(
        ab[ab.min_agree > 0].pivot(index="min_agree", columns="task", values="accuracy").add_prefix("accuracy "))))
    (ROOT / "docs" / "bench_ensembles.md").write_text("\n".join(md) + "\n")
    print("wrote", RES / "analysis" / "ensembles.csv")


if __name__ == "__main__":
    main()

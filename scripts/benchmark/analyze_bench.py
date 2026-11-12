"""Secondary analyses of the scored test outputs (no generation). Checklist items B2-B5 and C1-C6.

  python scripts/benchmark/analyze_bench.py --subset test_eval --confirm-test

Re-parses the saved outputs with the frozen parser, builds one per-example score table and writes
  results/bench/{subset dir}/analysis/*.csv   (all tables)
  docs/bench_analysis.md                      (report)
Per-example score = the task's main metric where it is defined per example (ROUGE-L, token F1,
correct/incorrect). For the three classification tasks the breakdowns use accuracy, because
macro-F1 is not defined per example; C1 reports per-class F1.
"""
import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_bench import CLS, CORPUS_ROOT, GEN, MAIN, get_refs, norm, token_f1  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from ju_m3c.benchmark.bench_parse import _PREFIX, cut_at_end_token, has_end_token, parse  # noqa: E402
from ju_m3c.benchmark.metrics import rouge_l_per_note  # noqa: E402
from ju_m3c.benchmark.parse import strip_thinking  # noqa: E402

SEED, N_BOOT = 42, 10_000
TASKS = ["diagnosis_mcq", "diagnosis_open", "notes_specialty", "notes_doctype", "notes_section",
         "summarization", "notes_summary", "title_generation", "dialogue2note"]
BINARY = ("diagnosis_mcq",) + CLS
ONE_LINE = ("title_generation", "diagnosis_open", "notes_summary") + BINARY


def md(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    d = df.reset_index() if index else df
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def novel_rate(text: str, source: str, n: int = 4) -> float:
    """Share of the text's word n-grams that do not occur in the source (1 = nothing copied)."""
    a = ngrams(norm(text).split(), n)
    return float("nan") if not a else len(a - ngrams(norm(source).split(), n)) / len(a)


def options_of(ex):
    o = ex["options"]
    return ast.literal_eval(o) if isinstance(o, str) else o


def build(root: Path, subset: str):
    clusters = defaultdict(set)
    for line in open(CORPUS_ROOT / "processed" / "doc_index.jsonl"):
        d = json.loads(line)
        clusters[d["article_cluster"]].add(d["source"])
    models = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "analysis")
    rows = []
    for task in TASKS:
        refs = get_refs(task, subset)
        ids = sorted(refs)
        for m in models:
            by_id = {r["example_id"]: r for r in map(json.loads, open(root / m / f"{task}.jsonl"))}
            ps = [parse(task, by_id[i]["raw_output"], refs[i]) for i in ids]
            if task in GEN:
                sc = rouge_l_per_note([p["parsed"] or "" for p in ps], [refs[i]["target"] for i in ids])
            elif task == "diagnosis_open":
                sc = [token_f1(p["parsed"] or "", refs[i]["target"]) for p, i in zip(ps, ids)]
            else:
                sc = [float((p["parsed"] if p["lenient"] else None) == refs[i]["target"]) for p, i in zip(ps, ids)]
            for i, p, s in zip(ids, ps, sc):
                e, r = refs[i], by_id[i]
                rows.append({"model": m, "task": task, "example_id": i, "source": e["source"],
                             "n_src": len(clusters[e["article_cluster"]]), "n_words_input": e["n_words_input"],
                             "target_kind": e.get("target_kind"), "score": s,
                             "pred": (p["parsed"] if p["lenient"] else "<unparsed>") if task in BINARY else (p["parsed"] or ""),
                             "strict": p["strict"], "lenient": p["lenient"],
                             "truncated": r["finish_reason"] == "length" and not has_end_token(r["raw_output"]),
                             "raw": r["raw_output"]})
    return pd.DataFrame(rows), models


def boot_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(N_BOOT)]
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev", choices=["dev", "test_eval"])
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    if a.subset == "test_eval" and not a.confirm_test:
        sys.exit("The test evaluation subset needs --confirm-test.")
    root = ROOT / "results" / "bench" / ("sealed_test_eval" if a.subset == "test_eval" else a.subset)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    df, models = build(root, a.subset)
    refs = {t: get_refs(t, a.subset) for t in TASKS}
    df.drop(columns=["raw", "pred"]).to_csv(out / "per_example.csv.gz", index=False)
    best = {t: df[df.task == t].groupby("model").score.mean().idxmax() for t in TASKS}
    rng = np.random.default_rng(SEED)
    R = [f"# Secondary analyses of the test results ({len(models)} models)\n",
         "Generated by `scripts/benchmark/analyze_bench.py` from the saved outputs; no model was run. Score = main "
         "metric per example (ROUGE-L, token F1, or correct/incorrect; accuracy for the classification tasks).\n"]

    # ---------- B2: overlapped vs non-overlapped articles
    R.append("\n## B2. Test items whose article occurs in several sources\n")
    R.append("`multi` = the example's article cluster contains documents from two or more sources. Mean over "
             "the 12 models; difference multi − single with a bootstrap 95% interval over examples.\n")
    b2 = []
    for t in TASKS:
        x = df[df.task == t].groupby(["example_id", "n_src"]).score.mean().reset_index()
        mu, si = x[x.n_src >= 2].score, x[x.n_src < 2].score
        row = {"task": t, "n_multi": len(mu), "n_single": len(si)}
        if min(len(mu), len(si)) >= 30:
            d, lo, hi = boot_diff(mu, si, rng)
            per_model = df[df.task == t].assign(multi=lambda z: z.n_src >= 2).groupby(["model", "multi"]).score.mean().unstack()
            row.update(score_multi=mu.mean(), score_single=si.mean(), diff=d, ci_lo=lo, ci_hi=hi,
                       models_higher_on_multi=int((per_model[True] > per_model[False]).sum()))
        b2.append(row)
    b2 = pd.DataFrame(b2).set_index("task")
    b2.to_csv(out / "b2_overlap.csv")
    R.append(md(b2))
    R.append("\n\nThe same comparison within one source, for the tasks that mix sources (overlap is far more "
             "common in some sources than in others, so the pooled difference can reflect the source):\n")
    b2s = []
    for t in ["summarization", "title_generation"]:
        x = df[df.task == t].groupby(["example_id", "source", "n_src"]).score.mean().reset_index()
        for src, z in x.groupby("source"):
            mu, si = z[z.n_src >= 2].score, z[z.n_src < 2].score
            row = {"task": t, "source": src, "n_multi": len(mu), "n_single": len(si)}
            if min(len(mu), len(si)) >= 30:
                d, lo, hi = boot_diff(mu, si, rng)
                row.update(score_multi=mu.mean(), score_single=si.mean(), diff=d, ci_lo=lo, ci_hi=hi)
            b2s.append(row)
    b2s = pd.DataFrame(b2s).set_index(["task", "source"])
    b2s.to_csv(out / "b2_overlap_within_source.csv")
    R.append(md(b2s))

    # ---------- B3: complementarity
    R.append("\n\n## B3. Complementarity between models\n")
    R.append("`best` = best single model of the task. `oracle_all` = score if, for every example, the best of "
             "the 12 outputs were chosen (an upper bound that no real router reaches; for ROUGE-L and token F1 it "
             "is also inflated by noise). `oracle_top3` = the same over the three best models of the task. "
             "`best_wrong_other_right` = share of the best model's errors that at least one other model answers "
             "correctly (correct/incorrect tasks only).\n")
    b3 = []
    for t in TASKS:
        w = df[df.task == t].pivot(index="example_id", columns="model", values="score")
        top3 = w.mean().sort_values(ascending=False).index[:3]
        row = {"task": t, "best": best[t], "best_score": w[best[t]].mean(), "oracle_top3": w[top3].max(axis=1).mean(),
               "oracle_all": w.max(axis=1).mean()}
        if t in BINARY:
            wrong = w[w[best[t]] == 0]
            row.update(best_wrong_other_right=(wrong.drop(columns=best[t]).max(axis=1) > 0).mean(),
                       all_models_right=(w.sum(axis=1) == len(models)).mean(), all_models_wrong=(w.sum(axis=1) == 0).mean())
        b3.append(row)
    b3 = pd.DataFrame(b3).set_index("task")
    b3.to_csv(out / "b3_complementarity.csv")
    R.append(md(b3))
    means = df.groupby(["task", "model"]).score.mean().unstack()
    R.append(f"\n\nTask-level choice (no per-example oracle): mean per-example score over the nine tasks is "
             f"{means.max(axis=1).mean():.3f} when the best model is chosen per task, against "
             f"{means.mean().max():.3f} for the best single model ({means.mean().idxmax()}).")

    # ---------- B4: by source
    R.append("\n\n## B4. Breakdown by source\n")
    b4 = df[df.task.isin(["summarization", "title_generation", "dialogue2note"])].groupby(["task", "source", "model"]).score.mean().unstack()
    n4 = df[df.model == models[0]].groupby(["task", "source"]).size()
    b4.insert(0, "n", [n4[i] for i in b4.index])
    b4.to_csv(out / "b4_by_source.csv")
    R.append(md(b4))

    # ---------- B5: by input length
    R.append("\n\n## B5. Breakdown by input length\n")
    R.append("Terciles of input length within each task (short / medium / long). Mean over the 12 models, and the "
             "number of models whose score falls from the short to the long tercile. `spearman` = mean over models "
             "of the rank correlation between input length and score.\n")
    b5, b5m = [], []
    for t in TASKS:
        x = df[df.task == t].copy()
        ex = x.drop_duplicates("example_id").set_index("example_id").n_words_input
        bins = pd.qcut(ex.rank(method="first"), 3, labels=["short", "medium", "long"])
        x["bin"] = x.example_id.map(bins)
        pm = x.groupby(["model", "bin"], observed=True).score.mean().unstack()
        rho = x.groupby("model").apply(lambda z: z.n_words_input.corr(z.score, method="spearman")).mean()
        edges = ex.groupby(bins, observed=True).agg(["min", "max"])
        b5.append({"task": t, "words_short": f"{edges.loc['short','min']}-{edges.loc['short','max']}",
                   "words_long": f"{edges.loc['long','min']}-{edges.loc['long','max']}", "short": pm["short"].mean(),
                   "medium": pm["medium"].mean(), "long": pm["long"].mean(),
                   "models_lower_on_long": int((pm["long"] < pm["short"]).sum()), "spearman": rho})
        b5m.append(pm.assign(task=t))
    b5 = pd.DataFrame(b5).set_index("task")
    b5.to_csv(out / "b5_by_length.csv")
    pd.concat(b5m).to_csv(out / "b5_by_length_per_model.csv")
    R.append(md(b5))

    # ---------- C1: classification
    R.append("\n\n## C1. Classification: per-class results and confusions\n")
    for t in CLS:
        x = df[df.task == t]
        gold = x.example_id.map(lambda i: refs[t][i]["target"])
        labels = sorted(set(gold))
        pc = pd.DataFrame({m: f1_score(gold[x.model == m], x.pred[x.model == m], labels=labels, average=None, zero_division=0)
                           for m in models}, index=labels)
        support = gold[x.model == models[0]].value_counts()
        tab = pd.DataFrame({"n": support.reindex(labels), "mean_f1": pc.mean(axis=1), f"best_model_f1 ({best[t]})": pc[best[t]],
                            "min_f1": pc.min(axis=1), "max_f1": pc.max(axis=1)}).sort_values("mean_f1").rename_axis("class")
        pc.assign(n=support.reindex(labels)).to_csv(out / f"c1_{t}_per_class_f1.csv")
        conf = pd.crosstab(gold[x.model == best[t]].values, x.pred[x.model == best[t]].values)
        conf.to_csv(out / f"c1_{t}_confusion_{best[t]}.csv")
        pairs = x[x.score == 0].assign(gold=gold[x.score == 0]).groupby(["gold", "pred"]).agg(errors=("model", "size"), models=("model", "nunique"))
        pairs = pairs.sort_values("errors", ascending=False)
        pairs.to_csv(out / f"c1_{t}_confusions_all_models.csv")
        tot = int((x.score == 0).sum())
        R.append(f"\n### {t} ({len(labels)} classes, {len(support) and int(support.sum())} examples)\n")
        R.append("Per-class F1 over the 12 models, lowest first:\n")
        R.append(md(tab))
        R.append(f"\n\nMost frequent confusions, all models pooled ({tot} errors in total):\n")
        top = pairs.head(8).reset_index()
        top["share_of_errors"] = top.errors / tot
        R.append(md(top, index=False))

    # ---------- C2: multiple choice
    R.append("\n\n## C2. Multiple-choice diagnosis: which wrong options are chosen\n")
    x = df[df.task == "diagnosis_mcq"]
    w = x.pivot(index="example_id", columns="model", values="pred")
    gold = pd.Series({i: refs["diagnosis_mcq"][i]["target"] for i in w.index})
    n_right = w.eq(gold, axis=0).sum(axis=1)
    wrong_counts = w.apply(lambda r: Counter(v for v in r if v != gold[r.name] and v in "ABCD"), axis=1)
    top_wrong = wrong_counts.map(lambda c: max(c.values()) if c else 0)
    same, pairs_n = 0, 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a_, b_ = w[models[i]], w[models[j]]
            both = (a_ != gold) & (b_ != gold) & a_.isin(list("ABCD")) & b_.isin(list("ABCD"))
            same += int((a_[both] == b_[both]).sum()); pairs_n += int(both.sum())
    c2 = pd.DataFrame({"value": {
        "examples": len(w), "all 12 models right (%)": 100 * (n_right == 12).mean(), "all 12 models wrong (%)": 100 * (n_right == 0).mean(),
        "at most 3 models right (%)": 100 * (n_right <= 3).mean(),
        "same wrong option chosen by >= 6 models (%)": 100 * (top_wrong >= 6).mean(),
        "same wrong option chosen by >= 9 models (%)": 100 * (top_wrong >= 9).mean(),
        "two wrong models agree on the wrong option (%; chance 33.3)": 100 * same / pairs_n}})
    conf = (x.assign(gold=x.example_id.map(gold), pred=x.pred.where(x.pred.isin(list("ABCD")), "none"))
            .groupby(["model", "gold", "pred"]).size().rename("n").reset_index())
    conf.to_csv(out / "c2_mcq_confusion.csv", index=False)
    c2 = c2.rename_axis("quantity")
    c2.to_csv(out / "c2_mcq_errors.csv")
    R.append(md(c2, digits=1))
    hard = pd.DataFrame({"models_right": n_right, "most_chosen_wrong_count": top_wrong}).sort_values(["models_right", "most_chosen_wrong_count"], ascending=[True, False])
    hard.to_csv(out / "c2_mcq_per_example.csv")
    R.append(f"\n\nDistribution of the number of models that answer an example correctly: "
             + ", ".join(f"{k}: {v}" for k, v in n_right.value_counts().sort_index().items()) + ".")

    # ---------- C3: open-ended diagnosis (automatic categories)
    R.append("\n\n## C3. Open-ended diagnosis: automatic error categories\n")
    R.append("Categories after normalisation (lowercase, no punctuation or articles), in this order: `exact`; "
             "`reference inside prediction` (the prediction adds words); `prediction inside reference` (the "
             "prediction is a shorter, more general form); `partial` (at least one shared word); `no overlap`. "
             "Shares in % of the 1,000 examples.\n")
    def cat(p, r):
        p, r = norm(p), norm(r)
        if not p: return "empty"
        if p == r: return "exact"
        if r and r in p: return "reference inside prediction"
        if p in r: return "prediction inside reference"
        return "partial" if set(p.split()) & set(r.split()) else "no overlap"
    x = df[df.task == "diagnosis_open"].copy()
    x["cat"] = [cat(p, refs["diagnosis_open"][i]["target"]) for p, i in zip(x.pred, x.example_id)]
    x["pred_words"] = x.pred.str.split().str.len()
    c3 = (100 * pd.crosstab(x.model, x.cat, normalize="index")).reindex(columns=["exact", "reference inside prediction", "prediction inside reference", "partial", "no overlap", "empty"]).fillna(0)
    c3["mean_pred_words"] = x.groupby("model").pred_words.mean()
    c3 = c3.loc[x.groupby("model").score.mean().sort_values(ascending=False).index]
    c3.to_csv(out / "c3_open_categories.csv")
    ref_words = np.mean([len(refs["diagnosis_open"][i]["target"].split()) for i in refs["diagnosis_open"]])
    R.append(md(c3, digits=1))
    R.append(f"\n\nMean reference length: {ref_words:.1f} words.")
    camel = sum(1 for e in refs["diagnosis_open"].values() if " " not in e["target"].strip() and re.search(r"[a-z][A-Z]", e["target"]))
    R.append(f" {camel} of the {len(refs['diagnosis_open'])} references are written without spaces in the source "
             "(for example \"AutoimmuneEnteropathy\"), so a correct answer with spaces scores zero on them.")
    coded = ROOT / "docs" / "c3_coded_sample.csv"
    if coded.exists():
        c = pd.read_csv(coded)
        tab = c.groupby("code").agg(n=("code", "size"), mean_token_f1=("token_f1", "mean")).reindex(
            ["exact", "same diagnosis, other wording", "related", "different diagnosis"])
        R.append(f"\n\n### Hand-coded sample ({len(c)} answers of qwen35_4b, fixed seed)\n")
        R.append("Each answer was read against its reference and given one code: `exact`; `same diagnosis, other "
                 "wording` (synonym, or the reference diagnosis with an added cause or site); `related` (same "
                 "disease group, or a more general or neighbouring diagnosis); `different diagnosis`. The coding was "
                 "done without clinical review and should be checked before it is reported "
                 "(`docs/c3_coded_sample.csv`).\n")
        R.append(md(tab))

    # ---------- C4: summaries and titles
    R.append("\n\n## C4. Summaries and titles: length, truncation and copying\n")
    R.append("`len_ratio` = output words / reference words (mean). `novel4` = share of the output's word 4-grams "
             "that do not occur in the input (higher = less copying). `novel4_ref` is the same for the reference.\n")
    c4 = []
    for t in ["summarization", "notes_summary", "title_generation"]:
        x = df[df.task == t]
        ref_nov = np.nanmean([novel_rate(e["target"], e["input"]) for e in refs[t].values()])
        for m in models:
            z = x[x.model == m]
            ow = z.pred.str.split().str.len()
            rw = z.example_id.map(lambda i: len(refs[t][i]["target"].split()))
            nov = np.nanmean([novel_rate(p, refs[t][i]["input"]) for p, i in zip(z.pred, z.example_id)])
            c4.append({"task": t, "model": m, "rougeL": z.score.mean(), "words": ow.mean(), "len_ratio": (ow / rw.clip(lower=1)).mean(),
                       "truncated_pct": 100 * z.truncated.mean(), "novel4": nov, "novel4_ref": ref_nov,
                       "rougeL_len_spearman": ow.corr(z.score, method="spearman")})
    c4 = pd.DataFrame(c4).set_index(["task", "model"])
    c4.to_csv(out / "c4_generation.csv")
    R.append(md(c4))

    # ---------- C5: dialogue to note
    R.append("\n\n## C5. Dialogue to note: note sections against full visit notes\n")
    x = df[df.task == "dialogue2note"]
    c5 = x.groupby(["model", "target_kind"]).agg(rougeL=("score", "mean"), truncated_pct=("truncated", lambda s: 100 * s.mean()), n=("score", "size")).unstack()
    c5.columns = [f"{a}_{b}" for a, b in c5.columns]
    c5 = c5.sort_values("rougeL_note_section", ascending=False)
    c5.to_csv(out / "c5_dialogue.csv")
    R.append(md(c5))

    # ---------- C6: format errors
    R.append("\n\n## C6. What non-compliant outputs look like\n")
    R.append("Outputs that fail the strict parse, by feature (an output can have several). `extra_lines` = more than "
             "one non-empty line where one line was requested; `tag_or_preamble` = starts with a tag such as "
             "\"Title:\" or with a phrase such as \"Here is\"; `markdown` = contains `**`, `__`, `#` headings or "
             "backticks; `quoted` = wrapped in quotation marks; `special_token` = contains a chat-template token such "
             "as `<|end|>`; `truncated` = reached the token limit. Shares in % "
             "of all outputs of the model in the task group.\n")
    pre = re.compile(r"^(here is|here's|sure|certainly|the (title|diagnosis|summary|answer|label) (is|of)|based on)", re.I)
    def feats(task, raw, trunc):
        text, stripped = strip_thinking(cut_at_end_token(raw))
        t = text.strip()
        lines = [l for l in t.splitlines() if l.strip()]
        return {"extra_lines": task in ONE_LINE and len(lines) > 1, "tag_or_preamble": bool(_PREFIX.match(t) or pre.match(t)),
                "markdown": bool(re.search(r"\*\*|__|^#+ |`", t, re.M)), "quoted": t[:1] in "\"'“‘" and len(t) > 1,
                "special_token": has_end_token(raw),
                "truncated": bool(trunc), "thinking_text": bool(stripped)}
    groups = {"classification": CLS, "multiple choice": ("diagnosis_mcq",), "open diagnosis": ("diagnosis_open",),
              "title": ("title_generation",), "summaries and notes": ("summarization", "notes_summary", "dialogue2note")}
    c6 = []
    for g, ts in groups.items():
        x = df[df.task.isin(ts)]
        F = pd.DataFrame([feats(t, r, tr) for t, r, tr in zip(x.task, x.raw, x.truncated)], index=x.index)
        bad = ~x.strict
        for m in models:
            k = x.model == m
            row = {"group": g, "model": m, "strict_fail_pct": 100 * bad[k].mean()}
            row.update({c: 100 * (F[c][k] & bad[k]).mean() for c in F.columns})
            c6.append(row)
    c6 = pd.DataFrame(c6).set_index(["group", "model"])
    c6.to_csv(out / "c6_format.csv")
    R.append(md(c6[c6.strict_fail_pct >= 5], digits=1))
    R.append("\n\n(Rows with a strict failure rate below 5% are omitted; the full table is in `c6_format.csv`.)")
    R.append("\n\nTwo of these patterns are not choices of the model about the answer. (i) In dialogue to note, the "
             "124 complete visit notes fail the strict parse for most models because section headings are written "
             "with markdown emphasis, which the parser removes. (ii) Almost every `special_token` case is Phi-3.5-mini: "
             "its end-of-turn token `<|end|>` was not a stop token in the runner, so it was written into the output "
             "and generation sometimes continued after it. Since bench-v1.1 every output is cut at the first "
             "end-of-turn token before parsing, and an output that contains one is not counted as truncated "
             "(see the note below).")
    # ---------- note: end-of-turn token (bench-v1 as scored against bench-v1.1)
    old_path = root / "summary_bench-v1_as_scored.csv"
    if old_path.exists():
        R.append("\n\n## Note: end-of-turn token in the outputs (bench-v1 against bench-v1.1)\n")
        R.append("Main metric as first scored (bench-v1, outputs parsed whole) and after cutting every output at the "
                 "first end-of-turn token (bench-v1.1). Only rows that changed are shown. `with_token` = share of "
                 "outputs that contain the token; `text_after` = share with text after it.\n")
        old = pd.read_csv(old_path); new = pd.read_csv(root / "summary.csv")
        k = ["model", "task", "metric"]
        mm = old[old.main == True].merge(new[new.main == True], on=k, suffixes=("_v1", "_v1_1"))
        mm = mm[(mm.value_v1 - mm.value_v1_1).abs() > 1e-9].copy()
        mm["change"] = mm.value_v1_1 - mm.value_v1
        tok = df.assign(w=df.raw.map(has_end_token), a=df.raw.map(lambda x: has_end_token(x) and bool(x[len(cut_at_end_token(x)):].replace("<|end|>", "").strip())))
        tok = tok.groupby(["model", "task"])[["w", "a"]].mean()
        mm["with_token"] = [tok.loc[(m, t), "w"] for m, t in zip(mm.model, mm.task)]
        mm["text_after"] = [tok.loc[(m, t), "a"] for m, t in zip(mm.model, mm.task)]
        note = mm.set_index(["model", "task"])[["metric", "value_v1", "value_v1_1", "change", "with_token", "text_after"]]
        note.to_csv(out / "note_end_token.csv")
        R.append(md(note))

    name = "bench_analysis.md" if a.subset == "test_eval" else f"bench_analysis_{a.subset}.md"
    (ROOT / "docs" / name).write_text("\n".join(R) + "\n")
    print("wrote docs/" + name, "and", out)


if __name__ == "__main__":
    main()

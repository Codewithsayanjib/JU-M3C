# Released results

Model outputs and analysis artifacts behind the paper's tables and figures. These
files contain scores, metadata and the cross-dataset leakage audit only — **no raw
case text** (the full JU-M3C corpus is released separately on Zenodo on acceptance).

| File | What it is | Backs |
|---|---|---|
| `contamination_map.json` | Share of each published test set whose documents appear in another dataset's training data, by source and layer | Table 9 (contamination), Fig. of leakage |
| `leakage_experiment.json` | Leaked vs. leak-free fine-tuning (Granite 3.3 2B, title generation): ROUGE-L on leaked and other titles, with CI | Table 10 / Fig. leak-effect |
| `analysis/per_example.csv.gz` | Per-example scores for all 12 models × 9 tasks (score, strict/lenient parse, input length) — no text | Tables 14, 15, 17 |
| `analysis/summary_reliability.csv` | Option-position accuracy, rotation stability, label-shuffle ΔF1 per model | Tables 18, 19 |
| `analysis/summary_prompt_variants.csv`, `summary_fewshot.csv` | Prompt-ablation (reworded / clinical-role / three-shot) Δ per model and task | Table 20 |
| `analysis/abstention.csv` | Coverage and accuracy at each agreement threshold k | Table 21 |
| `analysis/b3_complementarity.csv`, `vote.csv`, `ensembles.csv` | Best-single vs. per-example oracle; agreement voting | Table 22 |
| `analysis/b2_overlap*.csv`, `b4_by_source.csv` | Within-source overlap and per-source breakdowns | Sec. 5 |

The full raw per-example model generations (all models × all tasks) are attached as a
release asset (`model_outputs.tar.gz`). Regenerate `contamination_map.json` from the
corpus with `scripts/experiments` / the `jumc audit` tool.

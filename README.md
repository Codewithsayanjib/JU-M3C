# JU-M3C

**A Large-Scale, Leakage-Audited Multimodal Multi-Layer Network Corpus of Medical Cases Benchmarked with Small Vision-Language Models**

Sayanjib Sur and Pawan Kumar Singh · Department of Information Technology, Jadavpur University

[![ci](https://github.com/Codewithsayanjib/JU-M3C/actions/workflows/ci.yml/badge.svg)](https://github.com/Codewithsayanjib/JU-M3C/actions/workflows/ci.yml)
![code: MIT](https://img.shields.io/badge/code-MIT-6BBF59)
![data: CC BY-NC-SA 4.0](https://img.shields.io/badge/data-CC%20BY--NC--SA%204.0-EF9A9A)
![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)

---

## Status

This repository accompanies the paper *JU-M3C: A Large-Scale, Leakage-Audited Multimodal
Multi-Layer Network Corpus of Medical Cases Benchmarked with Small Vision-Language Models*
(Sur and Singh), **under review**.

## Data

The **JU-M3C corpus** (401,238 records from nine public clinical case datasets, the four network
layers, the article-level splits, the figure index and flags, the task subsets) and **all model
outputs** will be released on **Zenodo (CC BY-NC-SA 4.0)** with a persistent, versioned DOI
**upon acceptance** of the paper. Figure images are distributed as a licence-respecting fetch
index, not as image files.

The **code** to build the corpus from its public sources, the leakage-audit tool, and every
analysis in the paper are available here now. You can run the audit tool on the toy dataset in
`tests/fixtures/` **without any download**.

## The leakage audit, in 30 seconds

Public clinical case datasets are built from the same open-access articles, so a case in one
dataset's test set often sits in another's training set. The reusable audit finds this between any
two document sets — it needs only the standard library.

```bash
pip install -e .
jumc audit --train tests/fixtures/toy_train.jsonl --test tests/fixtures/toy_test.jsonl
```

```json
{ "n_test": 3, "leaked": 2, "leaked_pct": 66.67,
  "article_layer_only": 1, "text_layer_only": 1, "both_layers": 0,
  "copies_by_training_source": { "SourceA": 1 },
  "leaked_ids": ["tst-art", "tst-dup"] }
```

Each input is JSON Lines with one object per document:
`{"id": "...", "text": "...", "pmcid": "...", "source": "..."}`. `pmcid` enables the article
layer; both it and `source` are optional. Point `--train` and `--test` at your own datasets to
audit them before you train or report a score.

## Repository layout

```
src/ju_m3c/
  audit/        reusable cross-dataset leakage audit (standard library only)
  benchmark/    tasks, prompts, parsers, metrics, inference
  corpus/       build the corpus from the nine public sources
  network/      multi-layer network, link prediction, retrieval
scripts/        runnable pipeline: corpus · audit · benchmark · experiments · mm
configs/        model registry and source definitions
release/        data-release builder and dataset card
docs/           REPRODUCE · DATASHEET · DATA_FORMAT
tests/          unit tests + a toy dataset (no download needed)
```

## Install

```bash
pip install -e .              # core: the audit tool + parsers (numpy, scikit-learn)
pip install -e .[benchmark]   # add to run the model benchmark (MLX, transformers, ...)
pip install -e .[clip]        # add to run the BiomedCLIP / image experiments
```

`requirements-lock.txt` holds the exact frozen environment used for the paper.

## Reproducing the paper

Once the corpus is downloaded from Zenodo, `docs/REPRODUCE.md` maps each table and number to the
command that produces it. The scripts read their data location from `--data` / `--mm` arguments or
the `JUMC_DATA` / `JUMC_MV` environment variables; no path is hard-coded.

## Licence

Code is released under the **MIT** licence (`LICENSE`). The dataset will be released under
**CC BY-NC-SA 4.0**, inheriting the non-commercial share-alike terms of its sources.

## Citation

See `CITATION.cff`.

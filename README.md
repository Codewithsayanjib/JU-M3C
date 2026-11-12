<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:D9E8F5,50:E8E0F5,100:FCDAD7&height=200&section=header&text=JU-M3C&fontSize=74&fontColor=2F3B52&animation=twinkling&fontAlignY=38&desc=Leakage-Audited%20Multimodal%20Multi-Layer%20Network%20Corpus%20of%20Medical%20Cases&descSize=15&descAlignY=60" width="100%" alt="JU-M3C"/>

<a href="https://github.com/Codewithsayanjib/JU-M3C/actions/workflows/ci.yml"><img src="https://github.com/Codewithsayanjib/JU-M3C/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
<img src="https://img.shields.io/badge/Paper-under%20review-8E7CC3?style=flat-square&logo=readthedocs&logoColor=white" alt="Paper"/>
<img src="https://img.shields.io/badge/Zenodo-on%20acceptance-1682D4?style=flat-square&logo=zenodo&logoColor=white" alt="Zenodo"/>
<img src="https://img.shields.io/badge/code-MIT-6BBF59?style=flat-square" alt="MIT"/>
<img src="https://img.shields.io/badge/data-CC%20BY--NC--SA%204.0-EF9A9A?style=flat-square" alt="CC BY-NC-SA 4.0"/>
<img src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>

<br/><br/>

**A Large-Scale, Leakage-Audited Multimodal Multi-Layer Network Corpus of Medical Cases,<br/>Benchmarked with Small Vision-Language Models**

Sayanjib Sur · Pawan Kumar Singh — Department of Information Technology, Jadavpur University

<br/>

| 📄 **401,238** documents | 🔗 **247,552** articles | 🖼️ **737,203** figures | 🧪 **11** tasks | 🤖 **15** models | 💻 **16 GB** laptop |
|:---:|:---:|:---:|:---:|:---:|:---:|

</div>

---

## 🚨 The problem

Public clinical case datasets are built from the **same open-access articles**, and each draws its
own train/test split. Merge two of them and one's test set quietly becomes the other's training
data — a leak no single release can see.

<div align="center">

| Published test set | Test docs | Also in another public dataset |
|:---|---:|:---|
| **MultiClinSum**    | 3,396 | 🟥🟥🟥🟥🟥🟥🟥🟥🟥⬜ **91.0%** |
| **MedCaseReasoning**| 897   | 🟥🟥🟥🟥🟥🟥⬜⬜⬜⬜ **64.7%** |
| **CUPCase**         | 3,562 | 🟥🟥🟥🟥🟥⬜⬜⬜⬜⬜ **54.8%** |

</div>

A copy-the-nearest-case baseline exploits this to **outrank 10 of 12 language models** on
multiple-choice diagnosis — purely from leakage. **JU-M3C** makes the overlap explicit as a
four-layer network, validates its links without annotators (**99.99% precision**), and provides
splits with **zero** cross-split leakage.

---

## ⚡ Audit any two datasets in 30 seconds

The reusable audit needs only the standard library — no data, no GPU.

```bash
pip install -e .
jumc audit --train tests/fixtures/toy_train.jsonl --test tests/fixtures/toy_test.jsonl
```

```json
{ "n_test": 3, "leaked": 2, "leaked_pct": 66.67,
  "article_layer_only": 1, "text_layer_only": 1,
  "copies_by_training_source": { "SourceA": 1 },
  "leaked_ids": ["tst-art", "tst-dup"] }
```

Each line is one document: `{"id": "...", "text": "...", "pmcid": "...", "source": "..."}`
(`pmcid` enables the article layer; both it and `source` are optional). Point `--train` / `--test`
at **your** datasets to check them before you train or report a score.

---

## 📦 Status &amp; data

> **Status.** This repository accompanies the paper *JU-M3C: A Large-Scale, Leakage-Audited
> Multimodal Multi-Layer Network Corpus of Medical Cases Benchmarked with Small Vision-Language
> Models* (Sur &amp; Singh), **under review**.
>
> **Data.** The JU-M3C corpus (records, four network layers, article-level splits, figure index and
> flags, task subsets) and all model outputs will be released on **Zenodo (CC BY-NC-SA 4.0)** with a
> persistent, versioned DOI **upon acceptance**. Figure images ship as a licence-respecting fetch
> index. The code is available **now**, and the audit tool runs on the toy data with no download.

---

## 🗂️ What's inside

```
src/ju_m3c/
  audit/      ★ reusable cross-dataset leakage audit (standard library only)
  benchmark/    tasks · prompts · parsers · metrics · inference
  corpus/       build the corpus from the nine public sources
  network/      multi-layer network · link prediction · retrieval
scripts/        runnable pipeline: corpus · audit · benchmark · experiments · mm
configs/        model registry and source definitions
release/        data-release builder and dataset card
docs/           REPRODUCE · DATASHEET · DATA_FORMAT
tests/          unit tests + a toy dataset (no download needed)
```

## 🔧 Install

```bash
pip install -e .              # core: audit tool + parsers  (numpy, scikit-learn)
pip install -e ".[benchmark]" # run the model benchmark      (MLX, transformers, …)
pip install -e ".[clip]"      # run the BiomedCLIP / image experiments
```

`requirements-lock.txt` holds the exact frozen environment used for the paper.

## 🔁 Reproduce

Once the corpus is downloaded from Zenodo, [`docs/REPRODUCE.md`](docs/REPRODUCE.md) maps each table
and number to the command that produces it. Scripts read their data location from `--data` / `--mm`
or `JUMC_DATA` / `JUMC_MV`; nothing is hard-coded.

## 📜 Licence &amp; citation

Code: **MIT** (`LICENSE`). Data: **CC BY-NC-SA 4.0**, inheriting its sources' terms. Please cite the
paper — see [`CITATION.cff`](CITATION.cff).

<div align="center"><sub>Built at Jadavpur University · Kolkata, India</sub></div>

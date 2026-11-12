# Data format

The JU-M3C release (on Zenodo, upon acceptance) is JSON Lines, gzip-compressed. This document
describes the record schema so the files can be used without reading the build code.

## `corpus/records.jsonl.gz` — one object per document

| field | type | meaning |
|---|---|---|
| `id` | str | stable document id (`<source>-<local id>`) |
| `source` | str | one of the nine sources (see `configs/sources.yaml`) |
| `source_id` | str | the identifier the source uses |
| `pmcid`, `pmid` | str / null | PubMed Central / PubMed id of the originating article, when the source provides it |
| `doc_kind` | str | `case_report` · `transcribed_note` · `note_section` · `visit_note` |
| `licence` | str | licence of the record |
| `native_split` | str / null | the split the source published, if any |
| `split` | str | our article-level split: `train` · `val` · `test` |
| `duplicate_cluster` | str | id of the exact/near-duplicate cluster (one document) |
| `article_cluster` | str | id of the article cluster (the unit of splitting) |
| `text` | str | the document text |
| `n_words` | int | word count of `text` |
| `labels` | object | native labels (title, diagnosis, specialty, age, sex, …) |
| `aux` | object | auxiliary text some tasks need (abstract, dialogue, full article, reasoning) |

## `links/` — the network layers (edge lists, no text)

- `near_dup_pairs.jsonl.gz` — verified near-duplicate and containment pairs (text layer).
- article, image and topic layer edges, each as `{"a": id, "b": id, "layer": "art|txt|img|top"}`.
- `splits.jsonl.gz` — `{"id": ..., "split": ..., "rule": ...}`.

## `tasks/<task>/{train,val,test,test_eval,dev}.jsonl.gz`

Each example: `{"id": ..., "source": ..., "split": ..., "input": ..., "target": ...}`.

## `figures/index.jsonl.gz`

Per-figure metadata (article id, caption, image-type label, per-figure licence, flags). Image
files are fetched with `scripts/fetch_images.py`, which respects each figure's licence.

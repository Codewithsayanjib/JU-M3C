---
license: cc-by-nc-sa-4.0
language:
- en
pretty_name: JU-M3C
size_categories:
- 100K<n<1M
task_categories:
- text-classification
- summarization
- text-generation
- multiple-choice
- question-answering
tags:
- medical
- clinical
- case-reports
- benchmark
- small-language-models
- data-leakage
---

# JU-M3C

**JU-M3C** is a leak-free corpus of 401,238 public English clinical documents (case reports, transcribed
notes, note sections and doctor-patient dialogues) from nine public sources, converted to one record format,
de-duplicated and linked across sources, and split into training, validation and test sets **at the level of the
published article**. Its benchmark defines nine clinical tasks on it, with fixed evaluation and development
subsets, and the raw outputs of twelve small language models (1B to 4B parameters) evaluated on a laptop.

Paper: *{TITLE}* (Sur and Singh). Code: https://github.com/Codewithsayanjib/JU-M3C.

## Why article-level splits

Most public clinical case datasets derive from the same open-access articles. Between 54.8% and 91.0% of the
documents in three published test sets (MultiClinSum, MedCaseReasoning, CUPCase) belong to an article that is
also present in another public dataset. JU-M3C links documents by exact text, near-duplicate text,
containment and PubMed Central identifier, groups them into article clusters, and keeps every cluster on one side
of the split. Every published test set is contained in the JU-M3C test split.

## Contents

| Path | Content |
|---|---|
| `corpus/records.jsonl.gz` | One record per document: `doc_id`, `source`, `source_id`, `native_split`, `pmcid`, `pmid`, `doc_kind`, `licence`, `text`, `n_words`, `labels`, `aux`, `dup_cluster`, `article_cluster`, `split` |
| `links/doc_index.jsonl.gz` | Cluster identifiers per document (no text) |
| `links/near_dup_pairs.jsonl.gz` | Verified near-duplicate and containment pairs |
| `links/splits.jsonl.gz` | Split of every document and the rule that assigned it |
| `tasks/{task}/{train,val,test}.jsonl.gz` | Task examples per split |
| `tasks/{task}/test_eval.jsonl.gz` | Fixed evaluation subset (at most 1,000 examples) used for all reported results |
| `tasks/{task}/dev.jsonl.gz` | Development subset (at most 200 examples) used for all prompt and scoring decisions |
| `outputs/` | Raw outputs and scores of the twelve models (main run, option/label order, prompt variants, few-shot, 9B reference, leakage experiment) |
| `reports/` | De-duplication, split and task reports |
| `SOURCES.md` | Original sources, versions, licences and citations |
| `MANIFEST.sha256` | Checksum of every file |

Tasks: summarisation, title generation, multiple-choice diagnosis, open-ended diagnosis, note specialty, note
type, note summary, note section, dialogue to note. All targets are native labels supplied by the sources; nothing
was generated or re-annotated.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="tasks/diagnosis_mcq/test_eval.jsonl.gz")
```

## Licence

The combined corpus is released under **CC BY-NC-SA 4.0**, because PMC-Patients and part of MultiCaRe carry
non-commercial share-alike terms. Every record keeps the licence of its source (`licence` field), so a subset
restricted to permissive licences (CC BY, CC0, Apache 2.0, MIT) can be selected. See `LICENSE.md` and `SOURCES.md`.
Please also cite the original sources listed in `SOURCES.md`.

## Limitations

98.9% of the documents are published case reports, which describe unusual cases written for readers; they are not
a sample of routine hospital records. All sources are public, so the original articles may be part of the
pre-training data of language models; the splits prevent leakage between datasets, not exposure to the articles.
Labels come from the sources without re-annotation. The corpus is English-only and contains a small number of
veterinary case reports.

## Citation

```bibtex
@article{sur2026jumedcasepool,
  title   = {{TITLE}},
  author  = {Sur, Sayanjib and Singh, Pawan Kumar},
  year    = {2026},
  note    = {Under review}
}
```

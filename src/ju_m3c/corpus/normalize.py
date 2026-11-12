"""Phase 3: convert every raw source into one unified JSONL format.

  python scripts/corpus/normalize.py                 # all sources
  python scripts/corpus/normalize.py --only multicare

Reads  $CORPUS_ROOT/raw/{source}/   (never modified)
Writes $CORPUS_ROOT/interim/{source}.jsonl, one record per document:

  doc_id        "{source}:{source_id}"
  source, source_id
  native_split  split given by the source (train/val/test...) or null
  pmcid, pmid   article identifiers when the source provides them (pmcid as "PMC123")
  doc_kind      case_report | transcribed_note | note_section | visit_note
  licence       licence of the source (per-article licence for MultiCaRe)
  text          the clinical text of the document
  n_words
  labels        native labels only (nothing inferred): title, journal, year, keywords, mesh,
                summary, diagnosis, distractors, specialty, doc_type, section_header, age, gender,
                extraction (CaseReportBench categories)
  aux           extra native text kept for later tasks (dialogue, abstract, full article text,
                diagnostic reasoning)
Missing values are null / empty lists. English only.
"""
import argparse
import ast
import json
import os
import re
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
RAW, INTERIM = CORPUS_ROOT / "raw", CORPUS_ROOT / "interim"
SOURCES = yaml.safe_load(open(ROOT / "configs" / "sources.yaml"))["sources"]


def pmcid(x) -> str | None:
    m = re.search(r"(\d{4,})", str(x)) if x is not None else None
    return f"PMC{m.group(1)}" if m else None


def clean(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return s or None


def lst(x) -> list[str]:
    if x is None:
        return []
    return [str(v).strip() for v in list(x) if str(v).strip()]


def rec(source, source_id, text, doc_kind, licence=None, native_split=None, pmc=None, pmid=None,
        labels=None, aux=None) -> dict:
    text = (text or "").strip()
    return {"doc_id": f"{source}:{source_id}", "source": source, "source_id": str(source_id),
            "native_split": native_split, "pmcid": pmc, "pmid": clean(pmid), "doc_kind": doc_kind,
            "licence": licence or SOURCES[source]["licence"], "text": text, "n_words": len(text.split()),
            "labels": {k: v for k, v in (labels or {}).items() if v not in (None, [], "")},
            "aux": {k: v for k, v in (aux or {}).items() if v not in (None, [], "")}}


def raw_file(source, pattern):
    return next(p for p in sorted((RAW / source).rglob(pattern)) if not p.name.startswith("._"))


# ------------------------------------------------------------------ per-source readers
def pmc_patients():
    data = json.load(open(RAW / "pmc_patients" / "PMC-Patients-V2.json"))
    for d in data:
        yield rec("pmc_patients", d["patient_uid"], d["patient"], "case_report",
                  pmc=pmcid(d["file_path"].rsplit("/", 1)[-1]), pmid=d["PMID"],
                  labels={"title": clean(d["title"]), "year": clean(d["pub_date"]),
                          "age": clean(d["age"]), "gender": clean(d["gender"])})


def multicare():
    cases = pd.read_parquet(RAW / "multicare" / "cases.parquet")
    meta = pd.read_parquet(RAW / "multicare" / "metadata.parquet").set_index("article_id")["article_metadata"]
    abstracts = pd.read_parquet(RAW / "multicare" / "abstracts.parquet").set_index("article_id")["abstract"]
    for article_id, case_list in zip(cases["article_id"], cases["cases"]):
        m = meta.get(article_id) or {}
        for c in case_list:
            yield rec("multicare", c["case_id"], c["case_text"], "case_report",
                      licence=clean(m.get("license")) or "unspecified (PMC open access)",
                      pmc=pmcid(article_id), pmid=m.get("pmid"),
                      labels={"title": clean(m.get("title")), "journal": clean(m.get("journal")),
                              "year": clean(m.get("year")), "keywords": lst(m.get("keywords")),
                              "mesh": lst(m.get("mesh_terms")), "major_mesh": lst(m.get("major_mesh_terms")),
                              "age": clean(c.get("age")), "gender": clean(c.get("gender"))},
                      aux={"abstract": clean(abstracts.get(article_id))})


def medcasereasoning():
    df = pd.read_parquet(RAW / "medcasereasoning" / "medcasereasoning_core.pqt")
    for i, r in df.iterrows():
        # text = case presentation with the diagnosis withheld; the full article is kept in aux.
        yield rec("medcasereasoning", f"{pmcid(r['pmcid'])}_{i}", r["case_prompt"], "case_report",
                  native_split=clean(r["split"]), pmc=pmcid(r["pmcid"]),
                  labels={"title": clean(r["title"]), "journal": clean(r["journal"]),
                          "year": (clean(r["publication_date"]) or "")[:4] or None,
                          "diagnosis": clean(r["final_diagnosis"])},
                  aux={"article_text": clean(r["text"]),
                       "diagnostic_reasoning": clean(r["diagnostic_reasoning"])})


def multiclinsum():
    for zname, subset, split in [("multiclinsum_gs_train_en.zip", "gs", "train"),
                                 ("multiclinsum_large-scale_train_en.zip", "ls", "train"),
                                 ("multiclinsum_test_en.zip", "test", "test")]:
        zf = zipfile.ZipFile(RAW / "multiclinsum" / zname)
        names = [n for n in zf.namelist() if n.endswith(".txt") and "/._" not in n]
        sums = {n.rsplit("/", 1)[-1].replace("_sum.txt", ""): n for n in names if "/summaries/" in n}
        for n in sorted(x for x in names if "/summaries/" not in x):
            stem = n.rsplit("/", 1)[-1][:-4]
            summary = zf.read(sums[stem]).decode("utf8", "replace") if stem in sums else None
            yield rec("multiclinsum", stem, zf.read(n).decode("utf8", "replace"), "case_report",
                      native_split=split, labels={"summary": clean(summary)}, aux={"subset": subset})


def cupcase():
    df = pd.read_parquet(raw_file("cupcase", "*.parquet"))
    for i, r in df.iterrows():
        yield rec("cupcase", i, r["clean_case_presentation"], "case_report", native_split="test",
                  labels={"diagnosis": clean(r["correct_diagnosis"]),
                          "distractors": lst([r["distractor1"], r["distractor2"], r["distractor3"]])})


def casereportbench():
    df = pd.read_parquet(raw_file("casereportbench", "*.parquet"))
    cats = [c for c in df.columns if c not in ("pmcid", "text")]
    for _, r in df.iterrows():
        try:  # text is a stringified {section: text} dict
            sections = ast.literal_eval(r["text"])
            text = "\n\n".join(f"{k}: {v}" for k, v in sections.items()) if isinstance(sections, dict) else str(r["text"])
        except (ValueError, SyntaxError):
            text = str(r["text"])
        yield rec("casereportbench", pmcid(r["pmcid"]), text, "case_report", pmc=pmcid(r["pmcid"]),
                  labels={"extraction": {c: clean(r[c]) for c in cats if clean(r[c])}})


def mts_dialog():
    tf = tarfile.open(raw_file("mts_dialog", "*.tar.gz"))
    files = {"MTS-Dialog-TrainingSet.csv": "train", "MTS-Dialog-ValidationSet.csv": "val",
             "MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv": "test1",
             "MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv": "test2"}
    for member in tf.getmembers():
        split = files.get(member.name.rsplit("/", 1)[-1])
        if split and "/Main-Dataset/" in member.name:  # augmented (back-translated) sets are skipped
            df = pd.read_csv(tf.extractfile(member))
            for _, r in df.iterrows():
                yield rec("mts_dialog", f"{split}_{r['ID']}", r["section_text"], "note_section",
                          native_split=split, labels={"section_header": clean(r["section_header"])},
                          aux={"dialogue": clean(r["dialogue"])})


def aci_bench():
    tf = tarfile.open(raw_file("aci_bench", "*.tar.gz"))
    files = {"train.csv": "train", "valid.csv": "val", "clinicalnlp_taskB_test1.csv": "test1",
             "clinicalnlp_taskC_test2.csv": "test2", "clef_taskC_test3.csv": "test3"}
    for member in tf.getmembers():
        split = files.get(member.name.rsplit("/", 1)[-1])
        if split and "/data/challenge_data/" in member.name:
            df = pd.read_csv(tf.extractfile(member))
            for _, r in df.iterrows():
                yield rec("aci_bench", f"{split}_{r['encounter_id']}", r["note"], "visit_note",
                          native_split=split, labels={"subset": clean(r["dataset"])},
                          aux={"dialogue": clean(r["dialogue"])})


def mtsamples():
    for line in open(RAW / "mtsamples" / "notes.jsonl"):
        n = json.loads(line)
        yield rec("mtsamples", n["note_id"], n["text"], "transcribed_note", native_split=n["split"],
                  labels={"specialty": n["t1_specialty"], "doc_type": n["t2_doc_type"],
                          "summary": clean(n["description"]), "keywords": lst(n["keywords"]),
                          "title": clean(n["sample_name"]), "all_specialties": lst(n["specialties"]),
                          "all_doc_types": lst(n["doc_types"])})


READERS = {"pmc_patients": pmc_patients, "multicare": multicare, "medcasereasoning": medcasereasoning,
           "multiclinsum": multiclinsum, "cupcase": cupcase, "casereportbench": casereportbench,
           "mts_dialog": mts_dialog, "aci_bench": aci_bench, "mtsamples": mtsamples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    INTERIM.mkdir(parents=True, exist_ok=True)
    stats = {}
    for key in (a.only.split(",") if a.only else READERS):
        n = empty = words = 0
        seen = set()
        with open(INTERIM / f"{key}.jsonl", "w") as f:
            for r in READERS[key]():
                if not r["text"]:
                    empty += 1
                    continue
                assert r["doc_id"] not in seen, f"duplicate doc_id {r['doc_id']}"
                seen.add(r["doc_id"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                words += r["n_words"]
        stats[key] = {"docs": n, "empty_skipped": empty, "mean_words": round(words / max(n, 1), 1)}
        print(key, stats[key], flush=True)
    (CORPUS_ROOT / "reports").mkdir(exist_ok=True)
    out = CORPUS_ROOT / "reports" / "normalize_stats.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(stats)
    out.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()

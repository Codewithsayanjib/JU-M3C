"""Publication year per PMCID via NCBI E-utilities esummary (db=pmc, 200 IDs per request).

Used for the before/after-cutoff contamination check (e.g. BiomedCLIP / PMC-15M).
Resumable: PMCIDs already in the output are skipped. Stays under 3 requests/s (no API key).

Usage:
  python scripts/mm/fetch_pubyear.py --articles $JUMC_MV/full/articles.jsonl \
      --out $JUMC_MV/full/pubyear.jsonl
"""
import argparse, json, re, time, urllib.parse, urllib.request
from pathlib import Path

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(l)["pmcid"] for l in open(out)} if out.exists() else set()
    ids = [json.loads(l)["pmcid"] for l in open(a.articles)]
    todo = [p for p in dict.fromkeys(ids) if p not in done]
    print(f"{len(todo)} to fetch", flush=True)
    with open(out, "a") as fo:
        for i in range(0, len(todo), 200):
            batch = todo[i:i + 200]
            q = urllib.parse.urlencode({"db": "pmc", "retmode": "json", "id": ",".join(p[3:] for p in batch)})
            for t in range(5):
                try:
                    res = json.loads(urllib.request.urlopen(f"{URL}?{q}", timeout=60).read())["result"]
                    break
                except Exception:
                    time.sleep(2 ** t)
            else:
                continue
            for p in batch:
                r = res.get(p[3:], {})
                d = r.get("pubdate") or r.get("epubdate") or ""
                m = re.search(r"(19|20)\d{2}", d)
                fo.write(json.dumps({"pmcid": p, "pubdate": d, "year": int(m.group()) if m else None}) + "\n")
            fo.flush()
            time.sleep(0.4)
            if i % 20000 == 0:
                print(i, flush=True)


if __name__ == "__main__":
    main()

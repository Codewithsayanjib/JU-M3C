"""MeSH terms and publication types per article from PubMed (E-utilities efetch, 200 PMIDs per request).

PMIDs come from the corpus records; PMCIDs without one are resolved with the PMC ID Converter.
MeSH headings are assigned by NLM indexers, so this is an existing expert annotation layer,
not a label we create. Recently published articles may not be indexed yet (mesh = []).

Resumable: PMCIDs already in the output are skipped. Stays under 3 requests/s (no API key).

Usage:
  python scripts/mm/fetch_mesh.py --data $JUMC_DATA \
      --out $JUMC_MV/full/mesh.jsonl
"""
import argparse, gzip, json, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def get(url, data=None):
    for t in range(5):
        try:
            return urllib.request.urlopen(url, data=data, timeout=120).read()
        except Exception:
            time.sleep(2 ** t)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, out = Path(a.data).expanduser(), Path(a.out)

    pmc2pmid = {}
    for l in gzip.open(data / "corpus/records.jsonl.gz", "rt"):
        r = json.loads(l)
        if r["pmcid"] and (r.get("pmid") or r["pmcid"] not in pmc2pmid):
            pmc2pmid[r["pmcid"]] = str(r["pmid"]) if r.get("pmid") else None
    missing = [p for p, v in pmc2pmid.items() if v is None]
    print(f"{len(pmc2pmid)} PMCIDs, {len(missing)} without PMID -> ID converter", flush=True)
    for i in range(0, len(missing), 200):
        q = urllib.parse.urlencode({"ids": ",".join(missing[i:i + 200]), "format": "json", "tool": "jumedcasepool"})
        raw = get(f"{IDCONV}?{q}")
        if raw:
            for rec in json.loads(raw).get("records", []):
                if rec.get("pmid"):
                    pmc2pmid[rec["pmcid"]] = str(rec["pmid"])
        time.sleep(0.4)

    done = {json.loads(l)["pmcid"] for l in open(out)} if out.exists() else set()
    todo = [(p, m) for p, m in pmc2pmid.items() if p not in done]
    print(f"{len(todo)} to fetch", flush=True)
    with open(out, "a") as fo:
        for i in range(0, len(todo), 200):
            batch = todo[i:i + 200]
            have = {m: p for p, m in batch if m}
            recs = {}
            if have:
                raw = get(EFETCH, urllib.parse.urlencode({"db": "pubmed", "retmode": "xml", "id": ",".join(have)}).encode())
                if raw is None:
                    continue  # retried later on resume
                for art in ET.fromstring(raw).iter("PubmedArticle"):
                    pmid = art.findtext(".//MedlineCitation/PMID")
                    mesh = []
                    for h in art.iter("MeshHeading"):
                        d = h.find("DescriptorName")
                        mesh.append({"ui": d.get("UI"), "name": d.text, "major": d.get("MajorTopicYN") == "Y",
                                     "qualifiers": [q.text for q in h.findall("QualifierName")]})
                    recs[pmid] = {"mesh": mesh,
                                  "pub_types": [t.text for t in art.iter("PublicationType")],
                                  "keywords": [k.text for k in art.iter("Keyword") if k.text]}
            for p, m in batch:
                r = recs.get(m, {"mesh": [], "pub_types": [], "keywords": []})
                fo.write(json.dumps({"pmcid": p, "pmid": m, **r}) + "\n")
            fo.flush()
            time.sleep(0.4)
            if i % 20000 == 0:
                print(i, flush=True)
    print("MESH_DONE", flush=True)


if __name__ == "__main__":
    main()

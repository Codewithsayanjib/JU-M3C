"""Fetch figures, captions and citing sentences from the PMC Open Access S3 bucket.

For each PMCID: pick the latest article version, read its metadata JSON and JATS XML,
parse every <fig> (label, caption, image file, sentences that cite it), download the
image, resize to max side 512 px (JPEG q90) and write one JSON line per figure.

Images are packed into tar shards (shards/NNNNN.tar, about 1,000 articles each); every
start opens a new shard, so a stopped run never rewrites an old one.
Resumable and pausable: stop with Ctrl-C or kill; rerunning skips PMCIDs in done.txt.
Public bucket, no credentials.

Usage:
  python scripts/mm/fetch_figures.py --doc-index links/doc_index.jsonl.gz \
      --out $JUMC_MV --sample 200 --workers 16
"""
import argparse, gzip, io, tarfile, json, random, re, threading, time, urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
MAX_SIDE = 512
lock = threading.Lock()


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def latest_version(pmcid):
    xml = get(f"{BUCKET}/?list-type=2&delimiter=/&prefix={pmcid}.").decode()
    vers = [int(v) for v in re.findall(rf"<Prefix>{pmcid}\.(\d+)/</Prefix>", xml)]
    return max(vers) if vers else None


def text_of(el):
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def citing_sentences(root, fig_id):
    out = []
    for p in root.iter("p"):
        if any(x.get("rid", "").split().count(fig_id) for x in p.iter("xref")):
            for s in re.split(r"(?<=[.!?])\s+", text_of(p)):
                if s and (fig_id in s or re.search(r"\b(Fig|Figure)", s)):
                    out.append(s)
    return out[:5]


def parse_figs(xml_bytes):
    root = ET.fromstring(xml_bytes)
    figs = []
    for fig in root.iter("fig"):
        g = fig.find(".//graphic")
        href = None
        if g is not None:
            href = next((v for k, v in g.attrib.items() if k.endswith("href")), None)
        fid = fig.get("id", "")
        figs.append({
            "fig_id": fid,
            "label": text_of(fig.find("label")),
            "caption": text_of(fig.find("caption")),
            "graphic": href,
            "citing_sentences": citing_sentences(root, fid) if fid else [],
        })
    return figs


def find_media(media_urls, href):
    if not href:
        return None
    stem = href.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    for u in media_urls:
        name = u.split("?")[0].rsplit("/", 1)[-1]
        if name.rsplit(".", 1)[0] == stem and name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff")):
            return u.split("?")[0].replace("s3://pmc-oa-opendata", BUCKET)
    return None


def process(pmcid, out):
    ver = latest_version(pmcid)
    if ver is None:
        return {"pmcid": pmcid, "status": "not_in_bucket"}, [], []
    base = f"{BUCKET}/{pmcid}.{ver}/{pmcid}.{ver}"
    meta = json.loads(get(base + ".json"))
    figs = parse_figs(get(base + ".xml"))
    rows, images, n_bytes = [], [], 0
    for i, f in enumerate(figs):
        url = find_media(meta.get("media_urls", []), f["graphic"])
        row = {"pmcid": pmcid, "version": ver, "fig_index": i, **f,
               "licence": meta.get("license_code"), "source_url": url, "image_path": None}
        if url:
            try:
                raw = get(url)
                n_bytes += len(raw)
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                row["orig_size"] = im.size
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=90)
                row["image_path"] = f"{pmcid}_{i}.jpg"
                images.append((row["image_path"], buf.getvalue()))
            except Exception as e:
                row["error"] = repr(e)[:200]
        rows.append(row)
    return {"pmcid": pmcid, "status": "ok", "licence": meta.get("license_code"),
            "is_retracted": meta.get("is_retracted"), "n_figs": len(figs),
            "n_images": sum(r["image_path"] is not None for r in rows), "orig_bytes": n_bytes}, rows, images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="random sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pmcids = sorted({json.loads(l)["pmcid"] for l in gzip.open(a.doc_index, "rt")} - {None})
    if a.sample:
        pmcids = sorted(random.Random(a.seed).sample(pmcids, a.sample))
    done_f = out / "done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    todo = [p for p in pmcids if p not in done]
    print(f"{len(pmcids)} PMCIDs, {len(todo)} to fetch", flush=True)

    shard_dir = out / "shards"; shard_dir.mkdir(exist_ok=True)
    shard_no = len(list(shard_dir.glob("*.tar")))
    tar, in_shard = None, 0

    t0, n = time.time(), 0
    with open(out / "figures.jsonl", "a") as ff, open(out / "articles.jsonl", "a") as fa, \
         open(done_f, "a") as fd, ThreadPoolExecutor(a.workers) as ex:
        for pmcid, fut in [(p, ex.submit(process, p, out)) for p in todo]:
            try:
                art, rows, images = fut.result()
            except Exception as e:
                art, rows, images = {"pmcid": pmcid, "status": "error", "error": repr(e)[:200]}, [], []
            with lock:
                if tar is None or in_shard >= 1000:
                    if tar: tar.close()
                    name = f"{shard_no:05d}.tar"; shard_no += 1
                    tar, in_shard = tarfile.open(shard_dir / name, "w"), 0
                for fname, data in images:
                    ti = tarfile.TarInfo(fname); ti.size = len(data)
                    tar.addfile(ti, io.BytesIO(data))
                tar.fileobj.flush(); in_shard += 1
                for r in rows:
                    if r["image_path"]: r["shard"] = name
                    ff.write(json.dumps(r) + "\n")
                fa.write(json.dumps(art) + "\n"); fd.write(pmcid + "\n")
                ff.flush(); fa.flush(); fd.flush()
            n += 1
            if n % 500 == 0:
                print(f"{n}/{len(todo)}  {n / (time.time() - t0):.1f} art/s", flush=True)
    if tar: tar.close()
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

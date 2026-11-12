"""Download the corpus sources listed in configs/sources.yaml to $CORPUS_ROOT/raw/{key}/.

  python scripts/corpus/download.py                 # all sources
  python scripts/corpus/download.py --only pmc_patients,multicare

Pinned versions, resumable, public sources only. After each source a SHA-256 manifest
(raw/{key}/MANIFEST.sha256) and a SOURCE.json (URL, revision, licence, date) are written.
Raw files are never modified afterwards.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "corpus"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet downloads stall on this Mac


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-sS", "-o", str(dest), url],
                   check=True)


def get_huggingface(cfg, out: Path):
    from huggingface_hub import hf_hub_download, snapshot_download
    kw = dict(repo_id=cfg["repo"], repo_type="dataset", revision=cfg["revision"], local_dir=str(out))
    if cfg.get("files"):
        for f in cfg["files"]:
            hf_hub_download(filename=f, **kw)
    else:
        snapshot_download(**kw)
    shutil.rmtree(out / ".cache", ignore_errors=True)  # hub bookkeeping, not data
    return f"https://huggingface.co/datasets/{cfg['repo']}/tree/{cfg['revision']}"


def get_zenodo(cfg, out: Path):
    api = f"https://zenodo.org/api/records/{cfg['record']}"
    rec = json.load(urllib.request.urlopen(api))
    by_name = {f["key"]: f for f in rec["files"]}
    for name in cfg["files"]:
        f = by_name[name]
        dest = out / name
        want = f["checksum"].split(":")[-1]
        if not (dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == want):
            curl(f"{api}/files/{name}/content", dest)
        got = hashlib.md5(dest.read_bytes()).hexdigest()
        assert got == want, f"{name}: md5 {got} != zenodo {want}"
    return f"https://zenodo.org/records/{cfg['record']}"


def get_github(cfg, out: Path):
    tar = out / f"{cfg['repo'].replace('/', '__')}-{cfg['revision'][:10]}.tar.gz"
    if not tar.exists():
        curl(f"https://github.com/{cfg['repo']}/archive/{cfg['revision']}.tar.gz", tar)
    with tarfile.open(tar) as t:  # sanity check only; the archive itself is the raw artefact
        assert len(t.getnames()) > 1
    return f"https://github.com/{cfg['repo']}/tree/{cfg['revision']}"


def get_local(cfg, out: Path):
    src = ROOT / cfg["path"]
    shutil.copy2(src, out / src.name)
    return f"local:{cfg['path']}"


GETTERS = {"huggingface": get_huggingface, "zenodo": get_zenodo, "github": get_github, "local": get_local}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    sources = yaml.safe_load(open(ROOT / "configs" / "sources.yaml"))["sources"]
    keys = a.only.split(",") if a.only else list(sources)
    if not CORPUS_ROOT.parent.exists():
        sys.exit(f"{CORPUS_ROOT.parent} does not exist (set CORPUS_ROOT).")
    for key in keys:
        cfg = sources[key]
        out = CORPUS_ROOT / "raw" / key
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{key}] {cfg['name']} ...", flush=True)
        url = GETTERS[cfg["kind"]](cfg, out)
        files = sorted(p for p in out.rglob("*") if p.is_file()
                       and p.name not in ("MANIFEST.sha256", "SOURCE.json") and not p.name.startswith("."))
        with open(out / "MANIFEST.sha256", "w") as m:
            for p in files:
                m.write(f"{sha256(p)}  {p.relative_to(out)}\n")
        (out / "SOURCE.json").write_text(json.dumps({
            "key": key, "name": cfg["name"], "url": url, "revision": cfg.get("revision", cfg.get("record")),
            "licence": cfg["licence"], "citation": cfg["citation"],
            "downloaded": datetime.now(timezone.utc).isoformat(),
            "files": len(files), "bytes": sum(p.stat().st_size for p in files)}, indent=1))
        print(f"[{key}] {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB", flush=True)
    print("done")


if __name__ == "__main__":
    main()

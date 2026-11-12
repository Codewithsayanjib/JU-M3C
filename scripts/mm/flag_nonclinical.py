"""Flag likely non-clinical figures (no image analysis; caption rules + repeated templates).

Categories (first match wins):
  video_placeholder  caption mentions a video/movie/animation
  flow_diagram       PRISMA, flow chart/diagram, algorithm, timeline
  table_or_chart     table, graph, plot, chart, curve, Kaplan-Meier, bar
  schematic          schematic, illustration, drawing, cartoon, diagram, model of
  template           image confirmed as a duplicate of images in >= 3 other articles
Writes figure_flags.jsonl (image_path, flag) and prints counts, plus overlap with MM items.

Usage:
  python scripts/mm/flag_nonclinical.py --mm $JUMC_MV/full \
      --pairs $JUMC_MV/reports/mm_leakage_full.pairs.jsonl
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

RULES = [
    # only explicit video/movie references ("Video 1", "supplementary movie"); video-assisted surgery,
    # video-EEG, videolaryngoscopy and capsule endoscopy stills are clinical (spot-check, 2026-11-07)
    ("video_placeholder", r"^\W*(video|movie)\s*s?\d|\b(supplementary|additional) (video|movie|clip)s?\b|\b(video|movie)\s*s?\d+\b(?!\s*(shows?|showing|demonstrat))"),
    ("flow_diagram", r"\b(prisma|flow ?charts?|flow ?diagrams?|flowcharts?|algorithms?|timelines?|time-?lines?)\b"),
    ("table_or_chart", r"\b(tables?|graphs?|plots?|charts?|curves?|kaplan[- ]meier|bar diagram|histogram|trends? of)\b"),
    ("schematic", r"\b(schematics?|schematic(ally)?|illustrations?|drawings?|cartoons?|diagrams?|diagrammatic|model of|pedigree)\b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", required=True); ap.add_argument("--pairs", required=True)
    a = ap.parse_args()
    mm = Path(a.mm)
    partners = defaultdict(set)
    for l in open(a.pairs):
        p = json.loads(l)
        pa, pb = p["image_a"].split("_")[0], p["image_b"].split("_")[0]
        partners[p["image_a"]].add(pb); partners[p["image_b"]].add(pa)
    rx = [(k, re.compile(r, re.I)) for k, r in RULES]
    cnt, n, flags = Counter(), 0, {}
    with open(mm / "figure_flags.jsonl", "w") as fo:
        for l in open(mm / "figures.jsonl"):
            r = json.loads(l)
            ip = r.get("image_path")
            if not ip:
                continue
            n += 1
            cap = f"{r.get('label') or ''} {r.get('caption') or ''}"
            f = next((k for k, x in rx if x.search(cap)), None)
            if f == "video_placeholder" and re.search(r"video[- ]?(assisted|eeg|laryngo|endoscop|scop|capsule)|capsule endoscop", cap, re.I):
                f = None
            if f is None and len(partners.get(ip, ())) >= 3:
                f = "template"
            if f:
                flags[ip] = f; cnt[f] += 1
                fo.write(json.dumps({"image_path": ip, "flag": f}) + "\n")
    print(f"images {n}, flagged {len(flags)} ({100 * len(flags) / n:.1f}%)", dict(cnt))
    for t in ("mm1_dev", "mm1_test", "mm2_dev", "mm2_test"):
        items = [json.loads(l) for l in open(mm / "items" / f"{t}.jsonl")]
        hit = [it for it in items if any(i in flags for i in it["images"])]
        gold = 0
        if t.startswith("mm1"):
            gold = sum(it["images"]["ABCD".index(it["answer"])] in flags for it in items)
        print(t, len(items), "items with any flagged image", len(hit), "gold flagged", gold,
              Counter(flags[i] for it in hit for i in it["images"] if i in flags))


if __name__ == "__main__":
    main()

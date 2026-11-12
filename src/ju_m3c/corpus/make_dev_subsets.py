"""Development subsets for prompt and scoring design (never the test evaluation subset).

  python scripts/corpus/make_dev_subsets.py

Writes $CORPUS_ROOT/processed/tasks/{task}/dev.jsonl: <= 200 examples per task, one per article
cluster, stratified by source, seed 42. Source = val.jsonl. diagnosis_mcq has no val split
(CUPCase is test-only), so its dev examples come from test.jsonl, restricted to article clusters
that are NOT in test_eval.jsonl; they are excluded from final evaluation by construction.
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location("bt", Path(__file__).parent / "build_tasks.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)
DEV_N = 200
bt.MIN_PER_SOURCE = 20


def main():
    root = bt.CORPUS_ROOT / "processed" / "tasks"
    report = {}
    for task in bt.TASKS:
        src = root / task / "val.jsonl"
        rows = [json.loads(l) for l in open(src)]
        origin = "val"
        if not rows:
            used = {json.loads(l)["article_cluster"] for l in open(root / task / "test_eval.jsonl")}
            rows = [r for r in map(json.loads, open(root / task / "test.jsonl")) if r["article_cluster"] not in used]
            origin = "test (outside test_eval)"
        keep = {r["example_id"] for r in bt.eval_subset(rows, DEV_N)}
        out = [r for r in rows if r["example_id"] in keep]
        with open(root / task / "dev.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[task] = {"n": len(out), "from": origin, "by_source": dict(Counter(r["source"] for r in out))}
        print(f"{task:17} dev {len(out):4} from {origin:26} {report[task]['by_source']}")
    # dev must not overlap the evaluation subset
    for task in bt.TASKS:
        ev = {json.loads(l)["article_cluster"] for l in open(root / task / "test_eval.jsonl")}
        dv = {json.loads(l)["article_cluster"] for l in open(root / task / "dev.jsonl")}
        assert not ev & dv, f"{task}: dev overlaps test_eval"
    (bt.CORPUS_ROOT / "reports" / "dev_subsets.json").write_text(json.dumps(report, indent=1))
    print("no overlap between dev and test_eval: OK")


if __name__ == "__main__":
    main()

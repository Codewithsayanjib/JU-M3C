"""Command-line entry point for the JU-M3C leakage audit.

    jumc audit --train train.jsonl --test test.jsonl [--jaccard 0.9] [--containment 0.9] [--out report.json]

Each input is JSON Lines with one object per document: ``{"id": "...", "text": "...", "pmcid": "...",
"source": "..."}``. ``pmcid`` and ``source`` are optional; ``pmcid`` enables the article layer.
Prints a contamination report as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import DetectorConfig
from .report import audit


def _read_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jumc audit", description="Cross-dataset leakage audit (JU-M3C).")
    ap.add_argument("--train", required=True, help="training-pool documents (JSON Lines)")
    ap.add_argument("--test", required=True, help="test-set documents (JSON Lines)")
    ap.add_argument("--jaccard", type=float, default=0.9)
    ap.add_argument("--containment", type=float, default=0.9)
    ap.add_argument("--out", default="-", help="write the report here (default: stdout)")
    a = ap.parse_args(argv)

    cfg = DetectorConfig(jaccard=a.jaccard, containment=a.containment)
    report = audit(_read_jsonl(a.train), _read_jsonl(a.test), cfg)
    text = json.dumps(report, indent=2)
    if a.out == "-":
        print(text)
    else:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}: {report['leaked']}/{report['n_test']} test documents leaked "
              f"({report['leaked_pct']}%)", file=sys.stderr)
    return 0


def entrypoint() -> None:
    # console_scripts target: dispatches `jumc audit ...`
    argv = sys.argv[1:]
    if argv and argv[0] == "audit":
        raise SystemExit(main(argv[1:]))
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: jumc audit --train train.jsonl --test test.jsonl [options]")
        raise SystemExit(0)
    print(f"jumc: unknown command {argv[0]!r}; the available command is 'audit'.", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/bash
# Leakage-impact experiment (A1), all steps in order; each step is skipped when already complete.
#   CORPUS_ROOT=corpus nohup caffeinate -i scripts/experiments/run_leakage_all.sh >> logs/leakage.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=results/leakage
echo "leakage started $(date)"
while pgrep -f "scripts/benchmark/run_bench.py|scripts/experiments/run_reliability.py|scripts/experiments/run_fewshot.py" >/dev/null; do sleep 30; done
[ -f $OUT/data/meta.json ] || { echo "training sets missing: run 'leakage_experiment.py build' with CORPUS_ROOT set"; exit 1; }
for cond in leaky clean; do
  if [ -f "$OUT/adapters/$cond/DONE" ]; then echo "### train $cond already done"; continue; fi
  # an interrupted run resumes from its last checkpoint (saved every 200 steps)
  echo "### train $cond start $(date +%H:%M)"
  .venv/bin/python scripts/experiments/leakage_experiment.py train $cond 2>&1 | tr '\r' '\n' | grep -E "Iter [0-9]+: (Train|Val)|Saved final|resuming|Error|Traceback" || true
  [ -f "$OUT/adapters/$cond/DONE" ] || { echo "### train $cond FAILED $(date)"; exit 1; }
  echo "### train $cond done $(date +%H:%M)"
done
for cond in base clean leaky; do
  echo "### eval $cond start $(date +%H:%M)"
  .venv/bin/python scripts/experiments/leakage_experiment.py eval $cond 2>&1 | grep -E "^(base|clean|leaky) [0-9]+|Error|Traceback"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "### eval $cond FAILED $(date)"; exit 1; }
  echo "### eval $cond done $(date +%H:%M)"
done
.venv/bin/python scripts/experiments/leakage_experiment.py score
echo "all done $(date)"

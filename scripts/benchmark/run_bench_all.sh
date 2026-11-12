#!/bin/bash
# Run the corpus benchmark for a list of models, one at a time (resumable).
#   nohup caffeinate -i scripts/benchmark/run_bench_all.sh dev > logs/bench_dev.log 2>&1 &
#   nohup caffeinate -i scripts/benchmark/run_bench_all.sh test_eval --confirm-test > logs/bench_test_eval.log 2>&1 &
# 12 models. MedGemma 1.5 4B was run on dev and then left out (2026-10-06); its dev outputs are kept in
# results/bench/archive/dev/excluded/.
# Optional: MODELS="qwen35_4b gemma4_e4b" to run a subset.
set -u
export HF_HUB_OFFLINE=1   # never download weights mid-run; a missing model fails fast instead
cd "$(dirname "$0")/.."
SUBSET=${1:-dev}; shift || true
MODELS=${MODELS:-"lfm25_1p2b llama32_1b qwen35_2b gemma4_e2b granite33_2b llama32_3b smollm3_3b ministral3_3b phi35_mini phi4_mini qwen35_4b gemma4_e4b"}
echo "bench $SUBSET started $(date)"
while pgrep -f "scripts/benchmark/run_bench.py" >/dev/null; do sleep 30; done
[ -d "${CORPUS_ROOT:-corpus}/processed/tasks" ] || { echo "corpus not found (set CORPUS_ROOT)"; exit 1; }
for m in $MODELS; do
  echo "### $m start $(date +%H:%M)"
  .venv/bin/python scripts/benchmark/run_bench.py --model "$m" --subset "$SUBSET" "$@" 2>&1 \
    | grep -E "Loaded|examples,|Peak|Error|error|Traceback|Exception|SSD"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then echo "### $m FAILED (exit $status) $(date)"; exit "$status"; fi
  echo "### $m done $(date +%H:%M)"
done
echo "all done $(date)"

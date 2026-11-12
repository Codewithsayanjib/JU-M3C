#!/bin/bash
# Few-shot runs (three solved examples before the frozen prompt) for the 12 models, one at a time (resumable).
#   nohup caffeinate -i scripts/experiments/run_fewshot_all.sh >> logs/fewshot.log 2>&1 &
# Optional: MODELS="qwen35_4b gemma4_e4b" to run a subset; extra flags are passed to run_fewshot.py.
set -u
export HF_HUB_OFFLINE=1   # never download weights mid-run; a missing model fails fast instead
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"lfm25_1p2b llama32_1b qwen35_2b gemma4_e2b granite33_2b llama32_3b smollm3_3b ministral3_3b phi35_mini phi4_mini qwen35_4b gemma4_e4b"}
echo "fewshot started $(date)"
while pgrep -f "scripts/benchmark/run_bench.py|scripts/run_zeroshot.py|scripts/experiments/run_reliability.py" >/dev/null; do sleep 30; done
[ -d "${CORPUS_ROOT:-corpus}/processed/tasks" ] || { echo "corpus not found (set CORPUS_ROOT)"; exit 1; }
for m in $MODELS; do
  echo "### $m start $(date +%H:%M)"
  .venv/bin/python scripts/experiments/run_fewshot.py --model "$m" "$@" 2>&1 \
    | grep -E "Loaded|examples,|wall|Peak|Error|error|Traceback|Exception|SSD"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then echo "### $m FAILED (exit $status) $(date)"; exit "$status"; fi
  echo "### $m done $(date +%H:%M)"
done
echo "all done $(date)"

#!/bin/bash
# Prompt-variant runs (paraphrased instruction, then clinical role) for the 12 models, one at a time (resumable).
#   nohup caffeinate -i scripts/experiments/run_prompt_variants_all.sh >> logs/prompt_variants.log 2>&1 &
# Optional: STYLES="role" or MODELS="qwen35_4b gemma4_e4b" to run a subset.
# Task files come from corpus/; model weights are downloaded from the Hugging Face Hub.
set -u
export HF_HUB_OFFLINE=1   # never download weights mid-run; a missing model fails fast instead
cd "$(dirname "$0")/.."
STYLES=${STYLES:-"paraphrase role"}
MODELS=${MODELS:-"lfm25_1p2b llama32_1b qwen35_2b gemma4_e2b granite33_2b llama32_3b smollm3_3b ministral3_3b phi35_mini phi4_mini qwen35_4b gemma4_e4b"}
echo "prompt variants started $(date)"
while pgrep -f "scripts/benchmark/run_bench.py|scripts/experiments/run_reliability.py|scripts/experiments/run_fewshot.py|leakage_experiment.py|mlx_lm lora" >/dev/null; do sleep 30; done
for s in $STYLES; do
  for m in $MODELS; do
    echo "### $s $m start $(date +%H:%M)"
    .venv/bin/python scripts/experiments/run_prompt_variants.py --model "$m" --style "$s" "$@" 2>&1 \
      | grep -E "Loaded|examples,|wall|Peak|Error|error|Traceback|Exception"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then echo "### $s $m FAILED (exit $status) $(date)"; exit "$status"; fi
    echo "### $s $m done $(date +%H:%M)"
  done
done
echo "all done $(date)"

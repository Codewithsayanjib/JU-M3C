#!/bin/bash
# Reference run: Qwen3.5-9B (4-bit MLX), zero-shot with the frozen prompts on the same 300 examples per task
# as the few-shot and prompt-variant runs. Downloads the weights once, then runs offline (resumable).
#   nohup caffeinate -i scripts/experiments/run_reference_9b.sh >> logs/reference_9b.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
REPO=mlx-community/Qwen3.5-9B-MLX-4bit
REV=938d8919941c6e7efd3c7150eff7fe9d12afa631
echo "reference 9b started $(date)"
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$REPO', revision='$REV')
print('downloaded to', p, flush=True)" || { echo "### download FAILED $(date)"; exit 1; }
echo "### run start $(date +%H:%M)"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/experiments/run_prompt_variants.py --model qwen35_9b_ref --style zeroshot 2>&1 \
  | grep -E "Loaded|examples,|wall|Peak|Error|error|Traceback|Exception"
status=${PIPESTATUS[0]}
[ "$status" -eq 0 ] || { echo "### run FAILED (exit $status) $(date)"; exit "$status"; }
echo "all done $(date)"

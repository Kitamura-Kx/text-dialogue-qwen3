#!/usr/bin/env bash
set -euo pipefail

LOCAL_PYTHON="${LOCAL_PYTHON:-/tmp/text-dialogue-qwen3-local-venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/random_travel_dialogues_5x1000}"
BATCH_COUNT="${BATCH_COUNT:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BASE_SEED="${BASE_SEED:-1000}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RESUME="${RESUME:-1}"
HF_HOME="${HF_HOME:-$(pwd)/.hf-cache}"

if [[ ! -x "$LOCAL_PYTHON" ]]; then
  echo "Local generation environment is not installed. Run: bash scripts/setup_local.sh" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

export HF_HOME
exec "$LOCAL_PYTHON" -m generate_random_travel_local_batch \
  --model "$MODEL" \
  --prompt-file prompts/random_travel_dialogues_json.txt \
  --output-dir "$OUTPUT_DIR" \
  --file-count "$BATCH_COUNT" \
  --dialogues-per-file 5 \
  --batch-size "$BATCH_SIZE" \
  --base-seed "$BASE_SEED" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-retries "$MAX_RETRIES" \
  "${RESUME_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

LOCAL_PYTHON="${LOCAL_PYTHON:-/tmp/text-dialogue-qwen3-local-venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PROMPT_FILE="${PROMPT_FILE:-prompts/rewrite_jmultiwoz_tsv_to_json_one_dialogue.txt}"
INPUT_ROOT="${INPUT_ROOT:-jmultiwoz-tsv-by-dialogue}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/jmultiwoz_rewritten_json}"
SPLITS="${SPLITS:-train validation test}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BASE_SEED="${BASE_SEED:-2000}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.55}"
TOP_P="${TOP_P:-0.9}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RESUME="${RESUME:-1}"
LIMIT="${LIMIT:-0}"
HF_HOME="${HF_HOME:-/mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3/.hf-cache}"

cd "$(dirname "$0")/.."

if [[ ! -x "$LOCAL_PYTHON" ]]; then
  echo "Local generation environment is not installed." >&2
  echo "Run: cd /mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3 && bash scripts/setup_local.sh" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

read -r -a SPLIT_ARGS <<< "$SPLITS"

export HF_HOME
exec "$LOCAL_PYTHON" rewrite_jmultiwoz_tsv_to_json_batch.py \
  --model "$MODEL" \
  --prompt-file "$PROMPT_FILE" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --splits "${SPLIT_ARGS[@]}" \
  --batch-size "$BATCH_SIZE" \
  --base-seed "$BASE_SEED" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-retries "$MAX_RETRIES" \
  --limit "$LIMIT" \
  "${RESUME_ARGS[@]}"

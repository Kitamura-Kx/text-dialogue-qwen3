#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/random_travel_dialogues.local.json}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
EXPECTED_COUNT="${EXPECTED_COUNT:-5}"
SEED="${SEED:-}"
HF_HOME="${HF_HOME:-$(pwd)/.hf-cache}"

SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=(--seed "$SEED")
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/text-dialogue-qwen3-uv-cache}" \
UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
HF_HOME="$HF_HOME" \
uv run --no-sync python -m qwen_dialogue \
  --backend local \
  --model "$MODEL" \
  --prompt-file prompts/random_travel_dialogues_json.txt \
  --output-file "$OUTPUT_FILE" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  "${SEED_ARGS[@]}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/text-dialogue-qwen3-uv-cache}" \
UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
HF_HOME="$HF_HOME" \
uv run --no-sync python -m normalize_dialogue_json "$OUTPUT_FILE"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/text-dialogue-qwen3-uv-cache}" \
UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
HF_HOME="$HF_HOME" \
uv run --no-sync python -m validate_dialogue_json "$OUTPUT_FILE" \
  --expected-count "$EXPECTED_COUNT"

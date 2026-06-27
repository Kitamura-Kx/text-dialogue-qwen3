#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
DTYPE="${DTYPE:-bfloat16}"
HF_HOME="${HF_HOME:-$(pwd)/.hf-cache}"
VLLM_BIN="${VLLM_BIN:-/tmp/text-dialogue-qwen3-vllm-venv/bin/vllm}"
export HF_HOME

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM is not installed. Run: bash scripts/setup_vllm.sh" >&2
  exit 1
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --dtype "$DTYPE"

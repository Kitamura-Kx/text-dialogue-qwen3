#!/usr/bin/env bash
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-/tmp/text-dialogue-qwen3-vllm-venv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/text-dialogue-qwen3-vllm-uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export UV_CACHE_DIR UV_LINK_MODE

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  uv venv "$VLLM_VENV" --python 3.12
fi
uv pip install \
  --python "$VLLM_VENV/bin/python" \
  --torch-backend cu128 \
  "vllm==0.10.2" \
  "transformers>=4.55.2,<5"

"$VLLM_VENV/bin/python" -c \
  "import torch, vllm; print('vllm', vllm.__version__); print('torch', torch.__version__); print('cuda', torch.version.cuda)"

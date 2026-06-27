#!/usr/bin/env bash
set -euo pipefail

LOCAL_VENV="${LOCAL_VENV:-/tmp/text-dialogue-qwen3-local-venv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/text-dialogue-qwen3-vllm-uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export UV_CACHE_DIR UV_LINK_MODE

if [[ ! -x "$LOCAL_VENV/bin/python" ]]; then
  uv venv "$LOCAL_VENV" --python 3.12
fi

uv pip install \
  --python "$LOCAL_VENV/bin/python" \
  --torch-backend cu128 \
  "torch==2.8.0" \
  "transformers>=4.55.2,<5" \
  "accelerate>=1.8.0" \
  "safetensors>=0.5.0"

"$LOCAL_VENV/bin/python" -c \
  "import torch, transformers; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('transformers', transformers.__version__)"

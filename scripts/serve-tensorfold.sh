#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: $0 /path/to/CYBER-FROST-Qwen3.8-Flash-Next-TensorFold-MLX-8bit" >&2
  exit 2
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "model directory not found: $MODEL_PATH" >&2
  exit 2
fi

TENSORFOLD_BIN="${TENSORFOLD_BIN:-/opt/homebrew/bin/tensorfold}"
if [[ ! -x "$TENSORFOLD_BIN" ]]; then
  echo "native TensorFold binary not found: $TENSORFOLD_BIN" >&2
  echo "install it with: brew install ashhart/tensorfold/tensorfold" >&2
  exit 2
fi

export FLASHNEXT_LOAD_LIMIT_GB="${FLASHNEXT_LOAD_LIMIT_GB:-240}"

exec "$TENSORFOLD_BIN" serve "$MODEL_PATH" \
  --context "${TENSORFOLD_CONTEXT:-100000}" \
  --max-tokens "${TENSORFOLD_MAX_TOKENS:-8192}" \
  --temperature "${TENSORFOLD_TEMPERATURE:-0}" \
  --thinking \
  --reasoning-effort "${TENSORFOLD_REASONING_EFFORT:-xhigh}" \
  --host "${TENSORFOLD_HOST:-127.0.0.1}" \
  --port "${TENSORFOLD_PORT:-18080}" \
  --name "${TENSORFOLD_NAME:-cyberfrost}" \
  --no-update-check

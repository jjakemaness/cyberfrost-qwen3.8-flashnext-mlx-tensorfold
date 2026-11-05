#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${CYBERFROST_BASE_URL:-http://127.0.0.1:18080/v1}"

curl --fail-with-body --no-buffer \
  --header 'Content-Type: application/json' \
  --request POST "$BASE_URL/chat/completions" \
  --data @- <<'JSON'
{
  "model": "cyberfrost",
  "messages": [
    {
      "role": "user",
      "content": "Reply with exactly: TensorFold is working."
    }
  ],
  "temperature": 0,
  "max_tokens": 32,
  "stream": true,
  "reasoning_effort": "xhigh"
}
JSON

#!/bin/bash

# Mixed Workload benchmark for P/D disaggregated NELSSA.
#
# Short requests are sent at a fixed QPS (a background stream) while long
# requests are injected periodically. The key metric is whether the short
# requests' TPOT / p99 degrade while a long request occupies the KV cache —
# the vLLM baseline should degrade; NELSSA (KV offload) should not.
#
# Run against a live run_pd_with_nelssa.sh stack:
#   NELSSA ON : bash run_pd_with_nelssa.sh          (then) bash test_mixed_workload.sh
#   baseline  : NELSSA_OFF=1 bash run_pd_with_nelssa.sh (then) bash test_mixed_workload.sh
#
# Env knobs:
#   SHORT_QPS         short request rate (default 2)
#   SHORT_TOKENS      short prompt length in words (default 500)
#   LONG_TOKENS       long prompt length in words (default 96000)
#   LONG_INTERVAL     seconds between long injections (default 10)
#   LONG_COUNT        number of long injections (default 3)
#   SHORT_MAX_TOKENS  short output tokens (default 50)
#   LONG_MAX_TOKENS   long output tokens (default 200)
#   MODEL             model path
#   PROXY_URL         proxy endpoint (default http://localhost:8300/v1/completions)
#   OUT               results JSON path (default mixed_workload_results.json)

set -e

SHORT_QPS="${SHORT_QPS:-2}"
SHORT_TOKENS="${SHORT_TOKENS:-500}"
LONG_TOKENS="${LONG_TOKENS:-96000}"
LONG_INTERVAL="${LONG_INTERVAL:-10}"
LONG_COUNT="${LONG_COUNT:-3}"
SHORT_MAX_TOKENS="${SHORT_MAX_TOKENS:-50}"
LONG_MAX_TOKENS="${LONG_MAX_TOKENS:-200}"
MODEL="${MODEL:-/mnt/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
PROXY_URL="${PROXY_URL:-http://localhost:8300/v1/completions}"
OUT="${OUT:-mixed_workload_results.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/test_mixed_workload.py" \
  --short-qps "$SHORT_QPS" \
  --short-tokens "$SHORT_TOKENS" \
  --long-tokens "$LONG_TOKENS" \
  --long-interval "$LONG_INTERVAL" \
  --long-count "$LONG_COUNT" \
  --short-max-tokens "$SHORT_MAX_TOKENS" \
  --long-max-tokens "$LONG_MAX_TOKENS" \
  --model "$MODEL" \
  --proxy "$PROXY_URL" \
  --out "$OUT"

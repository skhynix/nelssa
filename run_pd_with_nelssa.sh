#!/bin/bash

# P/D Disaggregated NELSSA. Prefill worker (GPU 6): CPU offloading + CPU
# attention. Decode worker (GPU 7): KV reception only. Proxy routes short reqs
# to both, long reqs trigger P-side CPU attention.
#
# GPU 6/7 sit on NUMA node 1 (odd cores 1,3,...,255). All taskset/OMP/EC/GOMP/
# membind/cpunodebind settings below use node-1 odd cores (65-127, +1 from the
# node-0 even layout). Re-derive for your topology.

set -e

# ---- Default configuration ----
MODEL="${MODEL:-/mnt/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8300

# ---- NELSSA configuration ----
NELSSA_FIRST_N=16
NELSSA_LAST_M=64
NELSSA_PROMPT_LENGTH_THRESHOLD=2048
NELSSA_CPU_ATTENTION_KV_FRACTION=1.0
# Retrieval budget: fraction of clusters probed for CPU attention. Paper
# default 0.018 (16 ARM cores @ 200 GB/s); tune via measured latency.
NELSSA_RETRIEVAL_BUDGET=0.018

# ---- NELSSA on/off toggle ----
# NELSSA_OFF=1: same topology, all --nelssa-* flags removed → vLLM baseline.
NELSSA_OFF="${NELSSA_OFF:-0}"
if [ "$NELSSA_OFF" = "1" ]; then
  _NELSSA_FLAGS=""
else
  _NELSSA_FLAGS="--nelssa-enabled \
--nelssa-first-n $NELSSA_FIRST_N \
--nelssa-last-m $NELSSA_LAST_M \
--nelssa-prompt-length-threshold $NELSSA_PROMPT_LENGTH_THRESHOLD \
--nelssa-enable-attention \
--nelssa-cpu-attention-kv-fraction $NELSSA_CPU_ATTENTION_KV_FRACTION \
--nelssa-retrieval-budget $NELSSA_RETRIEVAL_BUDGET \
--nelssa-enable-offloading \
--nelssa-max-num-long-requests 1"
fi

# NELSSA_CORES_16=1: preset for a 16-core / 16-worker experiment — sets the
# affinity, GOMP, pool and OMP thread counts to 16 together (the 8-core OMP
# set below + 8 cores borrowed from the EC pool). EC-PIN stays OFF. Must run
# BEFORE the 8-core defaults below so the preset wins. Individual vars still
# override if set explicitly.
if [ "${NELSSA_CORES_16:-0}" = "1" ]; then
  NELSSA_AFFINITY_CORES="${NELSSA_AFFINITY_CORES:-73,81,89,97,105,113,121,125,67,69,71,75,77,79,83,85}"
  NELSSA_GOMP_AFFINITY="${NELSSA_GOMP_AFFINITY-73,81,89,97,105,113,121,125,67,69,71,75,77,79,83,85}"
  NELSSA_POOL_THREADS="${NELSSA_POOL_THREADS:-16}"
  NELSSA_OMP_THREADS="${NELSSA_OMP_THREADS:-16}"
fi

# 8-core CPU-attention pin set (1:1 with the OMP/intra-op pool).
NELSSA_AFFINITY_CORES="${NELSSA_AFFINITY_CORES:-73,81,89,97,105,113,121,125}"
NELSSA_WORKER_CORES=""

# EngineCore core isolation (EC-PIN): pin the EngineCore process off the
# CPU-attention cores via NELSSA_ENGINE_CORES (vllm/v1/engine/core.py).
# OFF by default — contention is real but not a bottleneck; A/B showed no TPOT
# gain. Re-enable: NELSSA_EC_NCORES=4 (or =23). ${VAR-default} honors explicit "".
# NELSSA_EC_NCORES auto-builds the set from node-1 odd cores excluding RPC(65)
# and OMP(73,81,..,125); overrides NELSSA_ENGINE_CORES.
_NELSSA_EC_POOL=(67 69 71 75 77 79 83 85 87 91 93 95 99 101 103 107 109 111 115 117 119 123 127)
if [ -n "${NELSSA_EC_NCORES:-}" ]; then
  _ec_list=()
  for _i in $(seq 0 $((NELSSA_EC_NCORES - 1))); do
    _ec_list+=("${_NELSSA_EC_POOL[$_i]}")
  done
  NELSSA_ENGINE_CORES="$(IFS=,; echo "${_ec_list[*]}")"
fi
NELSSA_ENGINE_CORES="${NELSSA_ENGINE_CORES-}"

# CPU-attention torch intra-op / OMP pool size. Default 8 (unset→32 is a ~4x
# regression). Revisit for multi-Long batches. Override: NELSSA_POOL_THREADS=16
NELSSA_POOL_THREADS="${NELSSA_POOL_THREADS:-8}"

# ---- Nsight Systems profiling ----
# NSYS_MODE: off (default) | split (per-worker reports) | combined (one report,
# all children on one timeline). NSYS_PROFILE=1 is a legacy alias for split.
NSYS_MODE="${NSYS_MODE:-}"
if [ "$NSYS_MODE" = "" ]; then
  [ "${NSYS_PROFILE:-0}" = "1" ] && NSYS_MODE=split || NSYS_MODE=off
fi
NSYS_OUT_DIR="${NSYS_OUT_DIR:-./nsys}"
# nsys implies NELSSA_NVTX=1. Override with explicit NELSSA_NVTX=0.
if [ "$NSYS_MODE" != "off" ] && [ -z "${NELSSA_NVTX:-}" ]; then
  NELSSA_NVTX=1
  export NELSSA_NVTX
fi
NSYS_WORKERS="${NSYS_WORKERS:-both}"   # split-mode: both | prefill | decode

# Combined mode: re-exec under one outer nsys (guarded by _NELSSA_UNDER_NSYS).
# The outer nsys finalizes after bash exits, so cleanup() must NOT kill it.
if [ "$NSYS_MODE" = "combined" ] && [ "${_NELSSA_UNDER_NSYS:-0}" != "1" ]; then
  mkdir -p "$NSYS_OUT_DIR"
  _ts=$(date +%Y%m%d_%H%M%S)
  _nsys_out="${NSYS_OUT_DIR}/combined_pd_${_ts}"
  # --gpu-metrics-devices: nsys GPU id (`cuda-visible` matches CUDA_VISIBLE_DEVICES).
  # NSYS_GPU_METRICS=0 drops sampling. Trace: cuda,nvtx is enough; add osrt,python-gil only for OS/GIL work.
  _gpu_metrics=""
  if [ "${NSYS_GPU_METRICS:-1}" = "1" ]; then
    _gpu_metrics="--gpu-metrics-devices=${NSYS_GPU_METRICS_DEVICES:-cuda-visible}"
  fi
  _trace="${NSYS_TRACE:-cuda,nvtx,osrt}"
  echo -e "${YELLOW}Combined nsys: re-execing under nsys -> ${_nsys_out}.nsys-rep${NC}"
  exec nsys profile --trace=${_trace} \
    --trace-fork-before-exec=true --kill=none ${_gpu_metrics} \
    -o "${_nsys_out}" -- \
    env _NELSSA_UNDER_NSYS=1 NSYS_MODE=combined \
    NELSSA_NVTX="${NELSSA_NVTX:-1}" NELSSA_POOL_THREADS="${NELSSA_POOL_THREADS:-8}" \
    NELSSA_AFFINITY_CORES="${NELSSA_AFFINITY_CORES:-73}" \
    NELSSA_WORKER_CORES="${NELSSA_WORKER_CORES:-}" \
    NELSSA_OMP_THREADS="${NELSSA_OMP_THREADS:-8}" \
    NELSSA_GOMP_AFFINITY="${NELSSA_GOMP_AFFINITY:-73,81,89,97,105,113,121,125}" \
    NELSSA_ENGINE_CORES="${NELSSA_ENGINE_CORES-}" \
    NELSSA_EC_NCORES="${NELSSA_EC_NCORES-}" \
    NELSSA_ATTN_CORELOG="${NELSSA_ATTN_CORELOG:-0}" \
    NELSSA_OFF="${NELSSA_OFF:-0}" \
    NSYS_TRACE="${_trace}" \
    bash "$0" "$@"
fi

# Split-mode nsys prefix for worker $1, spliced as `$(_nsys_prefix prefill)numactl ...`.
# Empty when profiling is off or this worker isn't selected.
_nsys_prefix() {
  [ "$NSYS_MODE" = "split" ] || { echo ""; return; }
  case ",$NSYS_WORKERS," in
    *,"$1",*) ;; *,"both",*) ;; *) echo ""; return;;
  esac
  mkdir -p "$NSYS_OUT_DIR"
  local _ts
  _ts=$(date +%Y%m%d_%H%M%S)
  # --gpu-metrics-devices OPT-IN (NSYS_GPU_METRICS=1): can perturb fork CUPTI
  # injection. cuda-visible matches CUDA_VISIBLE_DEVICES.
  local _gpu_metrics=""
  if [ "${NSYS_GPU_METRICS:-0}" = "1" ]; then
    local _dev="${NSYS_GPU_METRICS_DEVICES:-cuda-visible}"
    _gpu_metrics="--gpu-metrics-devices=${_dev}"
  fi
  local _trace="${NSYS_TRACE:-cuda,nvtx,osrt}"
  echo "nsys profile --trace=${_trace} --trace-fork-before-exec=true --kill=none ${_gpu_metrics} -o ${NSYS_OUT_DIR}/${1}_${_ts} -- "
}

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Cleanup: re-entrant (INT/TERM). Targets ONLY our workers by serve port (works
# for both NELSSA and NELSSA_OFF=1 baseline) — not a bare pkill (shared box).
_CLEANUP_RUN=0
cleanup() {
    [ "$_CLEANUP_RUN" = "1" ] && return
    _CLEANUP_RUN=1
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ "$NSYS_MODE" != "off" ]; then
        # Signal workers so nsys finalizes. Do NOT SIGTERM nsys (loses capture).
        pkill -TERM -f -- "--port $PREFILL_PORT" 2>/dev/null || true
        pkill -TERM -f -- "--port $DECODE_PORT" 2>/dev/null || true
        pkill -TERM -f "toy_proxy_server.py" 2>/dev/null || true
        for _ in $(seq 1 60); do
            pgrep -f "nsys profile" >/dev/null 2>&1 || break
            sleep 1
        done
    fi
    pkill -9 -f -- "--port $PREFILL_PORT" 2>/dev/null || true
    pkill -9 -f -- "--port $DECODE_PORT" 2>/dev/null || true
    pkill -9 -f "toy_proxy_server.py" 2>/dev/null || true
    # Split mode only: kill a stuck inner nsys. Must NOT match outer combined nsys.
    if [ "$NSYS_MODE" = "split" ]; then
        pkill -9 -f "nsys profile" 2>/dev/null || true
    fi
    sleep 1
    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup INT TERM

# Wait for server to be ready
wait_for_server() {
    local port=$1
    local name=$2
    echo -e "${BLUE}Waiting for $name on port $port...${NC}"

    for i in {1..60}; do
        if curl -s http://localhost:${port}/health > /dev/null 2>&1; then
            echo -e "${GREEN}$name is ready${NC}"
            return 0
        fi
        sleep 2
    done

    echo -e "${RED}$name failed to start${NC}"
    return 1
}

# ---- Print configuration ----
echo -e "${BLUE}======================================${NC}"
if [ "$NELSSA_OFF" = "1" ]; then
  echo -e "${BLUE}P/D Disaggregated BASELINE (NELSSA OFF)${NC}"
else
  echo -e "${BLUE}P/D Disaggregated with NELSSA${NC}"
fi
echo -e "${BLUE}======================================${NC}"
echo -e "Model:           ${GREEN}$MODEL${NC}"
echo -e "Prefill Worker:  ${GREEN}GPU 6, port $PREFILL_PORT${NC}"
if [ "$NELSSA_OFF" = "1" ]; then
  echo -e "  - NELSSA: ${RED}DISABLED (baseline)${NC}"
else
  echo -e "  - NELSSA: ENABLED${NC}"
  echo -e "  - first_n: $NELSSA_FIRST_N${NC}"
  echo -e "  - last_m: $NELSSA_LAST_M${NC}"
  echo -e "  - threshold: $NELSSA_PROMPT_LENGTH_THRESHOLD tokens${NC}"
  echo -e "  - CPU Attention: ENABLED${NC}"
  echo -e "  - CPU KV Fraction: $NELSSA_CPU_ATTENTION_KV_FRACTION${NC}"
  echo -e "  - Retrieval Budget: $NELSSA_RETRIEVAL_BUDGET (NUMA node 1 pinned)${NC}"
  echo -e "  - CPU Attn Cores/Threads: $NELSSA_AFFINITY_CORES (pool=$NELSSA_POOL_THREADS, omp=$NELSSA_OMP_THREADS)${NC}"
  echo -e "  - EngineCore Isolation: ${NELSSA_ENGINE_CORES:-(disabled)} (ncores=${NELSSA_EC_NCORES:-default})${NC}"
fi
echo -e "Decode Worker:   ${GREEN}GPU 7, port $DECODE_PORT${NC}"
if [ "$NELSSA_OFF" = "1" ]; then
  echo -e "  - NELSSA: ${RED}DISABLED (baseline)${NC}"
else
  echo -e "  - NELSSA: ENABLED (same config for KV compatibility)${NC}"
fi
echo -e "Proxy Server:    ${GREEN}port $PROXY_PORT${NC}"
if [ "$NSYS_MODE" != "off" ]; then
  echo -e "nsys:            ${GREEN}mode=$NSYS_MODE NVTX=${NELSSA_NVTX:-0}${NC}"
fi
echo -e "${BLUE}======================================${NC}"
echo ""

# ---- Launch Prefill Worker (KV Producer, CPU attention) ----
# NUMA node 1 (--membind=1) so CPU attention and offloaded KV stay local.
if [ "$NELSSA_OFF" = "1" ]; then
  echo -e "${YELLOW}Starting Prefill Worker (GPU 6, NELSSA OFF baseline)...${NC}"
else
  echo -e "${YELLOW}Starting Prefill Worker (GPU 6) with NELSSA...${NC}"
fi
CUDA_VISIBLE_DEVICES=6 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
OMP_NUM_THREADS=${NELSSA_OMP_THREADS:-8} \
OMP_PROC_BIND=close \
MKL_NUM_THREADS=${NELSSA_POOL_THREADS:-8} \
GOMP_CPU_AFFINITY="${NELSSA_GOMP_AFFINITY-73,81,89,97,105,113,121,125}" \
NELSSA_AFFINITY_CORES="$NELSSA_AFFINITY_CORES" \
NELSSA_WORKER_CORES="$NELSSA_WORKER_CORES" \
NELSSA_POOL_THREADS="${NELSSA_POOL_THREADS:-8}" \
NELSSA_ATTN_CORELOG="${NELSSA_ATTN_CORELOG:-0}" \
NELSSA_ENGINE_CORES="$NELSSA_ENGINE_CORES" \
$(_nsys_prefix prefill)numactl --membind=1 taskset -c 65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127 \
vllm serve "$MODEL" \
  --port $PREFILL_PORT \
  --enforce-eager \
  --max-model-len 128000 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.7 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
  $_NELSSA_FLAGS 2>&1 | tee prefill_log.txt &
PREFILL_PID=$!

# ---- Launch Decode Worker (KV Consumer; CPU attn disabled in code) ----
if [ "$NELSSA_OFF" = "1" ]; then
  echo -e "${YELLOW}Starting Decode Worker (GPU 7, NELSSA OFF baseline)...${NC}"
else
  echo -e "${YELLOW}Starting Decode Worker (GPU 7) with NELSSA...${NC}"
fi
CUDA_VISIBLE_DEVICES=7 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
OMP_NUM_THREADS=64 \
MKL_NUM_THREADS=64 \
$(_nsys_prefix decode)numactl --cpunodebind=1 --membind=1 \
vllm serve "$MODEL" \
  --port $DECODE_PORT \
  --enforce-eager \
  --max-model-len 128000 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.7 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
  $_NELSSA_FLAGS 2>&1 | tee decode_log.txt &
DECODE_PID=$!

# Wait for workers to be ready
wait_for_server $PREFILL_PORT "Prefill Worker" || exit 1
wait_for_server $DECODE_PORT "Decode Worker" || exit 1

# ---- Launch Proxy Server ----
echo -e "${YELLOW}Starting Proxy Server...${NC}"
python3 tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port $PROXY_PORT \
  --prefiller-hosts localhost \
  --prefiller-ports $PREFILL_PORT \
  --decoder-hosts localhost \
  --decoder-ports $DECODE_PORT 2>&1 | tee proxy_log.txt &
PROXY_PID=$!

# Wait for Proxy to be ready
wait_for_server $PROXY_PORT "Proxy Server" || exit 1

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}All components are ready!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${YELLOW}Benchmarks (against this stack):${NC}"
echo "  TOKENS=96000 MAX_TOKENS=200 bash test_long_request.sh        # Single-Long"
echo "  NREQS=2 bash test_multi_long_batch.sh                       # Multi-Long"
echo "  bash test_mixed_workload.sh                                   # Mixed"
echo ""
if [ "$NELSSA_OFF" = "1" ]; then
  echo -e "${YELLOW}Baseline mode: NELSSA OFF — KV stays on GPU, no CPU attention.${NC}"
else
  echo -e "${YELLOW}NELSSA ON — check logs for [NELSSA] OFFLOAD / CPU ATTENTION activity.${NC}"
fi
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Keep running
wait $PREFILL_PID $DECODE_PID $PROXY_PID

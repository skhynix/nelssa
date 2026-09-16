# NELSSA — A GPU–PNM Heterogeneous System for Mixed-Length LLM Serving (CPU Emulation, vLLM Integration)

> **Note:** This README is an initial draft and will be revised.

This repository integrates **NELSSA** into [vLLM](https://github.com/vllm-project/vllm).
NELSSA is presented in the paper *"NELSSA: A GPU–PNM Heterogeneous System for
Mixed-Length LLM Serving via Length-based Request Placement"* (accepted to
**MICRO 2026**, [arXiv:2607.26633](https://arxiv.org/abs/2607.26633)).

## Background

Modern LLM serving workloads span context lengths from a few hundred to
hundreds of thousands of tokens, and these requests frequently interleave within
the same serving window — exposing fundamental inefficiencies in GPU-centric
serving architectures, whose throughput depends on large, memory-constrained
batches. NELSSA addresses this by integrating GPUs with real-world
**Processing-near-Memory (PNM)** accelerator devices: **length-based request
placement** routes short-context requests to GPUs and long-context requests to
the PNM tier, with runtime migration to accommodate dynamic context growth
without recomputation. The end-to-end prototype implements device-level sparse
attention on PNM, GPU decode kernels, and a host-side runtime that orchestrates
scheduling and cross-tier memory movement over a CXL-enabled infrastructure with
RPC and RDMA support. Across mixed-length LLM workloads, NELSSA improves
**decode throughput by up to 5.5× in tokens/sec** and reduces **P99 latency by
up to 15×** compared to GPU-only baselines.

## CPU Emulation on vLLM

PNM-based inference is still limited in practice: the device used in the paper
(a 16× ARM Neoverse V2 module with CXL 2.0 x16, 4-channel DDR5-6400, 200 GB/s
bandwidth and 512 GB capacity) is not widely deployable. This repository
therefore provides a **CPU-emulated** version of NELSSA, integrated into the
vLLM serving stack. The PNM tier is emulated with host-CPU attention over a
CPU-offloaded sparse KV cache, preserving the same length-based placement and
cross-tier movement design as the hardware prototype.

The sparse-attention core builds on the ideas of
[RetrievalAttention](https://arxiv.org/abs/2409.10516) and
[RetroInfer](https://arxiv.org/abs/2505.02922): the KV cache is treated as
vector storage, a lightweight **segmented clustering** algorithm builds an
attention-aware index, and a **retrieval budget** selects the most relevant KV
tokens per query. The vLLM integration adds:

- **CPU offloading** of the sparse KV cache to host memory (PCIe),
- **CPU attention** computed on the offloaded KV via a fused OpenMP GEMM,
- **Prefill/Decode (P/D) disaggregation** where the prefill worker offloads and
  runs CPU attention while the decode worker consumes KV transferred over
  [NIXL](https://github.com/ai-dynamo/nixl); CPU-attention requests and
  results are exchanged over an RPC KV channel.

The standalone reference implementation lives under [`nelssa_gpu/`](nelssa_gpu)
(the upstream RetroInfer codebase); the vLLM integration lives under
[`vllm/v1/nelssa/`](vllm/v1/nelssa) and the modified serving core.

> **Paper vs. this repo:** the paper reports results from the real GPU–PNM
> hardware prototype. The numbers achievable here are those of the CPU
> emulation path, not the hardware prototype.

---

## Architecture

For a long request, the request is routed by a **proxy server** to the
prefill worker; subsequent decode tokens are produced by the decode worker.
KV computed during prefill is transferred to the decode worker so decode
can continue without recomputation.

<!-- Architecture diagram: to be replaced with the NELSSA paper figure. -->

Key components:

- **`KVCacheClusteringEngine`** (`clustering.py`): k-means clusters the GPU KV
  buffers and stores centroid metadata; reorganizes the CPU KV buffers into a
  cluster-contiguous layout via the OpenMP C++ extension `vllm_clustering_ext`
  (`clustering_cpu.cpp`) — a pure-PyTorch fallback exists (`NELSSA_USE_CPP_REORG=0`).
- **`SimilaritySearchEngine`** (`similarity_search.py`): centroid-based
  retrieval. Given query states, returns `cluster_ids` / `topk_indices` /
  `topk_values` per request, using the per-layer prestacked centroids. Has a
  C++ fast-path and a batched torch fallback.
- **`CPUAttentionEngine` / `RPCAttentionEngine`** (`cpu_attention.py`): computes
  attention over the offloaded CPU KV cache. `CPUAttentionEngine` is for
  single-GPU mode; `RPCAttentionEngine` is the P/D path where the **prefill**
  worker asks the **decode** worker (which holds the offloaded KV) to compute
  CPU attention and return `(output, lse)`.
- **`AttentionResultMerger`** (`cpu_attention.py`): merges the GPU-side
  (head + tail tokens) attention output with the CPU-side attention result via a
  numerically-stable **LSE (log-sum-exp) merge**, implemented as the fused CUDA
  kernel `vllm_merge_ext` (`merge_kernel.cu`) and replayed through a CUDA graph.

Cross-worker KV transfer:

- **NIXL** (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`):
  `NixlConnector` in `kv_producer` (prefill) / `kv_consumer` (decode) roles moves
  the GPU KV cache to the decode worker over a side channel.
- **RPC KV** (`vllm/distributed/rpc_kv/`): `RPCKVServer` (decode side) /
  `RPCKVClient` (prefill side) exchange sparse-attention
  `SparseAttentionRequest`/`Response` and `KVStoreNotification`/`CleanupRequest`
  messages (see `protocol.py`); tensors are serialized compactly
  (`tensor_to_bytes`/`bytes_to_tensor`).

Integration points in the serving core:

- `vllm/model_executor/models/llama.py` — per-layer callbacks
  (`_sparse_kv_offload_callback`, `_similarity_search_callback`,
  `_cpu_attention_callback`) and the LSE-merge CUDA-graph replay
  (`_nelssa_merge_graph_replay`).
- `vllm/v1/worker/gpu_model_runner.py` — wires the NELSSA config, the CPU
  attention thread-pool, offload streams, and the `[P]`/`[D]` role tags.
- `vllm/v1/engine/core.py` — `NELSSA_ENGINE_CORES` EngineCore core isolation
  (`os.sched_setaffinity`) to keep the engine process off the CPU-attention cores.

---

## Repository Layout

```
.
├── vllm/                          # vLLM serving core (NELSSA-integrated)
│   └── v1/nelssa/
│       ├── clustering.py          # KVCacheClusteringEngine
│       ├── clustering_cpu.cpp     # OpenMP CPU reorg kernel → vllm_clustering_ext
│       ├── similarity_search.py    # SimilaritySearchEngine
│       ├── cpu_attention.py       # CPUAttentionEngine / RPCAttentionEngine / merger
│       ├── merge_kernel.cu         # fused LSE-merge CUDA kernel → vllm_merge_ext
│       ├── setup_clustering_ext.py
│       ├── setup_merge_ext.py
│       └── build_ext.sh
├── vllm/distributed/rpc_kv/       # RPC KV channel (P↔D sparse-attn / cleanup)
├── vllm/distributed/kv_transfer/kv_connector/v1/nixl/   # NIXL KV transfer
├── nelssa_gpu/                    # standalone RetroInfer reference implementation
├── environment_info.txt           # reference hardware/software baseline
├── environment_requirements.txt    # frozen pip requirement list
└── ...                            # vLLM build files, tests, docs, benchmarks
```

---

## Environment Requirements

NELSSA's P/D disaggregation places CPU attention on the prefill worker's host
CPU, so it needs a multi-GPU server with a multi-socket NUMA CPU. Minimum:

- **GPUs**: 2 — one for the prefill worker (KV producer + CPU attention), one
  for the decode worker (KV consumer). They should sit on the **same NUMA
  node** so the prefill worker's CPU attention and the GPU KV cache stay
  memory-local (no cross-socket hop).
- **CPU**: a NUMA node with enough free cores for the prefill worker to run
  the EngineCore (GPU forward) **and** a dedicated CPU-attention OMP pool
  (default 8 cores) without oversubscribing. In practice ~16-32 physical cores
  on the GPU's NUMA node are a comfortable floor; fewer works but CPU attention
  will contend with the engine.
- **Host memory**: enough to hold the CPU-offloaded sparse KV cache for the
  long requests you serve (scales with prompt length × layers × KV heads ×
  `retrieval_budget`; 100+ GB is typical for ~100K-token contexts).
- **Software**: Python 3.12, a CUDA 12.8-class toolchain, and a vLLM-compatible
  PyTorch (the integration was developed with torch 2.11+cu128).

The reference setup used to develop this integration is captured in
`environment_info.txt`, and a frozen list of all pinned packages is in
`environment_requirements.txt`.


---

## Build

NELSSA's C++/CUDA extensions are **built separately** from the main vLLM
install — `pip install -e .` does not rebuild them.

### 1. vLLM core

```bash
uv venv --python 3.12
source .venv/bin/activate
# Python-only changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
# (or, with C/C++ changes: uv pip install -e . --torch-backend=auto)
```

### 2. NELSSA extensions

```bash
cd vllm/v1/nelssa

# OpenMP CPU reorg kernel → vllm_clustering_ext.so
python setup_clustering_ext.py build_ext --inplace
#   (or: bash build_ext.sh)

# Fused LSE-merge CUDA kernel → vllm_merge_ext.so (pinned to sm_90 by default)
python setup_merge_ext.py build_ext --inplace
```

`vllm_clustering_ext` compiles `clustering_cpu.cpp` with `-fopenmp -O3 -march=native`.
`vllm_merge_ext` compiles `merge_kernel.cu` with `-O3 -arch=sm_90 --use_fast_math`
(adjust `NVCC_FLAGS` in `setup_merge_ext.py` for a different GPU arch).

---

## Run

`run_pd_with_nelssa.sh` launches a 3-process P/D disaggregated setup:

| Process        | GPU   | Role                          | Port |
|----------------|-------|-------------------------------|------|
| Prefill worker | GPU A | KV producer + CPU attention (Host CPU)   | 8100 |
| Decode worker  | GPU B | KV consumer (KV reception)    | 8200 |
| Proxy server   | —     | routes short/long requests     | 8300 |

The two workers should be placed on GPUs that share a NUMA node, and the
prefill worker is pinned to that NUMA node (`numactl --membind`/`--cpunodebind`)
with a `taskset` covering a core range that excludes a dedicated subset for the
CPU-attention OMP workers. See `run_pd_with_nelssa.sh` for the concrete
defaults; the GPU indices (`CUDA_VISIBLE_DEVICES`), NUMA node, and core ranges
all need to be re-derived for your machine (see Environment Requirements).

```bash
# Start the P/D disaggregated NELSSA stack
bash run_pd_with_nelssa.sh

# In another terminal, send a long request (~100K tokens) and measure TTFT / decode TPOT:
TOKENS=100000 MAX_TOKENS=200 bash test_long_request.sh
```

The model path defaults to a Llama-3.1-8B-Instruct snapshot; override with
`MODEL=...`. Currently only **Llama-3.1-8B-Instruct** is supported; other
models will follow.

---

## Benchmarks

All benchmarks hit the proxy of a running `run_pd_with_nelssa.sh` stack and stream
responses. Compare NELSSA vs. the vLLM baseline by running each benchmark once
against a NELSSA-enabled server and once against a baseline server
(`NELSSA_OFF=1 bash run_pd_with_nelssa.sh`).

- **`test_long_request.sh`** — Single-Long. One long (~100K token) prompt through
  the proxy; reports streaming TTFT, decode-only TPOT
  , and decode throughput. Override with `TOKENS=` and
  `MAX_TOKENS=`.
- **`test_multi_long_batch.sh`** — Multi-Long. N concurrent long requests of
  identical length, with an aggregated batch throughput.
- **`test_mixed_workload.sh`** — Mixed Workload. Short requests sent at a fixed
  QPS with periodic long-request injection. The key metric is whether the short
  requests' TPOT / p99 degrade while a long request occupies the KV cache.

---

## Configuration Reference

### Environment variables 

| Variable | Meaning |
|----------|---------|
| `NELSSA_OFF` | `=1` launches the same P/D topology with vLLM baseline for comparison. |
| `NELSSA_POOL_THREADS` | CPU-attention torch intra-op pool size (default 8). |
| `NELSSA_AFFINITY_CORES` | 8-core CPU-attention pin set (NUMA-node cores on your box). |
| `NELSSA_GOMP_AFFINITY` | `GOMP_CPU_AFFINITY` for OMP workers. |

---

## License

This project is a fork of [vLLM](https://github.com/vllm-project/vllm), which is
licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE)). The
`nelssa_gpu/` directory contains the upstream RetroInfer codebase (see
[`nelssa_gpu/LICENSE`](nelssa_gpu/LICENSE)). Modifications in this fork are
released under the same Apache 2.0 license.

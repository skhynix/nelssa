# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU Attention for NELSSA.

1. CPUAttentionEngine  — single-GPU mode (offloaded KV cache, batched GEMM)
2. RPCAttentionEngine  — P/D disaggregated mode (Decode worker RPC -> Prefill CPU)
3. AttentionResultMerger — LSE-based merge of GPU + CPU attention outputs
"""

import os
import math
import time
import nvtx
import torch
import torch.nn.functional as F
from typing import Any

from vllm.distributed.rpc_kv.protocol import bytes_to_tensor
from vllm.logger import init_logger

logger = init_logger(__name__)

# C++ gather/fused-attention extension. Reorg makes KV cluster-contiguous,
# so each cluster is gathered with one memcpy (Python index-prep was ~12ms/layer).
_GATHER_EXT = None
try:
    if os.environ.get("NELSSA_USE_CPP_GATHER", "1") == "1":
        from vllm.v1.nelssa.clustering import _CLUSTERING_EXT as _GATHER_EXT
except Exception:
    _GATHER_EXT = None


def _buffer_total_tokens(buf: dict) -> int:
    off = buf.get('offset')
    return off.item() if isinstance(off, torch.Tensor) else off


# ---------------------------------------------------------------------------
# Scratch-buffer pools (reused across layers to avoid per-layer allocation)
# ---------------------------------------------------------------------------
_GATHER_BUF_POOL: dict[tuple[int, int], list[torch.Tensor]] = {}                        # Gather KV Cache
_FUSED_OUT_POOL: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}     # Fused Kernel Output
# Batched gather output buffers [N, kvH, M_CAP, D] for the P/D fused path.
# Keyed by (kvH, head_dim, max_reqs). M_CAP is a FIXED bound so the buffer is
# allocated once and reused across layers — collapses per-layer M to one shape,
# so the oneDNN matmul primitive cache hits every layer (no primitive thrash).
_GATHER_BATCH_POOL: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

# Fixed M upper bound for the gather batch buffer. 2304 covers the real per-layer
# M range with ~10% head, 128-aligned. The gather writes only [0, lens) per head;
# the tail [lens, M_CAP) must stay 0 (zero-initialized once) so the fused AV
# matmul's 0 * 0 = 0 is NaN-free (0 * garbage could be NaN). cpu_attention_fused
# masks the tail to -inf in softmax, and the leftover 0 tail is harmless in AV.
M_CAP_DEFAULT = 2304

# When 1 (default), M_CAP is a FIXED bound: the gather buffer is allocated once
# at M_CAP and reused every layer (no per-layer grow/slice). When 0, the buffer
# grows to the per-layer max_tokens (the pre-M_cap behavior) so the buffer
# exactly fits each layer's real M — the baseline used to measure M_CAP's
# effect. Override via NELSSA_FIXED_M_CAP=0.
_FIXED_M_CAP = os.environ.get("NELSSA_FIXED_M_CAP", "1") != "0"

# Verbose NELSSA debug logging (per-step CPU-ATTN/AFFINITY logs, /proc thread
# walk, timing aggregation). OFF by default = zero overhead on the hot path.
# Enable with NELSSA_ATTN_CORELOG=1 (same flag the C++ CORELOG probes use).
_CORELOG = os.environ.get("NELSSA_ATTN_CORELOG", "0") == "1"


def _get_gather_batch_buffers(
    num_reqs: int, num_kv_heads: int, head_dim: int, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pooled (keys_batch, values_batch) [N, kvH, M, D] bf16.

    FIXED-M_CAP mode (default): allocated ONCE with M_CAP (zero-init) and reused
    across layers — no per-layer reallocation or slicing. ``max_tokens`` is
    ignored. Baseline mode (NELSSA_FIXED_M_CAP=0): sized to the per-layer
    ``max_tokens`` and re-grown when a later layer needs more — the original
    pre-M_cap buffer behavior.
    """
    key = (num_kv_heads, head_dim, max(num_reqs, 1))
    cached = _GATHER_BATCH_POOL.get(key)
    if _FIXED_M_CAP:
        if cached is not None:
            return cached
        m_cap = int(os.environ.get("NELSSA_M_CAP", M_CAP_DEFAULT))
        keys_batch = torch.zeros(
            (max(num_reqs, 1), num_kv_heads, m_cap, head_dim), dtype=torch.bfloat16)
        values_batch = torch.zeros(
            (max(num_reqs, 1), num_kv_heads, m_cap, head_dim), dtype=torch.bfloat16)
        _GATHER_BATCH_POOL[key] = (keys_batch, values_batch)
        return keys_batch, values_batch
    # Baseline grow path: reuse if big enough, else reallocate to max_tokens.
    if cached is not None and cached[0].shape[2] >= max_tokens:
        return cached
    keys_batch = torch.zeros(
        (max(num_reqs, 1), num_kv_heads, max(max_tokens, 1), head_dim),
        dtype=torch.bfloat16)
    values_batch = torch.zeros(
        (max(num_reqs, 1), num_kv_heads, max(max_tokens, 1), head_dim),
        dtype=torch.bfloat16)
    _GATHER_BATCH_POOL[key] = (keys_batch, values_batch)
    return keys_batch, values_batch


def _get_gather_buffers(num_kv_heads: int, head_dim: int,
                        max_selected: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pooled (keys_dst, values_dst) [kvH, max_selected, D], grown if needed."""
    key = (num_kv_heads, head_dim)
    pool = _GATHER_BUF_POOL.setdefault(key, [])
    if pool:
        k_dst, v_dst = pool.pop()
        if k_dst.shape[1] >= max_selected:
            return k_dst, v_dst
    k_dst = torch.zeros((num_kv_heads, max_selected, head_dim), dtype=torch.bfloat16)
    v_dst = torch.zeros((num_kv_heads, max_selected, head_dim), dtype=torch.bfloat16)
    return k_dst, v_dst


def _return_gather_buffers(k_dst: torch.Tensor, v_dst: torch.Tensor) -> None:
    pool = _GATHER_BUF_POOL.setdefault((k_dst.shape[0], k_dst.shape[2]), [])
    if len(pool) < 32:
        pool.append((k_dst, v_dst))


def _get_fused_outputs(num_reqs: int, num_heads: int, head_dim: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (output[N,H,D] bf16, lse[N,H,1] f32) for the fused kernel."""
    key = (num_reqs, num_heads, head_dim)
    cached = _FUSED_OUT_POOL.get(key)
    if cached is not None:
        return cached
    out = torch.empty((num_reqs, num_heads, head_dim), dtype=torch.bfloat16)
    lse = torch.empty((num_reqs, num_heads, 1), dtype=torch.float32)
    _FUSED_OUT_POOL[key] = (out, lse)
    return out, lse


# ---------------------------------------------------------------------------
# Sparse gather helpers (RetrievalAttention-style per-kv-head selection)
# ---------------------------------------------------------------------------

def _normalize_cluster_ids(cluster_ids, num_kv_heads: int, device) -> torch.Tensor:
    """Normalize cluster_ids to a [num_kv_heads, max_nprobe] long tensor.

    Accepts a tensor or per-kv-head list-of-lists (possibly ragged); ragged rows
    are right-padded with 0.
    """
    if isinstance(cluster_ids, torch.Tensor):
        ids = cluster_ids.to(device=device, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0).expand(num_kv_heads, -1).contiguous()
        return ids
    if not cluster_ids:
        return torch.zeros(num_kv_heads, 1, dtype=torch.long, device=device)
    # Fast path: rectangular [H][P] list (common P/D case) — one tensor creation.
    if (isinstance(cluster_ids[0], (list, tuple))
            and len(cluster_ids) == num_kv_heads
            and all(len(r) == len(cluster_ids[0]) for r in cluster_ids)):
        return torch.tensor(cluster_ids, dtype=torch.long, device=device)
    rows = cluster_ids if isinstance(cluster_ids[0], (list, tuple)) else [cluster_ids]
    max_nprobe = max((len(r) for r in rows), default=1)
    padded = torch.zeros(num_kv_heads, max_nprobe, dtype=torch.long, device=device)
    for h, r in enumerate(rows[:num_kv_heads]):
        if r:
            padded[h, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
    return padded


def _gather_selected_clusters(
    keys: torch.Tensor,             # [num_kv_heads, num_tokens, head_dim]
    values: torch.Tensor,           # [num_kv_heads, num_tokens, head_dim]
    cluster_ids,                    # [num_kv_heads, nprobe]
    cluster_offsets: torch.Tensor,  # [num_kv_heads, n_centroids]
    cluster_size: torch.Tensor,     # [num_kv_heads, n_centroids]
    reuse_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_kv_heads, num_tokens, head_dim = keys.shape
    device = keys.device

    ids_t = _normalize_cluster_ids(cluster_ids, num_kv_heads, device)  # [H, P]
    nprobe = ids_t.shape[1]

    cpp_ready = (_GATHER_EXT is not None
                 and keys.device.type == 'cpu'
                 and cluster_offsets.device.type == 'cpu'
                 and cluster_size.device.type == 'cpu'
                 and cluster_offsets.dtype == torch.int32
                 and cluster_size.dtype == torch.int32)

    nvtx.push_range("[P] KV_gather")
    if cpp_ready:
        starts = torch.gather(cluster_offsets, 1, ids_t)  # [H, P]
        sizes = torch.gather(cluster_size, 1, ids_t)      # [H, P]
        per_head_lens = sizes.sum(dim=1)                   # [H]
        # max_selected = largest per-head total (heads can pick different clusters).
        max_selected = max(int(per_head_lens.max().item())
                           if per_head_lens.numel() else 1, 1)

        if (reuse_buffers is not None
                and reuse_buffers[0].shape[1] >= max_selected
                and reuse_buffers[0].device.type == device.type):
            keys_dst, values_dst = reuse_buffers
        else:
            keys_dst = torch.zeros((num_kv_heads, max_selected, head_dim),
                                   dtype=keys.dtype, device=device)
            values_dst = torch.zeros((num_kv_heads, max_selected, head_dim),
                                     dtype=values.dtype, device=device)
        lens_out = torch.zeros(num_kv_heads, dtype=torch.int32, device=device)

        _GATHER_EXT.gather_selected_clusters_cpu(
            keys_dst, values_dst,
            keys, values,
            cluster_offsets, cluster_size, ids_t,
            lens_out, num_kv_heads, nprobe,
        )
        nvtx.pop_range()
        return keys_dst, values_dst, per_head_lens

    result = _gather_selected_clusters_torch(
        keys, values, ids_t, cluster_offsets, cluster_size,
        num_kv_heads, num_tokens, head_dim, device)
    nvtx.pop_range()
    return result


def _gather_selected_clusters_torch(
    keys: torch.Tensor,
    values: torch.Tensor,
    ids_t: torch.Tensor,
    cluster_offsets: torch.Tensor,
    cluster_size: torch.Tensor,
    num_kv_heads: int,
    num_tokens: int,
    head_dim: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original vectorized torch gather (fallback + C++ verification path)."""
    starts = torch.gather(cluster_offsets, 1, ids_t)  # [H, P]
    sizes = torch.gather(cluster_size, 1, ids_t)      # [H, P]
    per_head_lens = sizes.sum(dim=1)                   # [H] — caller's mask

    P = ids_t.shape[1]
    C = max(int(sizes.max().item()) if sizes.numel() else 1, 1)
    max_selected = max(int(per_head_lens.max().item())
                       if per_head_lens.numel() else 1, 1)

    # Compact valid slots to the front of each row. A valid entry's rank
    # (cumsum of valid flags) is its target column; invalid slots keep idx 0.
    arange_c = torch.arange(C, device=device)
    pos = (starts.unsqueeze(-1) + arange_c).reshape(num_kv_heads, P * C)
    valid = (arange_c < sizes.unsqueeze(-1)).reshape(num_kv_heads, P * C)

    rank = valid.int().cumsum(dim=1) - 1                 # [H, P*C]
    idx = torch.zeros(num_kv_heads, max_selected, dtype=torch.long, device=device)
    idx.scatter_(1, rank.clamp(min=0), pos)

    # Advanced indexing beats expand+torch.gather (0-stride last-dim view is
    # ~50x slower at this shape).
    ah = torch.arange(num_kv_heads, device=device).unsqueeze(1).expand(
        num_kv_heads, max_selected)
    gathered_k = keys[ah, idx]
    gathered_v = values[ah, idx]

    return gathered_k, gathered_v, per_head_lens


def _select_kv_sparse(
    offload_keys: torch.Tensor,     # [num_kv_heads, num_tokens, head_dim]
    offload_values: torch.Tensor,
    topk_ids,
    meta: dict | None,
    layer_idx: int,
    buffer_key: str,
    cpu_attention_kv_fraction: float,
    num_valid_tokens: int | None = None,
    return_lens: bool = False,
    reuse_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Select KV via sparse cluster gather, falling back to a leading fraction
    when metadata or cluster_ids are unavailable.

    Returns (keys, values, mask); with ``return_lens`` also per_head_lens for
    the fused kernel. ``reuse_buffers`` is forwarded to the C++ gather.
    """
    num_kv_heads = offload_keys.shape[0]
    num_offload_tokens = (num_valid_tokens if num_valid_tokens and num_valid_tokens > 0
                          else offload_keys.shape[1])

    reorganized = meta is not None and meta.get('reorganized', False)
    if isinstance(topk_ids, torch.Tensor):
        has_cluster_ids = topk_ids.numel() > 0
    elif topk_ids and isinstance(topk_ids[0], (list, tuple)):
        has_cluster_ids = any(len(h) > 0 for h in topk_ids)
    else:
        has_cluster_ids = bool(topk_ids)

    use_sparse = (reorganized and has_cluster_ids
                  and meta.get('cluster_offsets') is not None)

    per_head_lens = None
    if use_sparse:
        # Use the CPU copy prepared at reorg to avoid a per-layer GPU->CPU copy.
        # Use explicit `is not None` (Python `or` calls bool() on a multi-element
        # tensor, which raises). Ensure a CPU tensor: the C++ gather reads it via
        # a raw pointer, and P/D decode may store only a GPU cluster_size.
        _csc = meta.get('cluster_size_cpu')
        if _csc is not None:
            cluster_size_cpu = _csc
        else:
            _cs = meta['cluster_size']
            cluster_size_cpu = _cs if not _cs.is_cuda else _cs.cpu().contiguous()
        keys_cpu, values_cpu, per_head_lens = _gather_selected_clusters(
            offload_keys, offload_values, topk_ids,
            meta['cluster_offsets'], cluster_size_cpu,
            reuse_buffers=reuse_buffers)
        max_tokens = keys_cpu.shape[1]
        mask = (torch.arange(max_tokens, device=keys_cpu.device).unsqueeze(0)
                < per_head_lens.unsqueeze(1)).to(torch.float32)
    else:
        n = max(1, int(num_offload_tokens * cpu_attention_kv_fraction))
        keys_cpu = offload_keys[:, :n, :]
        values_cpu = offload_values[:, :n, :]
        mask = torch.ones(num_kv_heads, n, dtype=torch.float32)
        per_head_lens = torch.full((num_kv_heads,), n, dtype=torch.long)

    if return_lens:
        return keys_cpu, values_cpu, mask, per_head_lens
    return keys_cpu, values_cpu, mask


# ---------------------------------------------------------------------------
# Core batched GQA attention (shared by single-GPU and P/D paths)
# ---------------------------------------------------------------------------

def _batched_gqa(
    queries: torch.Tensor,      # [num_reqs, num_heads, head_dim]
    keys_batch: torch.Tensor,   # [num_reqs, num_kv_heads, max_tokens, head_dim]
    values_batch: torch.Tensor, # [num_reqs, num_kv_heads, max_tokens, head_dim]
    masks: torch.Tensor,        # [num_reqs, num_kv_heads, max_tokens]
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GQA attention with per-kv-head masks.

    Returns (output[N, num_heads, D], lse[N, num_heads, 1]).
    """
    num_reqs, num_heads, head_dim = queries.shape
    num_kv_heads = keys_batch.shape[1]
    group_size = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    q = queries.view(num_reqs, num_kv_heads, group_size, head_dim)
    scores = torch.matmul(q, keys_batch.transpose(-1, -2)) * scale  # [N,H,g,M]

    mask_expanded = masks.unsqueeze(2).bool()                       # [N,H,1,M]
    scores = scores.masked_fill(~mask_expanded, float('-inf'))
    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    scores_exp = torch.exp(scores - scores_max).masked_fill(~mask_expanded, 0.0)
    scores_sum = scores_exp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    attn_weights = scores_exp / scores_sum
    lse = scores_max + torch.log(scores_sum)

    output = torch.matmul(attn_weights, values_batch)               # [N,H,g,D]
    return output.view(num_reqs, num_heads, head_dim).to(out_dtype), \
        lse.view(num_reqs, num_heads, 1)


def _batched_attention(
    queries: torch.Tensor,      # [num_reqs, num_heads, head_dim]
    keys_batch: torch.Tensor,   # [num_reqs, num_kv_heads, max_tokens, head_dim]
    values_batch: torch.Tensor, # [num_reqs, num_kv_heads, max_tokens, head_dim]
    masks: torch.Tensor,        # [num_reqs, num_kv_heads, max_tokens]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-GPU batched GQA attention (default dtype path)."""
    return _batched_gqa(queries, keys_batch, values_batch, masks,
                        out_dtype=queries.dtype)


def _batched_attention_f32(
    queries: torch.Tensor,      # [num_reqs, num_heads, head_dim]
    keys_batch: torch.Tensor,   # [num_reqs, num_kv_heads, max_tokens, head_dim]
    values_batch: torch.Tensor, # [num_reqs, num_kv_heads, max_tokens, head_dim]
    masks: torch.Tensor,        # [num_reqs, num_kv_heads, max_tokens]
) -> tuple[torch.Tensor, torch.Tensor]:
    """P/D hot-path batched GQA attention (bf16 output).

    CPU bf16 matmul upcasts to fp32 (no AVX512_BF16 GEMM), so bf16 is
    numerically equivalent to fp32 for the GEMM while halving bandwidth.
    """
    return _batched_gqa(queries, keys_batch, values_batch, masks,
                        out_dtype=torch.bfloat16)


def _pad_to_batch(
    per_req_kv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad per-request (keys, values, mask) into a batched
    [num_reqs, num_kv_heads, max_tokens, ...] tensor.
    """
    # Single-request fast path: just add the batch dim (F.pad+stack is ~1-2ms/layer).
    if len(per_req_kv) == 1:
        keys, values, mask = per_req_kv[0]
        return (keys.unsqueeze(0), values.unsqueeze(0), mask.unsqueeze(0))

    max_tokens = max(k.shape[1] for k, _, _ in per_req_kv)

    keys_list, values_list, masks_list = [], [], []
    for keys, values, mask in per_req_kv:
        pad = max_tokens - keys.shape[1]
        if pad > 0:
            keys = F.pad(keys, (0, 0, 0, pad))
            values = F.pad(values, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad))
        keys_list.append(keys)
        values_list.append(values)
        masks_list.append(mask)

    return (torch.stack(keys_list, dim=0),
            torch.stack(values_list, dim=0),
            torch.stack(masks_list, dim=0))


# ---------------------------------------------------------------------------
# Single-GPU engine
# ---------------------------------------------------------------------------

class CPUAttentionEngine:
    """CPU attention for single-GPU mode (batched GEMM on offloaded KV cache)."""

    @staticmethod
    def compute_batch(
        cpu_attention_data: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute CPU attention for (req_id, query, keys, values) tuples.

        Returns (output[N, num_heads, D], lse[N, num_heads, 1]) or None.
        """
        if not cpu_attention_data:
            return None

        queries = torch.stack([d[1] for d in cpu_attention_data], dim=0)
        per_req_kv = [(keys, values, torch.ones(keys.shape[0], keys.shape[1], dtype=torch.float32))
                      for _, _, keys, values in cpu_attention_data]
        keys_batch, values_batch, masks = _pad_to_batch(per_req_kv)
        return _batched_attention(queries, keys_batch, values_batch, masks)


# ---------------------------------------------------------------------------
# P/D RPC engine
# ---------------------------------------------------------------------------

class RPCAttentionEngine:
    """RPC-based batch CPU attention for P/D disaggregated mode"""

    # Class-level collector for per-stage server-side timing (deserialize_q /
    # deserialize_ids / dict_loop / buf_prep / gather / fused). Set by
    # gpu_model_runner to attribute the srv_handler overhead.
    _server_stage_collector: list | None = None

    @staticmethod
    async def handle_batch(
        batch_request: Any,
        sparse_kv_cpu_buffers: dict,
        layer_idx: int,
        request_id_to_buffer_key: dict | None = None,
        cpu_attention_kv_fraction: float = 0.1,
        nelssa_cluster_metadata: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import time as _time
        import os as _os
        _st = {}
        _t0 = _time.perf_counter()
        query_tensor = bytes_to_tensor(batch_request.query_data)    # [N, num_heads, D]
        request_ids = batch_request.request_ids
        cluster_ids_list = batch_request.cluster_ids_list
        num_reqs = len(request_ids)
        if num_reqs == 0:
            return [], []
        _st['deserialize_q'] = (_time.perf_counter() - _t0) * 1e3

        # Transfer selected cluster_ids with 'Raw-bytes' path
        _t0 = _time.perf_counter()
        if cluster_ids_list and isinstance(cluster_ids_list[0], (bytes, bytearray)):
            cluster_ids_list = [bytes_to_tensor(c) for c in cluster_ids_list]
        _st['deserialize_ids'] = (_time.perf_counter() - _t0) * 1e3

        # Thread/affinity capping is done once at RPC-server startup, so the
        # hot path pays no set_num_threads/sched_setaffinity per layer.

        # Fused C++ path (default) vs legacy Python 3-op path.
        _legacy = bool(os.environ.get("_NELSSA_LEGACY_CPU_ATTN", ""))
        _fused_available = (_GATHER_EXT is not None
                            and hasattr(_GATHER_EXT, "cpu_attention_fused"))
        # Batched gather: one C++ dispatch gathers all N reqs in parallel
        # (OpenMP across N*kvH) instead of a Python for-loop of N dispatches.
        _batch_available = (_GATHER_EXT is not None
                            and hasattr(_GATHER_EXT, "gather_selected_clusters_batch_cpu")
                            and hasattr(_GATHER_EXT, "cpu_attention_fused"))

        per_req_kv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        per_req_lens: list[torch.Tensor] = []  # fused path only
        _checked_out: list[tuple[torch.Tensor, torch.Tensor]] = []  # for pool return
        result = None

        try:
            nvtx.push_range("[P] CPU_attention_batch")

            # ---- Batched fused path: gather all N reqs in one C++ dispatch ----
            # Check whether Fused Gather is available
            all_fused = (not _legacy and _batch_available and num_reqs > 0)
            if all_fused:
                _t0 = _time.perf_counter()
                keys_src_list: list[torch.Tensor] = []
                values_src_list: list[torch.Tensor] = []
                offsets_list: list[torch.Tensor] = []
                sizes_list: list[torch.Tensor] = []
                ids_list: list[torch.Tensor] = []
                num_tokens_list: list[int] = []  # valid tokens per req (debug log only)
                num_kv_heads = 0
                head_dim = query_tensor.shape[2]

                for req_id, topk_ids in zip(request_ids, cluster_ids_list):
                    buffer_key = _resolve_buffer_key(req_id, request_id_to_buffer_key)
                    if buffer_key not in sparse_kv_cpu_buffers:
                        raise RuntimeError(f"No CPU KV buffer found for request {req_id}")
                    buffer = sparse_kv_cpu_buffers[buffer_key][layer_idx]
                    if buffer.get('keys') is None or buffer.get('values') is None:
                        raise RuntimeError(f"CPU KV buffer not initialized for layer {layer_idx}")

                    meta = (nelssa_cluster_metadata.get(buffer_key, {}).get(layer_idx)
                            if nelssa_cluster_metadata else None)
                    if meta is None or not meta.get('reorganized', False) or meta.get('cluster_offsets') is None:
                        all_fused = False
                        break

                    valid_tokens = _buffer_total_tokens(buffer)
                    num_tokens_list.append(valid_tokens)
                    keys_src_list.append(buffer['keys'])
                    values_src_list.append(buffer['values'])
                    offsets_list.append(meta['cluster_offsets'])

                    # cluster_size must be a CPU int32 tensor for the C++ raw-pointer path
                    _csc = meta.get('cluster_size_cpu')
                    if _csc is not None:
                        cluster_size_cpu = _csc
                    else:
                        _cs = meta['cluster_size']
                        cluster_size_cpu = _cs if not _cs.is_cuda else _cs.cpu().contiguous()
                    sizes_list.append(cluster_size_cpu)

                    num_kv_heads = buffer['keys'].shape[0]
                    ids_list.append(_normalize_cluster_ids(
                        topk_ids, num_kv_heads, 'cpu'))
                _st['dict_loop'] = (_time.perf_counter() - _t0) * 1e3

            if all_fused:
                # Fixed M_CAP (default): gather buffer is a fixed bound, C++ writes
                # [0, lens) and zeroes the tail, fused masks via lens_batch. Baseline
                # (NELSSA_FIXED_M_CAP=0): size to the real per-layer M and slice.
                if not _FIXED_M_CAP:
                    # Per-layer M = max over (req, kvH) of the SUM of selected
                    # cluster sizes — the contiguous gather length, NOT the max
                    # source end-offset (that would be ~total_tokens, wrong).
                    per_req_m = []
                    for i in range(num_reqs):
                        _sz = sizes_list[i]
                        _ids = ids_list[i]
                        _sel = torch.gather(_sz, 1, _ids)  # [kvH, nprobe]
                        per_req_m.append(int(_sel.sum(1).max().item()))
                    max_tokens = max(per_req_m) if per_req_m else 1
                else:
                    max_tokens = 0  # _get_gather_batch_buffers ignores it
                _t0 = _time.perf_counter()
                keys_batch, values_batch = _get_gather_batch_buffers(
                    num_reqs, num_kv_heads, head_dim, max_tokens)
                if not _FIXED_M_CAP:
                    keys_batch = keys_batch[:, :, :max_tokens, :].contiguous()
                    values_batch = values_batch[:, :, :max_tokens, :].contiguous()
                    max_tokens = keys_batch.shape[2]
                else:
                    max_tokens = keys_batch.shape[2]  # M_CAP (fixed)
                lens_batch = torch.zeros((num_reqs, num_kv_heads), dtype=torch.int32)
                _st['buf_prep'] = (_time.perf_counter() - _t0) * 1e3

                # Per-req workload snapshot (layer 0 only, to keep the log quiet).
                if _CORELOG and layer_idx == 0:
                    _wl = []
                    for i in range(num_reqs):
                        _wl.append(
                            f"req{i}:num_tokens={num_tokens_list[i]}"
                            f",n_centroids={offsets_list[i].shape[1]}"
                            f",nprobe={ids_list[i].shape[1]}")
                    logger.info(
                        "[NELSSA][CPU-ATTN] batch path layer=0 N=%d kvH=%d "
                        "M_CAP=%d | %s",
                        num_reqs, num_kv_heads, max_tokens, " ".join(_wl))

                # Gather requests using C++ dispatch
                _t_gather = time.perf_counter()
                nvtx.push_range("[P] cpp_gather")
                _GATHER_EXT.gather_selected_clusters_batch_cpu(
                    keys_batch, values_batch, lens_batch,
                    keys_src_list, values_src_list,
                    offsets_list, sizes_list, ids_list,
                    num_kv_heads,
                )
                nvtx.pop_range()  # end [P] cpp_gather
                _t_gather = time.perf_counter() - _t_gather
                _st['gather'] = _t_gather * 1e3

                total_gathered = int(lens_batch.sum().item())
                _t_fused = time.perf_counter()
                # M_CAP (fixed) in the label; tok= actual gathered tokens so
                # padding waste = N*kvH*M_CAP - tok is still visible per layer.
                nvtx.push_range(
                    f"[P] fused_attn M={max_tokens} tok={total_gathered} "
                    f"N={num_reqs}")
                out, lse = _get_fused_outputs(
                    num_reqs, query_tensor.shape[1], query_tensor.shape[2])
                _GATHER_EXT.cpu_attention_fused(
                    query_tensor, keys_batch, values_batch, lens_batch,
                    out, lse, num_kv_heads, head_dim)
                result = (out, lse)
                nvtx.pop_range()  # end [P] fused_attn
                _t_fused = time.perf_counter() - _t_fused
                _st['fused'] = _t_fused * 1e3

                if _CORELOG and layer_idx == 0:
                    logger.info(
                        "[NELSSA][CPU-ATTN] batch path layer=0 timing: "
                        "batch_gather=%.2fms fused_attn=%.2fms "
                        "(M_CAP=%d tok=%d N=%d kvH=%d)",
                        _t_gather * 1e3, _t_fused * 1e3,
                        max_tokens, total_gathered, num_reqs, num_kv_heads)
            else:
                # ---- Fallback: per-req gather loop + pad_to_batch (legacy) ----
                nvtx.push_range("[P] gather_loop")
                _t_gather = time.perf_counter()
                for req_id, topk_ids in zip(request_ids, cluster_ids_list):
                    buffer_key = _resolve_buffer_key(req_id, request_id_to_buffer_key)
                    if buffer_key not in sparse_kv_cpu_buffers:
                        raise RuntimeError(f"No CPU KV buffer found for request {req_id}")
                    buffer = sparse_kv_cpu_buffers[buffer_key][layer_idx]
                    if buffer.get('keys') is None or buffer.get('values') is None:
                        raise RuntimeError(f"CPU KV buffer not initialized for layer {layer_idx}")

                    meta = (nelssa_cluster_metadata.get(buffer_key, {}).get(layer_idx)
                            if nelssa_cluster_metadata else None)
                    valid_tokens = _buffer_total_tokens(buffer)

                    use_fused = (not _legacy and _fused_available
                                 and meta is not None and meta.get('reorganized', False)
                                 and meta.get('cluster_offsets') is not None)
                    if use_fused:
                        # Gather into a pooled buffer; the fused kernel runs over
                        # the batched result (forwards reuse_buffers).
                        _kvH = buffer['keys'].shape[0]
                        _reuse = _get_gather_buffers(_kvH, buffer['keys'].shape[2], 1)
                        keys, values, mask, lens = _select_kv_sparse(
                            buffer['keys'], buffer['values'], topk_ids, meta,
                            layer_idx, buffer_key, cpu_attention_kv_fraction,
                            num_valid_tokens=valid_tokens, return_lens=True,
                            reuse_buffers=_reuse)
                        per_req_kv.append((keys, values, mask))
                        per_req_lens.append(lens.to(torch.int32))
                        _checked_out.append((keys, values))
                    else:
                        keys, values, mask = _select_kv_sparse(
                            buffer['keys'], buffer['values'], topk_ids, meta,
                            layer_idx, buffer_key, cpu_attention_kv_fraction,
                            num_valid_tokens=valid_tokens)
                        per_req_kv.append((keys, values, mask))
                nvtx.pop_range()  # end [P] gather_loop
                _t_gather = time.perf_counter() - _t_gather

                _t_pad = time.perf_counter()
                nvtx.push_range("[P] pad_to_batch")
                keys_batch, values_batch, masks = _pad_to_batch(per_req_kv)
                nvtx.pop_range()  # end [P] pad_to_batch
                _t_pad = time.perf_counter() - _t_pad

                if (not _legacy and _fused_available and per_req_lens
                        and len(per_req_lens) == num_reqs):
                    # Fused kernel: int32 [N, kvH] per_head_lens batch padded to
                    # max_tokens. The NVTX label encodes M and gathered tokens to
                    # correlate per-layer variance with the GEMM M-dim.
                    max_tokens = int(keys_batch.shape[1])
                    total_gathered = int(sum(int(l.sum().item()) for l in per_req_lens))
                    _t_fused = time.perf_counter()
                    nvtx.push_range(
                        f"[P] fused_attn M={max_tokens} tok={total_gathered} "
                        f"N={num_reqs}")
                    num_kv_heads = keys_batch.shape[1]
                    lens_batch = torch.zeros((num_reqs, num_kv_heads), dtype=torch.int32)
                    for i, l in enumerate(per_req_lens):
                        lens_batch[i, :l.numel()] = l
                    out, lse = _get_fused_outputs(
                        num_reqs, query_tensor.shape[1], query_tensor.shape[2])
                    _GATHER_EXT.cpu_attention_fused(
                        query_tensor, keys_batch, values_batch, lens_batch,
                        out, lse, num_kv_heads, query_tensor.shape[2])
                    result = (out, lse)
                    nvtx.pop_range()  # end [P] fused_attn
                    _t_fused = time.perf_counter() - _t_fused

                    if layer_idx == 0:
                        logger.info(
                            "[NELSSA][CPU-ATTN] fallback path layer=0 timing: "
                            "gather_loop=%.2fms pad=%.2fms fused_attn=%.2fms "
                            "(M=%d tok=%d N=%d kvH=%d)",
                            _t_gather * 1e3, _t_pad * 1e3, _t_fused * 1e3,
                            max_tokens, total_gathered, num_reqs, num_kv_heads)
                else:
                    _t_legacy = time.perf_counter()
                    nvtx.push_range("[P] legacy_attn")
                    result = _batched_attention_f32(
                        query_tensor, keys_batch, values_batch, masks)
                    nvtx.pop_range()  # end [P] legacy_attn
                    _t_legacy = time.perf_counter() - _t_legacy
                    if layer_idx == 0:
                        logger.info(
                            "[NELSSA][CPU-ATTN] legacy path layer=0 timing: "
                            "gather=%.2fms pad=%.2fms legacy_attn=%.2fms N=%d",
                            _t_gather * 1e3, _t_pad * 1e3, _t_legacy * 1e3, num_reqs)
        finally:
            # Return pooled gather buffers (fallback path only; the batched path
            # uses _GATHER_BATCH_POOL, not the per-req pool).
            for _k, _v in _checked_out:
                _return_gather_buffers(_k, _v)
            nvtx.pop_range()
            # Record per-stage server-side timing to attribute the srv_handler
            # overhead (deserialize / dict_loop / buf_prep / gather / fused).
            _col = getattr(RPCAttentionEngine, '_server_stage_collector', None)
            if _col is not None:
                _col.append(_st)
            # Log this thread's core/affinity + every other thread's /proc
            # Cpus_allowed to confirm 1:1 core spread. Gated by
            # NELSSA_ATTN_CORELOG=1 (the /proc walk is the hot-path's heaviest
            # debug cost; zero overhead when off).
            if _CORELOG and layer_idx == 0:
                try:
                    import threading as _threading
                    _cur_tid = str(_threading.get_native_id())
                    _cur_cpu = _os.sched_getcpu() if hasattr(_os, "sched_getcpu") else -1
                    _aff = sorted(_os.sched_getaffinity(0))
                    _nt = torch.get_num_threads()
                    _per_thread = []
                    _cur_marker = ""
                    for _tid in sorted(_os.listdir("/proc/self/task"),
                                       key=lambda x: int(x)):
                        try:
                            with open(f"/proc/self/task/{_tid}/status") as _f:
                                _lines = _f.read().splitlines()
                            _name = next((l.split("\t")[-1].strip()
                                          for l in _lines
                                          if l.startswith("Name:")), "?")
                            _cpus = next((l.split(":", 1)[1].strip()
                                          for l in _lines
                                          if l.startswith("Cpus_allowed:")), "?")
                            _cpus_list = next((l.split(":", 1)[1].strip()
                                               for l in _lines
                                               if l.startswith("Cpus_allowed_list:")), "?")
                            _is_cur = " *" if _tid == _cur_tid else ""
                            if _is_cur:
                                _cur_marker = (_cur_marker or
                                                f"{_tid}:{_name[:10]} cpu={_cur_cpu} "
                                                f"aff={_aff} mask={_cpus}")
                            _per_thread.append(f"{_tid}:{_name[:10]}={_cpus_list}{_is_cur}")
                        except Exception:
                            pass
                    logger.info(
                        "[NELSSA][CPU-ATTN-AFFINITY] layer=0 cur_tid=%s "
                        "cur_cpu=%d affinity_cores=%d %s torch_threads=%d | "
                        "cur_thread=[%s] | per_thread_affinity=%s",
                        _cur_tid, _cur_cpu, len(_aff), _aff, _nt,
                        _cur_marker,
                        " ".join(_per_thread[:48]))
                except Exception as _e:
                    logger.warning("[NELSSA][CPU-ATTN-AFFINITY] %s", _e)

        return result


def _resolve_buffer_key(req_id: str, request_id_to_buffer_key: dict | None) -> str:
    """Map a D-side request_id to the P-side buffer_key via shared uuid."""
    if not request_id_to_buffer_key:
        return req_id
    decode_uuid = req_id.split('-')[1] if len(req_id.split('-')) >= 2 else req_id
    for local_req_id, mapped_key in request_id_to_buffer_key.items():
        local_uuid = (local_req_id.split('-')[1]
                      if len(local_req_id.split('-')) >= 2 else local_req_id)
        if decode_uuid == local_uuid:
            return mapped_key
    return req_id


# ---------------------------------------------------------------------------
# LSE-based merge of GPU + CPU attention outputs
# ---------------------------------------------------------------------------

class AttentionResultMerger:
    """LSE-based online softmax merge of GPU + CPU attention results."""

    @staticmethod
    def merge(
        result1: tuple[torch.Tensor, torch.Tensor],
        result2: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out1, lse1 = result1
        out2, lse2 = result2

        global_lse = torch.logsumexp(torch.cat([lse1, lse2], dim=-1), dim=-1, keepdim=True)
        weight1 = torch.nan_to_num(torch.exp(lse1 - global_lse), nan=0.0)
        weight2 = torch.nan_to_num(torch.exp(lse2 - global_lse), nan=0.0)
        return out1 * weight1 + out2 * weight2, global_lse

    @staticmethod
    def merge_with_mask(
        gpu_output: torch.Tensor,    # [batch_size, num_heads, head_dim]
        gpu_lse: torch.Tensor,       # [batch_size, num_heads, 1]
        cpu_output: torch.Tensor,    # [max_num_long_requests, num_heads, head_dim]
        cpu_lse: torch.Tensor,       # [max_num_long_requests, num_heads, 1]
        long_request_mask: torch.Tensor,  # [batch_size] 1 if Long, 0 if Short
    ) -> torch.Tensor:
        batch_size = gpu_output.shape[0]
        mask_f32 = long_request_mask.float()
        num_long = int(mask_f32.sum().item())
        if num_long == 0:
            return gpu_output

        # Extract GPU long-request results, pad to max_num_long_requests.
        mask_expanded = mask_f32.view(batch_size, 1, 1)
        gpu_long_output = gpu_output * mask_expanded
        gpu_long_lse = gpu_lse * mask_expanded

        max_num_long = cpu_output.shape[0]
        gpu_long_output_padded = torch.zeros(
            max_num_long, *gpu_output.shape[1:], device=gpu_output.device, dtype=gpu_output.dtype)
        gpu_long_lse_padded = torch.full(
            (max_num_long, gpu_lse.shape[1], 1), float('-inf'),
            device=gpu_lse.device, dtype=gpu_lse.dtype)
        bool_mask = mask_f32.bool()
        gpu_long_output_padded[:num_long] = gpu_long_output[bool_mask]
        gpu_long_lse_padded[:num_long] = gpu_lse[bool_mask]

        # LSE-weighted merge of GPU + CPU for long positions.
        cpu_out_bf = cpu_output.to(dtype=torch.bfloat16)
        cpu_lse_bf = cpu_lse.to(dtype=torch.bfloat16)
        global_lse = torch.logsumexp(
            torch.cat([gpu_long_lse_padded, cpu_lse_bf], dim=-1), dim=-1, keepdim=True)
        weight_gpu = torch.exp(gpu_long_lse_padded - global_lse)
        weight_cpu = torch.exp(cpu_lse_bf - global_lse)

        merged = (gpu_long_output_padded * weight_gpu + cpu_out_bf * weight_cpu
                  ).to(dtype=gpu_output.dtype)

        final_output = gpu_output.clone()
        final_output[bool_mask] = merged[:num_long]
        return final_output

    _merge_kernel = None

    @classmethod
    def _ensure_merge_kernel(cls):
        """Load the fused merge kernel on first use. Returns the module or None."""
        if cls._merge_kernel is None:
            cls._merge_kernel = cls._load_merge_kernel()
        return cls._merge_kernel

    @staticmethod
    def _load_merge_kernel():
        import importlib.util
        ext_dir = os.path.dirname(os.path.abspath(__file__))
        so_files = [f for f in os.listdir(ext_dir)
                    if f.startswith("vllm_merge_ext") and f.endswith(".so")]
        if not so_files:
            return None
        try:
            spec = importlib.util.spec_from_file_location(
                "vllm_merge_ext", os.path.join(ext_dir, so_files[0]))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logger.info(f"[NELSSA] Fused merge kernel loaded from {so_files[0]}")
            return mod
        except Exception as e:
            logger.warning(f"[NELSSA] Fused merge kernel not available: {e}; "
                           "using torch-op fallback.")
            return None

    @staticmethod
    def merge_with_mask_static(
        gpu_output,          # [num_reqs, H, D] bf16 GPU (in-place)
        gpu_lse,              # [num_reqs, H, 1] f32 GPU
        cpu_output_padded,    # [max_num_long, H, D] bf16 GPU (static-pad)
        cpu_lse_padded,       # [max_num_long, H, 1] f32 GPU (static-pad)
        long_indices,         # [max_num_long] int64 GPU (static, values per step)
        valid_mask,           # [max_num_long] float GPU (1=valid, 0=pad)
    ) -> torch.Tensor:
        """CUDA-Graph-safe LSE merge, fixed shape = max_num_long."""
        if AttentionResultMerger._ensure_merge_kernel() is None:
            return AttentionResultMerger._merge_with_mask_static_torch(
                gpu_output, gpu_lse, cpu_output_padded, cpu_lse_padded, long_indices, valid_mask)
        AttentionResultMerger._merge_kernel.merge_lse(
            gpu_output, gpu_lse, cpu_output_padded, cpu_lse_padded, long_indices, valid_mask)
        return gpu_output

    @staticmethod
    def _merge_with_mask_static_torch(
        gpu_output, gpu_lse, cpu_output_padded, cpu_lse_padded, long_indices, valid_mask,
    ) -> torch.Tensor:
        """Torch-op fallback, bit-equivalent to the fused kernel."""
        m = long_indices.shape[0]
        gpu_long_out = torch.index_select(gpu_output, 0, long_indices)
        gpu_long_lse = torch.index_select(gpu_lse, 0, long_indices)

        # Pad cpu_lse slots with -inf so their CPU weight is 0.
        vm = valid_mask.view(m, 1, 1)
        cpu_lse_eff = torch.where(
            vm > 0, cpu_lse_padded, torch.full_like(cpu_lse_padded, float('-inf')))

        g = torch.logsumexp(
            torch.cat([gpu_long_lse.float(), cpu_lse_eff.float()], dim=-1), dim=-1, keepdim=True)
        merged_long = (gpu_long_out.float() * torch.exp(gpu_long_lse.float() - g)
                       + cpu_output_padded.float() * torch.exp(cpu_lse_eff.float() - g))
        gpu_output.index_copy_(0, long_indices, merged_long.to(gpu_output.dtype))
        return gpu_output

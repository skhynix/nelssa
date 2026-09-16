# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import torch
import math
import time
import nvtx
import os

from vllm.logger import init_logger
from vllm.v1.core.kmeans_utils import segment_k_means

logger = init_logger(__name__)


def _load_clustering_ext():
    """Load the OpenMP C++ reorganization extension (.so) if present."""
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    so_files = [f for f in os.listdir(ext_dir)
                if f.startswith("vllm_clustering_ext") and f.endswith(".so")]
    if not so_files:
        logger.warning("[NELSSA] C++ Clustering Extension (.so file) not found. Using PyTorch fallback.")
        return None
    so_path = os.path.join(ext_dir, so_files[0])
    spec = importlib.util.spec_from_file_location("vllm_clustering_ext", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logger.info(f"[NELSSA] C++ Clustering Extension loaded from {so_files[0]} (multi-threaded reorganization with OpenMP)")
    return mod


try:
    _CLUSTERING_EXT = _load_clustering_ext()
except Exception as e:
    logger.warning(f"[NELSSA] C++ Clustering Extension not available: {e}. Using PyTorch fallback.")
    _CLUSTERING_EXT = None


def _to_cpu_contiguous(t: torch.Tensor) -> torch.Tensor:
    """CPU-contiguous copy of t (k-means outputs may live on GPU)."""
    return t.cpu().contiguous() if t.is_cuda else t.contiguous()


def _buffer_total_tokens(buf: dict) -> int:
    off = buf['offset']     # Valid tokens
    return off.item() if isinstance(off, torch.Tensor) else off


class KVCacheClusteringEngine:
    @staticmethod
    def _get_buffer_key(req_id: str, is_pd_disaggregated: bool, get_remote_request_id_fn) -> str:
        if is_pd_disaggregated and get_remote_request_id_fn:
            return get_remote_request_id_fn(req_id) or req_id
        return req_id

    @staticmethod
    def _compute_cluster_offsets(cluster_size_cpu: torch.Tensor,
                                 num_kv_heads: int) -> torch.Tensor:
        # cluster_offsets[h, c] = sum of cluster_size[h, 0..c-1] (cumsum, shifted).
        n_centroids = cluster_size_cpu.shape[1]
        offsets = torch.zeros((num_kv_heads, n_centroids), dtype=torch.int32)
        offsets[:, 1:] = cluster_size_cpu.cumsum(dim=1)[:, :-1].to(torch.int32)
        return offsets

    @staticmethod
    def cluster_sparse_kv(
        request_ids: list[str],
        nelssa_gpu_buffers: dict,
        nelssa_cluster_metadata: dict,
        is_pd_disaggregated: bool,
        get_remote_request_id_fn,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """K-means cluster the GPU KV buffers and store centroid metadata."""
        nvtx.push_range("[P] Clustering")
        try:
            logger.info(f"[NELSSA Clustering] Starting clustering for {len(request_ids)} requests")
            for req_id in request_ids:
                buffer_key = KVCacheClusteringEngine._get_buffer_key(
                    req_id, is_pd_disaggregated, get_remote_request_id_fn)
                if buffer_key not in nelssa_gpu_buffers:
                    logger.warning(f"[NELSSA Clustering] No GPU buffer found for {req_id} (key: {buffer_key})")
                    continue
                gpu_buffers = nelssa_gpu_buffers[buffer_key]

                for layer_idx, gpu_buffer in gpu_buffers.items():
                    nvtx.push_range(f"[P] Clustering_L{layer_idx}")
                    try:
                        total_tokens = _buffer_total_tokens(gpu_buffer)
                        if total_tokens <= 0:
                            logger.warning(f"[NELSSA Clustering] Invalid total_tokens={total_tokens} for {buffer_key} layer {layer_idx}")
                            continue

                        n_segment = max(round(total_tokens / 8192), 1)
                        n_factor = math.lcm(8, n_segment)
                        n_centroids = max(round(round(total_tokens / 16) / n_factor) * n_factor, n_factor)

                        keys_view = gpu_buffer['keys'][:, :total_tokens, :].reshape(num_kv_heads, total_tokens, head_dim)
                        values_view = gpu_buffer['values'][:, :total_tokens, :].reshape(num_kv_heads, total_tokens, head_dim)
                        mean_key = torch.mean(keys_view, dim=1, keepdim=True)

                        # Center keys before k-means, then shift centroids back.
                        centroids, _, clusters, cluster_size = segment_k_means(
                            key=keys_view - mean_key, value=values_view,
                            num_centroids=n_centroids, num_segments=n_segment,
                        )
                        centroids = centroids + mean_key

                        meta = nelssa_cluster_metadata.setdefault(buffer_key, {}).setdefault(layer_idx, {})
                        # Cache a CPU int32 copy of cluster_size so the RPC
                        # server's CPU-attention hot path avoids a per-layer
                        # GPU->CPU sync (.cpu().contiguous()) on every decode
                        # layer (the 529-534 cluster_size_cpu fallback). This
                        # is computed once at clustering (prefill) time.
                        cluster_size_cpu = (cluster_size
                                             if not cluster_size.is_cuda
                                             else cluster_size.cpu().contiguous())
                        meta.update(centroids=centroids, cluster_size=cluster_size,
                                    cluster_size_cpu=cluster_size_cpu,
                                    n_centroids=n_centroids, clusters=clusters,
                                    cluster_size_mask=cluster_size == 0)

                        if layer_idx == 0 and req_id == request_ids[0]:
                            logger.info(f"[NELSSA Clustering] Layer 0: tokens={total_tokens}, segments={n_segment}, centroids={n_centroids}, cluster_size_range=[{cluster_size.min().item()}, {cluster_size.max().item()}]")

                        del keys_view, values_view, mean_key
                    except Exception as e:
                        logger.error(f"[NELSSA Clustering] ERROR in layer {layer_idx} for {buffer_key}: {e}")
                        raise
                    finally:
                        nvtx.pop_range()
            logger.info(f"[NELSSA Clustering] Clustering complete for {len(request_ids)} requests")
        finally:
            nvtx.pop_range()

    @staticmethod
    def cleanup_gpu_buffers(
        request_ids: list[str],
        nelssa_gpu_buffers: dict,
        is_pd_disaggregated: bool,
        kv_role: str | None = None,
        get_remote_request_id_fn=None,
    ) -> None:
        logger.info(f"[NELSSA Cleanup] Starting cleanup for {len(request_ids)} requests, kv_role={kv_role}")
        for req_id in request_ids:
            buffer_key = KVCacheClusteringEngine._get_buffer_key(req_id, is_pd_disaggregated, get_remote_request_id_fn)
            if buffer_key not in nelssa_gpu_buffers:
                logger.warning(f"[NELSSA Cleanup] No buffer found for {req_id} (key: {buffer_key}), skipping")
                continue

            buf = nelssa_gpu_buffers[buffer_key]
            total_size_gb = sum((b['keys'].numel() + b['values'].numel()) * 2 / (1024**3) for b in buf.values())
            del nelssa_gpu_buffers[buffer_key]
            logger.info(f"[NELSSA Cleanup] Freed gpu_buffer for {req_id} (key: {buffer_key}): ~{total_size_gb:.2f}GB")

    @staticmethod
    def reorganize_by_clusters_cpu_cpp(
        request_ids: list[str],
        nelssa_cluster_metadata: dict,
        nelssa_cpu_buffers: dict,
        is_pd_disaggregated: bool,
        get_remote_request_id_fn,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
    ) -> None:
        """Reorganize CPU KV buffers into cluster-contiguous layout via the
        OpenMP C++ extension (default). Bit-exact with the PyTorch fallback;
        set NELSSA_USE_CPP_REORG=0 to force the slower PyTorch path."""
        if _CLUSTERING_EXT is None or os.environ.get("NELSSA_USE_CPP_REORG", "1") != "1":
            if _CLUSTERING_EXT is None:
                logger.warning("[NELSSA Reorg] C++ extension not available, falling back to PyTorch")
            else:
                logger.info("[NELSSA Reorg] Using PyTorch fallback (NELSSA_USE_CPP_REORG=0)")
            return KVCacheClusteringEngine.reorganize_by_clusters_cpu(
                request_ids, nelssa_cluster_metadata, nelssa_cpu_buffers,
                is_pd_disaggregated, get_remote_request_id_fn,
                num_kv_heads, head_dim, block_size,
            )

        nvtx.push_range("[P] Reorganize_CPP")
        start_time = time.perf_counter()
        try:
            logger.info(f"[NELSSA Reorg] Starting C++ cluster reorganization for {len(request_ids)} requests")

            for req_id in request_ids:
                buffer_key = KVCacheClusteringEngine._get_buffer_key(req_id, is_pd_disaggregated, get_remote_request_id_fn)
                if buffer_key not in nelssa_cluster_metadata:
                    logger.warning(f"[NELSSA Reorg] No cluster metadata found for {req_id} (key: {buffer_key})")
                    continue
                if buffer_key not in nelssa_cpu_buffers:
                    logger.warning(f"[NELSSA Reorg] No CPU buffer found for {req_id} (key: {buffer_key})")
                    continue

                cpu_buffers = nelssa_cpu_buffers[buffer_key]
                cluster_metadata = nelssa_cluster_metadata[buffer_key]
                
                keys_dst_list, values_dst_list = [], []
                keys_src_list, values_src_list = [], []
                clusters_list, cluster_size_list = [], []
                total_tokens_list: list[int] = []
                layer_entries: list[tuple[int, dict, torch.Tensor, torch.Tensor, int]] = []

                for layer_idx, cpu_buffer in cpu_buffers.items():
                    if layer_idx not in cluster_metadata:
                        logger.warning(f"[NELSSA Reorg] No cluster metadata for layer {layer_idx}")
                        continue

                    meta = cluster_metadata[layer_idx]
                    clusters = meta.get('clusters')
                    cluster_size = meta.get('cluster_size')
                    if clusters is None or cluster_size is None:
                        logger.warning(f"[NELSSA Reorg] Missing clusters or cluster_size for layer {layer_idx}")
                        continue

                    total_tokens = _buffer_total_tokens(cpu_buffer)
                    if total_tokens <= 0:
                        logger.warning(f"[NELSSA Reorg] Invalid total_tokens={total_tokens} for {buffer_key} layer {layer_idx}")
                        continue

                    keys_src = cpu_buffer['keys'][:, :total_tokens, :].contiguous()
                    values_src = cpu_buffer['values'][:, :total_tokens, :].contiguous()

                    keys_dst = torch.empty_like(keys_src)
                    values_dst = torch.empty_like(values_src)

                    # Copy to CPU here so the batched C++ kernel stays free of
                    # per-layer CUDA syncs (which would serialize under OpenMP).
                    clusters_cpu = _to_cpu_contiguous(clusters)
                    cluster_size_cpu = _to_cpu_contiguous(cluster_size)

                    keys_dst_list.append(keys_dst)
                    values_dst_list.append(values_dst)
                    keys_src_list.append(keys_src)
                    values_src_list.append(values_src)
                    clusters_list.append(clusters_cpu)
                    cluster_size_list.append(cluster_size_cpu)
                    total_tokens_list.append(total_tokens)
                    layer_entries.append((layer_idx, meta, keys_src, keys_dst, total_tokens))

                if not layer_entries:
                    continue

                _CLUSTERING_EXT.reorganize_by_clusters_cpu_batch(
                    keys_dst_list, values_dst_list,
                    keys_src_list, values_src_list,
                    clusters_list, cluster_size_list,
                    total_tokens_list,
                    num_kv_heads,
                )

                for i, (layer_idx, meta, keys_src, keys_dst, total_tokens) in enumerate(layer_entries):
                    cpu_buffer = cpu_buffers[layer_idx]
                    values_dst = values_dst_list[i]
                    cluster_size_cpu = cluster_size_list[i]

                    cpu_buffer['keys'][:, :total_tokens, :] = keys_dst
                    cpu_buffer['values'][:, :total_tokens, :] = values_dst

                    meta['cluster_offsets'] = KVCacheClusteringEngine._compute_cluster_offsets(cluster_size_cpu, num_kv_heads)
                    meta['cluster_size_cpu'] = cluster_size_cpu
                    meta['reorganized'] = True

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"[NELSSA Reorg] C++ Cluster reorganization complete for {len(request_ids)} requests in {elapsed_ms:.1f}ms")
        finally:
            nvtx.pop_range()

    @staticmethod
    def reorganize_by_clusters_cpu(
        request_ids: list[str],
        nelssa_cluster_metadata: dict,
        nelssa_cpu_buffers: dict,
        is_pd_disaggregated: bool,
        get_remote_request_id_fn,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_threads: int = 4,
    ) -> None:
        """PyTorch fallback: reorganize CPU KV buffers into cluster-contiguous
        layout using torch.index_select (slower; reference implementation)."""
        nvtx.push_range("[P] Reorganize")
        start_time = time.perf_counter()
        try:
            logger.info(f"[NELSSA Reorg] Starting cluster reorganization for {len(request_ids)} requests")

            for req_id in request_ids:
                buffer_key = KVCacheClusteringEngine._get_buffer_key(req_id, is_pd_disaggregated, get_remote_request_id_fn)
                if buffer_key not in nelssa_cluster_metadata:
                    logger.warning(f"[NELSSA Reorg] No cluster metadata found for {req_id} (key: {buffer_key})")
                    continue
                if buffer_key not in nelssa_cpu_buffers:
                    logger.warning(f"[NELSSA Reorg] No CPU buffer found for {req_id} (key: {buffer_key})")
                    continue

                cpu_buffers = nelssa_cpu_buffers[buffer_key]
                cluster_metadata = nelssa_cluster_metadata[buffer_key]

                for layer_idx, cpu_buffer in cpu_buffers.items():
                    if layer_idx not in cluster_metadata:
                        logger.warning(f"[NELSSA Reorg] No cluster metadata for layer {layer_idx}")
                        continue

                    meta = cluster_metadata[layer_idx]
                    clusters = meta.get('clusters')
                    cluster_size = meta.get('cluster_size')
                    n_centroids = meta.get('n_centroids', 0)
                    if clusters is None or cluster_size is None:
                        logger.warning(f"[NELSSA Reorg] Missing clusters or cluster_size for layer {layer_idx}")
                        continue

                    try:
                        total_tokens = _buffer_total_tokens(cpu_buffer)
                        if total_tokens <= 0:
                            logger.warning(f"[NELSSA Reorg] Invalid total_tokens={total_tokens} for {buffer_key} layer {layer_idx}")
                            continue

                        keys_src = cpu_buffer['keys'][:, :total_tokens, :]
                        values_src = cpu_buffer['values'][:, :total_tokens, :]
                        keys_dst = torch.empty_like(keys_src)
                        values_dst = torch.empty_like(values_src)

                        # Build the per-head src->dst index, then vectorized copy.
                        for kv_head in range(num_kv_heads):
                            cluster_indices = clusters[kv_head]
                            cluster_sizes_vec = cluster_size[kv_head]
                            offsets = torch.zeros(n_centroids + 1, dtype=torch.int64)
                            for i in range(n_centroids):
                                offsets[i + 1] = offsets[i] + cluster_sizes_vec[i]

                            dst_positions = torch.zeros(total_tokens, dtype=torch.int64)
                            src_positions = torch.zeros(total_tokens, dtype=torch.int64)
                            pos = 0
                            for centroid_idx in range(n_centroids):
                                size = cluster_sizes_vec[centroid_idx].item()
                                if size > 0:
                                    dst_positions[pos:pos + size] = offsets[centroid_idx]
                                    src_positions[pos:pos + size] = cluster_indices[centroid_idx, :size]
                                    pos += size

                            keys_dst[kv_head] = torch.index_select(keys_src[kv_head], 0, src_positions[:pos])
                            values_dst[kv_head] = torch.index_select(values_src[kv_head], 0, src_positions[:pos])

                        cpu_buffer['keys'][:, :total_tokens, :] = keys_dst
                        cpu_buffer['values'][:, :total_tokens, :] = values_dst

                        cluster_size_cpu = _to_cpu_contiguous(cluster_size)
                        meta['cluster_offsets'] = KVCacheClusteringEngine._compute_cluster_offsets(cluster_size_cpu, num_kv_heads)
                        # CPU copy so decode-side attention never pays a per-layer
                        # GPU->CPU copy for cluster_size.
                        meta['cluster_size_cpu'] = cluster_size_cpu
                        meta['reorganized'] = True
                    except Exception as e:
                        logger.error(f"[NELSSA Reorg] ERROR reorganizing layer {layer_idx} for {buffer_key}: {e}")
                        raise

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"[NELSSA Reorg] Cluster reorganization complete for {len(request_ids)} requests in {elapsed_ms:.1f}ms")
        finally:
            nvtx.pop_range()

    @staticmethod
    def get_cluster_kv_indices(
        req_id: str,
        layer_idx: int,
        nelssa_cluster_metadata: dict,
        is_pd_disaggregated: bool,
        get_remote_request_id_fn,
        topk_cluster_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (kv_indices[num_kv_heads, nprobe, max_cluster_size],
        kv_sizes[num_kv_heads, nprobe]) for the selected clusters."""
        buffer_key = KVCacheClusteringEngine._get_buffer_key(req_id, is_pd_disaggregated, get_remote_request_id_fn)

        if buffer_key not in nelssa_cluster_metadata:
            raise ValueError(f"No cluster metadata found for {req_id} (key: {buffer_key})")
        if layer_idx not in nelssa_cluster_metadata[buffer_key]:
            raise ValueError(f"No cluster metadata for layer {layer_idx}")

        meta = nelssa_cluster_metadata[buffer_key][layer_idx]
        clusters = meta.get('clusters')
        cluster_size = meta.get('cluster_size')
        if clusters is None or cluster_size is None:
            raise ValueError(f"Missing clusters or cluster_size for layer {layer_idx}")

        num_kv_heads = clusters.shape[0]
        nprobe = topk_cluster_ids.shape[1]
        max_cluster_size = clusters.shape[2]

        kv_indices = torch.zeros((num_kv_heads, nprobe, max_cluster_size), dtype=torch.int64, device=clusters.device)
        kv_sizes = torch.zeros((num_kv_heads, nprobe), dtype=torch.int32, device=clusters.device)

        for kv_head in range(num_kv_heads):
            for i in range(nprobe):
                centroid_idx = topk_cluster_ids[kv_head, i].item()
                size = cluster_size[kv_head, centroid_idx].item() if isinstance(cluster_size[kv_head, centroid_idx], torch.Tensor) else cluster_size[kv_head, centroid_idx]
                kv_indices[kv_head, i, :size] = clusters[kv_head, centroid_idx, :size]
                kv_sizes[kv_head, i] = size

        return kv_indices, kv_sizes

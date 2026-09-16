# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Similarity search for sparse KV cache using centroid-based retrieval.

This module implements the similarity search logic from nelssa_gpu, adapted for vLLM.
Uses torch.bmm for efficient Q @ C^T computation followed by softmax and top-k selection.

Implementation matches RetrievalAttention's approach:
https://github.com/your-repo/RetrievalAttention/blob/main/cache_hub/retroinfer_cache.py
"""

import torch


def centroid_search(
    query: torch.Tensor,          # [kv_heads, 1, group_size, head_dim]
    centroids: torch.Tensor,      # [kv_heads, n_centroids, head_dim]
    cluster_size: torch.Tensor,   # [kv_heads, n_centroids]
    topk: int,
    rsqrt_dim: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform similarity search to find top-k centroids for each query.

    This is the vLLM adaptation of nelssa_gpu's sparse_attention() similarity search:
    https://github.com/your-repo/nelssa_gpu/blob/main/cache_hub/nelssa_cache.py#L1470-L1501

    Pipeline: GEMM (Q @ C^T) -> Scale -> Softmax -> Sum over group_size -> Top-k

    Args:
        query: Query tensor of shape [kv_heads, 1, group_size, head_dim]
        centroids: Centroid tensor of shape [kv_heads, n_centroids, head_dim]
        cluster_size: Cluster size tensor of shape [kv_heads, n_centroids]
                     (0 means empty cluster)
        topk: Number of top centroids to select (nprobe + es_cluster_num)
        rsqrt_dim: 1.0 / sqrt(head_dim) for scaling

    Returns:
        topk_indices: Top-k cluster indices of shape [kv_heads, topk]
        topk_values: Top-k similarity values of shape [kv_heads, topk]
    """
    kv_heads, _, group_size, head_dim = query.shape
    _, n_centroids, _ = centroids.shape

    # Reshape query: [kv_heads, 1, group_size, head_dim] -> [kv_heads, group_size, head_dim]
    query_squeezed = query.squeeze(1)

    # GEMM: Q @ C^T
    # query_squeezed: [kv_heads, group_size, head_dim]
    # centroids: [kv_heads, n_centroids, head_dim]
    # scores: [kv_heads, group_size, n_centroids]
    scores = torch.bmm(query_squeezed, centroids.transpose(1, 2))

    # Scale by 1/sqrt(head_dim)
    scores = scores * rsqrt_dim

    # Softmax
    softmax_scores = torch.softmax(scores, dim=-1)

    # Merge groups: sum over group_size dimension
    # scores: [kv_heads, group_size, n_centroids] -> [kv_heads, n_centroids]
    dist = torch.sum(softmax_scores, dim=1)

    # Mask empty clusters (cluster_size == 0)
    DTYPE_MIN = torch.finfo(dist.dtype).min
    dist = dist.masked_fill(cluster_size == 0, DTYPE_MIN)

    # Top-k selection
    topk_values, topk_indices = torch.topk(
        dist, topk, dim=-1, largest=True, sorted=True
    )

    return topk_indices, topk_values


def get_topk_cluster_ids(
    topk_indices: torch.Tensor,  # [batch_groups, topk]
    nprobe: int,
) -> torch.Tensor:
    """
    Extract retrieval zone cluster IDs (top nprobe).

    This matches nelssa_gpu's:
    self.cluster_ids.copy_(self.cI[..., :self.nprobe])

    Args:
        topk_indices: Top-k indices of shape [batch_groups, topk]
        nprobe: Number of clusters for retrieval zone (will be converted to int if tensor)

    Returns:
        cluster_ids: Cluster IDs of shape [batch_groups, nprobe]
    """
    # Convert nprobe to int
    nprobe = int(nprobe)
    return topk_indices[:, :nprobe]


def batched_centroid_search(
    queries: torch.Tensor,        # [num_long_requests, kv_heads, group_size, head_dim]
    centroids: torch.Tensor,      # [num_long_requests, kv_heads, n_centroids, head_dim]
    cluster_sizes: torch.Tensor,  # [num_long_requests, kv_heads, n_centroids]
    topk: int,
    rsqrt_dim: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform batched centroid-based similarity search for multiple long requests.

    This is an optimized version of centroid_search that processes multiple requests
    in a single batched operation using torch.bmm.

    Pipeline: GEMM (Q @ C^T) -> Scale -> Softmax -> Sum over group_size -> Top-k

    Args:
        queries: Query tensors of shape [num_long_requests, kv_heads, group_size, head_dim]
        centroids: Centroid tensors of shape [num_long_requests, kv_heads, n_centroids, head_dim]
        cluster_sizes: Cluster size tensors of shape [num_long_requests, kv_heads, n_centroids]
        topk: Number of top centroids to select
        rsqrt_dim: 1.0 / sqrt(head_dim) for scaling

    Returns:
        topk_indices: Top-k cluster indices of shape [num_long_requests * kv_heads, topk]
        topk_values: Top-k similarity values of shape [num_long_requests * kv_heads, topk]
    """
    # Handle 5D case: [num_requests, kv_heads, 1, group_size, head_dim] -> squeeze dim 2
    if queries.dim() == 5:
        if queries.shape[2] == 1:
            queries = queries.squeeze(2)  # [num_requests, kv_heads, group_size, head_dim]
            # print(f"[DEBUG] After squeeze(2): queries.shape = {queries.shape}")
        elif queries.shape[1] == 1:
            queries = queries.squeeze(1)  # [num_requests, kv_heads, group_size, head_dim]
            # print(f"[DEBUG] After squeeze(1): queries.shape = {queries.shape}")
        else:
            raise ValueError(f"Cannot squeeze 5D tensor with shape {queries.shape}: no dimension of size 1 found at positions 1 or 2")

    if queries.dim() != 4:
        raise ValueError(f"Expected queries to be 4D tensor [num_requests, kv_heads, group_size, head_dim], got {queries.dim()}D: {queries.shape}")
    num_requests, kv_heads, group_size, head_dim = queries.shape

    if centroids.dim() != 4:
        raise ValueError(f"Expected centroids to be 4D tensor, got {centroids.dim()}D: {centroids.shape}")
    _, _, n_centroids, _ = centroids.shape

    # Flatten for batched processing: [num_requests * kv_heads, group_size, head_dim]
    batch = num_requests * kv_heads
    queries_flat = queries.view(batch, group_size, head_dim)
    centroids_flat = centroids.view(batch, n_centroids, head_dim)
    cluster_sizes_flat = cluster_sizes.view(batch, n_centroids)

    # GEMM: Q @ C^T
    # queries_flat: [batch, group_size, head_dim]
    # centroids_flat: [batch, n_centroids, head_dim]
    # scores: [batch, group_size, n_centroids]
    scores = torch.bmm(queries_flat, centroids_flat.transpose(1, 2))

    # Scale by 1/sqrt(head_dim) in place (no new allocation).
    scores.mul_(rsqrt_dim)

    # Softmax
    softmax_scores = torch.softmax(scores, dim=-1)

    # Merge groups: sum over group_size dimension
    # scores: [batch, group_size, n_centroids] -> [batch, n_centroids]
    dist = torch.sum(softmax_scores, dim=1)

    # Mask empty clusters (cluster_size == 0)
    DTYPE_MIN = torch.finfo(dist.dtype).min
    dist = dist.masked_fill(cluster_sizes_flat, DTYPE_MIN)

    # Top-k selection
    topk_values, topk_indices = torch.topk(
        dist, topk, dim=-1, largest=True, sorted=True
    )

    return topk_indices, topk_values


def gather_centroids(
    centroids: torch.Tensor,  # [batch_groups, n_centroids, head_dim]
    indices: torch.Tensor,    # [batch_groups, topk]
) -> torch.Tensor:
    """
    Gather centroid vectors by indices.

    This matches nelssa_gpu's gather_copy_vectors kernel for estimation zone.

    Args:
        centroids: Centroid tensor of shape [batch_groups, n_centroids, head_dim]
        indices: Index tensor of shape [batch_groups, topk]

    Returns:
        Gathered centroids of shape [batch_groups, topk, head_dim]
    """
    batch_groups, n_centroids, head_dim = centroids.shape
    _, topk = indices.shape

    # Use advanced indexing for efficient gathering
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, head_dim)
    output = torch.gather(centroids, 1, expanded_indices)

    return output
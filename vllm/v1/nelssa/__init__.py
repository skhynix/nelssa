# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
NELSSA (Named Entity Layer-wise Sparse Selective Attention) Module.

This module provides:
1. CPU attention computation (Single-GPU mode)
2. RPC-based batch attention (P/D disaggregated mode)
3. Similarity search for cluster selection
4. K-means clustering for sparse KV cache
"""

from vllm.v1.nelssa.clustering import KVCacheClusteringEngine
from vllm.v1.nelssa.cpu_attention import AttentionResultMerger, CPUAttentionEngine, RPCAttentionEngine
from vllm.v1.nelssa.similarity_search import SimilaritySearchEngine

__all__ = [
    "CPUAttentionEngine",
    "RPCAttentionEngine",
    "SimilaritySearchEngine",
    "AttentionResultMerger",
    "KVCacheClusteringEngine",
]
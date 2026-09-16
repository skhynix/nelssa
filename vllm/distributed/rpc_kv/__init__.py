# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RPC-based KV cache communication for P/D disaggregated serving.

This module provides RPC communication between:
- Prefill Worker + Host CPU (RPC Server)
- Decode Worker (RPC Client)

Usage:
    # Single-GPU (no RPC needed)
    runner = GPUModelRunner(vllm_config)  # Works as before

    # P/D Disaggregated (Prefill Worker + Host CPU)
    vllm_config.kv_transfer_config.kv_role = "kv_producer"
    vllm_config.kv_transfer_config.kv_connector_extra_config["rpc_port"] = 8765
    runner = GPUModelRunner(vllm_config)  # Starts RPC Server

    # P/D Disaggregated (Decode Worker)
    vllm_config.kv_transfer_config.kv_role = "kv_consumer"
    vllm_config.kv_transfer_config.kv_connector_extra_config["host_cpu_rpc_addr"] = "localhost"
    vllm_config.kv_transfer_config.kv_connector_extra_config["host_cpu_rpc_port"] = 8765
    runner = GPUModelRunner(vllm_config)  # Starts RPC Client
"""

from vllm.distributed.rpc_kv.protocol import (
    SparseAttentionRequest,
    SparseAttentionResponse,
    KVStoreNotification,
    MESSAGE_TYPE_SPARSE_ATTN,
    MESSAGE_TYPE_KV_STORE,
    MESSAGE_TYPE_HEARTBEAT,
)
from vllm.distributed.rpc_kv.server import (
    RPCKVServer,
    RPCKVServerConfig,
)
from vllm.distributed.rpc_kv.client import (
    RPCKVClient,
    RPCKVClientConfig,
)

__all__ = [
    # Protocol
    "SparseAttentionRequest",
    "SparseAttentionResponse",
    "KVStoreNotification",
    "MESSAGE_TYPE_SPARSE_ATTN",
    "MESSAGE_TYPE_KV_STORE",
    "MESSAGE_TYPE_HEARTBEAT",
    # Server
    "RPCKVServer",
    "RPCKVServerConfig",
    # Client
    "RPCKVClient",
    "RPCKVClientConfig",
]
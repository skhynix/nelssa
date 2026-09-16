# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RPC protocol definitions for P/D disaggregated serving.

Defines request/response structures for communication between:
- Prefill Worker + Host CPU (RPC Server)
- Decode Worker (RPC Client)
"""

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Message types
MESSAGE_TYPE_SPARSE_ATTN = "sparse_attention"
MESSAGE_TYPE_HEARTBEAT = "heartbeat"
MESSAGE_TYPE_KV_STORE = "kv_store"
MESSAGE_TYPE_CLEANUP = "cleanup"
MESSAGE_TYPE_SLOT_FREE = "slot_free"


# ---------------------------------------------------------------------------
# Raw-bytes tensor serialization (replaces pickle for tensors)
#
# Pickling torch tensors goes through __reduce_ex__ which is noticeably slower
# than a plain memory copy for the small tensors on the RPC critical path. We
# instead serialize a compact header (dtype code + ndim + shape) followed by
# the tensor's contiguous raw bytes, and rebuild with np.frombuffer + copy.
# Requires CPU tensors; callers must .cpu() GPU tensors first.
# ---------------------------------------------------------------------------

# Stable dtype <-> single-byte code mapping. Keep in sync on both sides.
_DTYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}

# numpy dtype for each non-bfloat16 torch dtype (bfloat16 is decoded via uint16
# + view since numpy has no native bfloat16).
_TORCH_TO_NP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a CPU tensor to compact header + raw bytes.

    Header: 1B dtype code | 1B ndim | ndim*8B shape (int64, little-endian).
    The caller is responsible for ensuring `tensor` is on CPU.
    """
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    t = tensor.contiguous()
    code = _DTYPE_CODE.get(t.dtype)
    if code is None:
        raise ValueError(f"Unsupported dtype for tensor_to_bytes: {t.dtype}")
    shape = t.shape
    header = struct.pack("<BB", code, len(shape)) + struct.pack(
        f"<{len(shape)}q", *shape)
    # numpy has no bfloat16; view as uint16 to grab the raw bytes.
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.uint16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    return header + raw


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Rebuild a CPU tensor from tensor_to_bytes output."""
    code, ndim = struct.unpack_from("<BB", data, 0)
    offset = 2
    shape = struct.unpack_from(f"<{ndim}q", data, offset)
    offset += ndim * 8
    body = data[offset:]
    dtype = _CODE_TO_DTYPE[code]
    if dtype == torch.bfloat16:
        arr = np.frombuffer(body, dtype=np.uint16).reshape(shape)
        # Reinterpret uint16 bits as bfloat16; copy() makes the buffer writable.
        out = torch.from_numpy(arr.copy()).view(torch.bfloat16)
    else:
        np_dtype = _TORCH_TO_NP[dtype]
        arr = np.frombuffer(body, dtype=np_dtype).reshape(shape)
        out = torch.from_numpy(arr.copy())
    return out


@dataclass
class SparseAttentionRequest:
    """
    Decode Worker → Host CPU: Sparse attention request.

    Contains:
    - Query tensor from current decode step
    - Top-K cluster IDs selected by Decode Worker
    - Metadata for attention computation
    """
    request_id: str
    query_data: bytes  # Serialized query tensor
    topk_cluster_ids: list[int]
    seq_len: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    layer_idx: int

    def serialize(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query_data": self.query_data,
            "topk_cluster_ids": self.topk_cluster_ids,
            "seq_len": self.seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "layer_idx": self.layer_idx,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SparseAttentionRequest":
        return cls(
            request_id=data["request_id"],
            query_data=data["query_data"],
            topk_cluster_ids=data["topk_cluster_ids"],
            seq_len=data["seq_len"],
            num_heads=data["num_heads"],
            head_dim=data["head_dim"],
            num_kv_heads=data["num_kv_heads"],
            layer_idx=data["layer_idx"],
        )


@dataclass
class SparseAttentionResponse:
    """
    Host CPU → Decode Worker: Sparse attention result.

    Contains:
    - Attention output tensor
    - Success status
    - Optional error message
    """
    attention_output_data: bytes  # Serialized output tensor
    success: bool
    error_message: str | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "attention_output_data": self.attention_output_data,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SparseAttentionResponse":
        return cls(
            attention_output_data=data["attention_output_data"],
            success=data["success"],
            error_message=data.get("error_message"),
        )


@dataclass
class KVStoreNotification:
    """
    Prefill Worker → Decode Worker (via Host CPU): KV cache stored notification.

    Used to inform Decode Worker that new KV blocks are available.
    """
    request_id: str
    block_ids: list[int]
    layer_idx: int
    num_tokens: int

    def serialize(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "block_ids": self.block_ids,
            "layer_idx": self.layer_idx,
            "num_tokens": self.num_tokens,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "KVStoreNotification":
        return cls(
            request_id=data["request_id"],
            block_ids=data["block_ids"],
            layer_idx=data["layer_idx"],
            num_tokens=data["num_tokens"],
        )


@dataclass
class CleanupRequest:
    """
    Decode Worker → Host CPU: Cleanup request for a completed request.

    Used to free CPU KV buffers after decode is complete.
    """
    request_id: str

    def serialize(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "CleanupRequest":
        return cls(
            request_id=data["request_id"],
        )


@dataclass
class CleanupResponse:
    """
    Host CPU → Decode Worker: Cleanup response.
    """
    success: bool
    error_message: str | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "CleanupResponse":
        return cls(
            success=data["success"],
            error_message=data.get("error_message"),
        )


@dataclass
class SlotFreeRequest:
    """
    Decode Worker → Host CPU: Slot free notification.
    """
    request_id: str
    slot: int

    def serialize(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "slot": self.slot,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SlotFreeRequest":
        return cls(
            request_id=data["request_id"],
            slot=int(data["slot"]),
        )


@dataclass
class SlotFreeResponse:
    """
    Host CPU → Decode Worker: Slot free response.
    """
    success: bool
    error_message: str | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SlotFreeResponse":
        return cls(
            success=data["success"],
            error_message=data.get("error_message"),
        )


@dataclass
class SparseAttentionBatchRequest:
    """
    Decode Worker → Host CPU: Batched sparse attention request.

    Contains:
    - Multiple query tensors from multiple Long Requests (batched)
    - Top-K cluster IDs for each request
    - Request IDs for tracking
    - Metadata for attention computation
    """
    request_ids: list[str]           # Multiple request IDs
    query_data: bytes                # Serialized batched query tensor [num_reqs, num_heads, head_dim]
    cluster_ids_list: list           # Per-request cluster_ids
    layer_idx: int
    num_heads: int
    head_dim: int
    num_kv_heads: int                # Number of KV heads (for GQA)

    def serialize(self) -> dict[str, Any]:
        return {
            "request_ids": self.request_ids,
            "query_data": self.query_data,
            "cluster_ids_list": self.cluster_ids_list,
            "layer_idx": self.layer_idx,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SparseAttentionBatchRequest":
        return cls(
            request_ids=data["request_ids"],
            query_data=data["query_data"],
            cluster_ids_list=data["cluster_ids_list"],
            layer_idx=data["layer_idx"],
            num_heads=data["num_heads"],
            head_dim=data["head_dim"],
            num_kv_heads=data["num_kv_heads"],
        )


@dataclass
class SparseAttentionBatchResponse:
    """
    Host CPU → Decode Worker: Batched sparse attention result.

    Contains:
    - List of attention output tensors for each request
    - Success status
    - Optional error message
    """
    attention_outputs_data: list[bytes]  # List of serialized output tensors
    success: bool
    error_message: str | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "attention_outputs_data": self.attention_outputs_data,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SparseAttentionBatchResponse":
        return cls(
            attention_outputs_data=data["attention_outputs_data"],
            success=data["success"],
            error_message=data.get("error_message"),
        )
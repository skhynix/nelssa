# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RPC Client for Decode Worker in P/D disaggregated serving.

Communicates with Host CPU RPC Server for:
1. Sparse attention computation
2. KV cache status queries (optional)
"""

import asyncio
import os
import pickle
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.distributed.rpc_kv.protocol import (
    SparseAttentionRequest,
    SparseAttentionResponse,
    SparseAttentionBatchRequest,
    SparseAttentionBatchResponse,
    CleanupRequest,
    CleanupResponse,
    SlotFreeRequest,
    SlotFreeResponse,
    MESSAGE_TYPE_SPARSE_ATTN,
    MESSAGE_TYPE_HEARTBEAT,
    MESSAGE_TYPE_CLEANUP,
    MESSAGE_TYPE_SLOT_FREE,
    tensor_to_bytes,
    bytes_to_tensor,
)

logger = init_logger(__name__)


@dataclass
class RPCKVClientConfig:
    server_host: str = "localhost"
    server_port: int = 8765
    timeout: float = 30.0  # seconds


class RPCKVClient:
    # Class-level collector for per-stage RPC times (serialize/send/recv, ms).
    # Set by gpu_model_runner so the periodic ATTN-TIMING log can attribute the
    # RPC roundtrip overhead. None disables collection.
    _rpc_timing_collector: list | None = None

    def __init__(self, config: RPCKVClientConfig):
        self.config = config
        self.server_host = config.server_host
        self.server_port = config.server_port
        self.timeout = config.timeout

        # Connection state
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._connected:
            logger.warning("Already connected to RPC server")
            return

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host=self.server_host,
                    port=self.server_port,
                ),
                timeout=self.timeout,
            )
            self._connected = True
            # (1) TCP_NODELAY disables Nagle so the small header flush immediately
            # (2) Expand socket buffer size (SO_SNDBUF / SO_RCVBUF = 1MB)
            try:
                _sock = self._writer.get_extra_info('socket')
                if _sock is not None:
                    import socket as _socket
                    _sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                    _sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1 << 20)
                    _sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1 << 20)
            except Exception:
                pass
            await self._send_heartbeat()
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Timeout connecting to RPC server at "
                f"{self.server_host}:{self.server_port}"
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to RPC server: {e}"
            )


    async def disconnect(self) -> None:
        if not self._connected:
            return

        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None

        self._reader = None
        self._connected = False
        logger.info("Disconnected from RPC server")


    async def send_sparse_attention(
        self,
        request_id: str,
        query_tensor: torch.Tensor,
        topk_cluster_ids: list[int],
        layer_idx: int,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        if not self._connected:
            await self.connect()

        query_data = tensor_to_bytes(query_tensor)
        num_tokens = query_tensor.shape[0] if query_tensor.dim() > 1 else 1
        num_heads = query_tensor.shape[-2] if query_tensor.dim() > 1 else query_tensor.shape[0]
        head_dim = query_tensor.shape[-1]
        num_kv_heads = num_heads  # Simplified; adjust if needed

        request = SparseAttentionRequest(
            request_id=request_id,
            query_data=query_data,
            topk_cluster_ids=topk_cluster_ids,
            seq_len=seq_len or num_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            layer_idx=layer_idx,
        )

        message = {
            "type": MESSAGE_TYPE_SPARSE_ATTN,
            "data": request.serialize(),
        }

        response_data = await self._send_receive(message)

        # Deserialize the response (server sends pickled dict)
        response_dict = pickle.loads(response_data)
        response = SparseAttentionResponse.deserialize(response_dict)
        if not response.success:
            raise RuntimeError(
                f"Sparse attention failed: {response.error_message}"
            )
        output_tensor = bytes_to_tensor(response.attention_output_data)

        logger.debug(
            f"Sparse attention response: "
            f"req_id={request_id}, layer={layer_idx}, "
            f"output_shape={output_tensor.shape}"
        )

        return output_tensor


    async def send_sparse_attention_batch(
        self,
        request_ids: list[str],
        query_tensor: torch.Tensor,
        cluster_ids_list: list[list[int]],
        layer_idx: int,
        num_kv_heads: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Send batched sparse attention request for multiple Long Requests.

        Args:
            request_ids: List of request IDs
            query_tensor: Batched query tensor of shape [num_reqs, num_heads, head_dim]
            cluster_ids_list: List of cluster_ids for each request
            layer_idx: Layer index
            num_kv_heads: Number of KV heads (for GQA). If None, inferred from query_tensor.

        Returns:
            List of attention output tensors for each request
        """
        if not self._connected:
            await self.connect()

        num_reqs = len(request_ids)
        num_heads = query_tensor.shape[1] if query_tensor.dim() == 3 else query_tensor.shape[0]
        head_dim = query_tensor.shape[-1]

        # num_kv_heads must be provided for GQA models
        if num_kv_heads is None:
            logger.warning("num_kv_heads not provided, assuming num_heads (may be incorrect for GQA models)")
            num_kv_heads = num_heads

        query_data = tensor_to_bytes(query_tensor)

        request = SparseAttentionBatchRequest(
            request_ids=request_ids,
            query_data=query_data,
            cluster_ids_list=cluster_ids_list,
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )

        message = {
            "type": MESSAGE_TYPE_SPARSE_ATTN,
            "data": request.serialize(),
            "batch": True,  # Indicate this is a batch request
        }

        # Ensure connection before sending
        if not self._connected or self._writer is None:
            await self.connect()

        response_data = await self._send_receive(message, layer_idx=layer_idx)

        response_dict = pickle.loads(response_data)
        response = SparseAttentionBatchResponse.deserialize(response_dict)
        if not response.success:
            raise RuntimeError(
                f"Sparse attention batch failed: {response.error_message}"
            )

        # Deserialize: the server sends raw-bytes tensors. For a batched
        # (output, lse) result the list holds exactly two tensors.
        deserialized = [bytes_to_tensor(data) for data in response.attention_outputs_data]

        # If we got a (output, lse) pair, return it directly
        if len(deserialized) == 2:
            result = tuple(deserialized)  # (output, lse)
        else:
            # Legacy: list of output tensors
            result = deserialized

        logger.debug(
            f"Sparse attention batch response: "
            f"num_reqs={num_reqs}, layer={layer_idx}"
        )

        return result


    async def send_cleanup(self, request_id: str) -> bool:
        """Send cleanup request to release CPU KV buffers."""
        if not self._connected:
            await self.connect()

        cleanup_request = CleanupRequest(request_id=request_id)

        message = {
            "type": MESSAGE_TYPE_CLEANUP,
            "data": cleanup_request.serialize(),
        }

        response_data = await self._send_receive(message)
        # Deserialize the response (server sends pickled dict)
        response_dict = pickle.loads(response_data)
        response = CleanupResponse.deserialize(response_dict)

        if not response.success:
            logger.error(f"Cleanup failed: {response.error_message}")
            return False

        logger.info(f"Cleanup successful for request_id={request_id}")
        return True


    # NELSSA: Cluster Metadata Transfer
    async def send_slot_free(self, request_id: str, slot: int) -> bool:
        if not self._connected:
            await self.connect()

        slot_free_request = SlotFreeRequest(request_id=request_id, slot=slot)
        message = {
            "type": MESSAGE_TYPE_SLOT_FREE,
            "data": slot_free_request.serialize(),
        }

        response_data = await self._send_receive(message)
        response_dict = pickle.loads(response_data)
        response = SlotFreeResponse.deserialize(response_dict)

        if not response.success:
            logger.error(f"Slot free failed: {response.error_message}")
            return False

        logger.debug(f"Slot free notification sent for request_id={request_id}")
        return True


    async def _send_receive(self, message: dict, layer_idx: int | None = None) -> bytes:
        async with self._lock:
            if not self._connected or self._writer is None:
                raise ConnectionError("Not connected to RPC server")

            # Per-stage NVTX (torch.cuda.nvtx, captured by nsys) to decompose the
            # RPC roundtrip on the D-side worker thread. No-op unless NELSSA_NVTX=1.
            import os as _os
            import time as _time
            if _os.environ.get("NELSSA_NVTX", "0") == "1":
                from torch.cuda.nvtx import range_push as _push, range_pop as _pop
            else:
                _push = _pop = None

            try:
                if _push:
                    _push("[D] rpc:serialize")
                _t0 = _time.perf_counter()
                message_data = pickle.dumps(message)
                message_length = len(message_data).to_bytes(4, 'big')
                _ser_ms = (_time.perf_counter() - _t0) * 1e3
                if _pop:
                    _pop()

                if _push:
                    _push("[D] rpc:send")
                _t0 = _time.perf_counter()
                # Skip the await drain() — the attention request is small
                # (~11KB query+ids) and the socket buffer (1MB) absorbs it in
                # one write, so draining just adds an event-loop yield per layer.
                self._writer.write(message_length + message_data)
                _send_ms = (_time.perf_counter() - _t0) * 1e3
                if _pop:
                    _pop()

                if _push:
                    _push("[D] rpc:recv")
                _t0 = _time.perf_counter()
                length_data = await asyncio.wait_for(
                    self._reader.readexactly(4),
                    timeout=self.timeout,
                )
                response_length = int.from_bytes(length_data, 'big')

                response_data = await asyncio.wait_for(
                    self._reader.readexactly(response_length),
                    timeout=self.timeout,
                )
                _recv_ms = (_time.perf_counter() - _t0) * 1e3
                if _pop:
                    _pop()

                # Record the per-stage RPC times (ms) on the class-level
                # collector so _nelssa_log_attn_timings can attribute the RPC
                # roundtrip overhead (serialize / send / recv-incl-network-RTT
                # and server-side compute + response serialize).
                col = RPCKVClient._rpc_timing_collector
                if col is not None:
                    col.append((_ser_ms, _send_ms, _recv_ms))

                return response_data

            except asyncio.TimeoutError:
                self._connected = False
                raise ConnectionError("Request timeout")
            except Exception as e:
                self._connected = False
                raise ConnectionError(f"Communication error: {e}")


    async def _send_heartbeat(self) -> bool:
        message = {"type": MESSAGE_TYPE_HEARTBEAT}
        try:
            response_data = await self._send_receive(message)
            # Heartbeat response is raw bytes, not pickled dict
            response = pickle.loads(response_data)
            if isinstance(response, dict):
                return response.get("status") == "ok"
            return True
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False


    async def __aenter__(self):
        await self.connect()
        return self


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
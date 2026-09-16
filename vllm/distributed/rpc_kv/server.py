# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RPC Server for Host CPU in P/D disaggregated serving.

Runs in the Prefill Worker process and provides:
1. KV Cache Storage (from Prefill Worker's GPU)
2. Sparse Attention Computation (for Decode Worker)
"""

import asyncio
import json
import os
import pickle
import socket as _socket_mod
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import torch

if os.environ.get("NELSSA_NVTX", "0") == "1":
    from torch.cuda.nvtx import range_push as _nvtx_range_push, \
        range_pop as _nvtx_range_pop

    class _Nvtx:
        @staticmethod
        def push_range(msg, *a, **k):
            _nvtx_range_push(str(msg))

        @staticmethod
        def pop_range(*a, **k):
            _nvtx_range_pop()

    nvtx = _Nvtx()  # type: ignore
else:
    class _NvtxNoop:
        @staticmethod
        def push_range(*a, **k): pass
        @staticmethod
        def pop_range(*a, **k): pass
    nvtx = _NvtxNoop()  # type: ignore

# Verbose NELSSA debug logging (SERVER-TIMING / SERVER-STAGE aggregation). OFF
# by default = zero overhead (timing collection + per-320-layer aggregation
# skipped). Enable with NELSSA_ATTN_CORELOG=1 (same flag as cpu_attention.py).
_CORELOG = os.environ.get("NELSSA_ATTN_CORELOG", "0") == "1"

from vllm.logger import init_logger
from vllm.distributed.rpc_kv.protocol import (
    SparseAttentionRequest,
    SparseAttentionResponse,
    SparseAttentionBatchRequest,
    SparseAttentionBatchResponse,
    KVStoreNotification,
    CleanupRequest,
    CleanupResponse,
    SlotFreeRequest,
    SlotFreeResponse,
    MESSAGE_TYPE_SPARSE_ATTN,
    MESSAGE_TYPE_KV_STORE,
    MESSAGE_TYPE_HEARTBEAT,
    MESSAGE_TYPE_CLEANUP,
    MESSAGE_TYPE_SLOT_FREE,
    tensor_to_bytes,
)

logger = init_logger(__name__)


@dataclass
class RPCKVServerConfig:
    """Configuration for RPCKVServer."""
    host: str = "localhost"
    port: int = 8765
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128


class RPCKVServer:
    # Class-level collector for per-layer server-side timing:
    # (handler_ms, total_ms) where handler = pure CPU attention compute and
    # total = deserialize + handler + serialize. D-side compares its rpcrecv
    # against total to isolate the network/protocol roundtrip overhead.
    # Set by gpu_model_runner; None disables collection.
    _server_timing_collector: list | None = None

    def __init__(
        self,
        config: RPCKVServerConfig,
        sparse_attention_handler: Callable | None = None,
    ):
        self.config = config
        self.host = config.host
        self.port = config.port
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        # Custom handler (from gpu_model_runner.py)
        self.sparse_attention_handler = sparse_attention_handler

        # Server state
        self._server: asyncio.Server | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._sock_cb = None  # set in start(); applied to accepted sockets


    async def start(self) -> None:
        if self._running:
            logger.warning("RPC server is already running")
            return

        self._running = True

        # (1) TCP_NODELAY disables Nagle so the small header flush immediately
        # (2) Expand socket buffer size (SO_SNDBUF / SO_RCVBUF = 1MB)
        def _apply_low_latency_sock(sock):
            try:
                sock.setsockopt(_socket_mod.IPPROTO_TCP, _socket_mod.TCP_NODELAY, 1)
                sock.setsockopt(_socket_mod.SOL_SOCKET, _socket_mod.SO_SNDBUF, 1 << 20)
                sock.setsockopt(_socket_mod.SOL_SOCKET, _socket_mod.SO_RCVBUF, 1 << 20)
            except Exception:
                pass

        self._server = await asyncio.start_server(      # TCP Server Start
            self._handle_client,
            host=self.host,
            port=self.port,
        )
        for _s in self._server.sockets:
            _apply_low_latency_sock(_s)
        self._sock_cb = _apply_low_latency_sock

        addr = self._server.sockets[0].getsockname()
        logger.info(f"RPCKVServer started on {addr[0]}:{addr[1]}")

        # Run server in background
        async with self._server:
            await self._server.serve_forever()


    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("RPCKVServer stopped")


    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info('peername')
        client_id = f"{addr[0]}:{addr[1]}"
        logger.debug(f"New connection from {addr}")

        # Apply low-latency socket options on the accepted connection socket.
        _acc_sock = writer.get_extra_info('socket')
        if _acc_sock is not None and self._sock_cb is not None:
            self._sock_cb(_acc_sock)

        try:
            while self._running:
                #   recv_hdr = epoll wakeup + TCP propagation of the 4B length
                #   recv_body = reading message_length bytes off the socket
                nvtx.push_range("[P] rpc:recv_hdr")
                length_data = await reader.readexactly(4)
                if len(length_data) < 4:
                    nvtx.pop_range()  # end [P] rpc:recv_hdr
                    break
                message_length = int.from_bytes(length_data, 'big')
                nvtx.push_range("[P] rpc:recv_body")
                message_data = await reader.readexactly(message_length)
                nvtx.pop_range()  # end [P] rpc:recv_body
                nvtx.pop_range()  # end [P] rpc:recv_hdr

                try:
                    response = await self._route_message(message_data)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    response = self._create_error_response(str(e))

                # Send response
                response_data = pickle.dumps(response)
                response_length = len(response_data).to_bytes(4, 'big')
                # Skip await drain() — the response (~8KB output+lse) fits in
                # the 1MB socket buffer in one write; draining only adds a
                # per-layer event-loop yield.
                writer.write(response_length + response_data)

        except asyncio.CancelledError:
            logger.debug(f"Connection to {addr} cancelled")
        except (ConnectionResetError, asyncio.IncompleteReadError, BrokenPipeError):
            # Client disconnected - normal behavior after response
            logger.debug(f"Client disconnected: {addr}")
        except Exception as e:
            # Log other errors as error level
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            # Close connection gracefully
            if not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
            logger.debug(f"Connection closed: {addr}")


    async def _route_message(self, data: bytes) -> dict[str, Any]:
        try:
            nvtx.push_range("[P] rpc:loads")
            message = pickle.loads(data)
            nvtx.pop_range()  # end [P] rpc:loads
            msg_type = message.get("type")

            if msg_type == MESSAGE_TYPE_SPARSE_ATTN:
                return await self._handle_sparse_attention(message)
            elif msg_type == MESSAGE_TYPE_HEARTBEAT:
                return {"type": MESSAGE_TYPE_HEARTBEAT, "status": "ok"}
            elif msg_type == MESSAGE_TYPE_KV_STORE:
                return await self._handle_kv_store(message)
            elif msg_type == MESSAGE_TYPE_CLEANUP:
                return await self._handle_cleanup(message)
            elif msg_type == MESSAGE_TYPE_SLOT_FREE:
                return await self._handle_slot_free(message)
            else:
                return self._create_error_response(f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return self._create_error_response(str(e))


    async def _handle_sparse_attention(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            # Check if this is a batch request
            is_batch = message.get("batch", False)

            if is_batch:
                # Handle batch request
                import time as _t
                _t_total = _t.perf_counter()
                nvtx.push_range("[P] rpc:deserialize")
                batch_request = SparseAttentionBatchRequest.deserialize(message["data"])
                nvtx.pop_range()  # end [P] rpc:deserialize

                # Call batch handler
                if self.sparse_attention_handler:
                    _t_h0 = _t.perf_counter()
                    nvtx.push_range("[P] rpc:handler")
                    result = await self.sparse_attention_handler(
                        batch_request=batch_request,
                    )
                    nvtx.pop_range()  # end [P] rpc:handler
                    _handler_ms = (_t.perf_counter() - _t_h0) * 1e3
                else:
                    raise RuntimeError(
                        "No sparse attention handler registered for batch requests."
                    )

                # Handle (output, lse) tuple — serialize each tensor as raw bytes
                # (compact header + data) instead of pickle for the RPC hot path.
                nvtx.push_range("[P] rpc:serialize")
                if isinstance(result, tuple) and len(result) == 2:
                    output, lse = result
                    output_data_list = [tensor_to_bytes(output), tensor_to_bytes(lse)]
                else:
                    output_data_list = [tensor_to_bytes(t) for t in result]
                nvtx.pop_range()  # end [P] rpc:serialize

                _total_ms = (_t.perf_counter() - _t_total) * 1e3
                # Record per-layer server-side timing to attribute the D-side
                # rpcrecv into "pure CPU attention (handler)" vs "RPC overhead"
                # (deserialize + serialize + TCP roundtrip). Gated by
                # NELSSA_ATTN_CORELOG=1: the aggregation + logging is skipped
                # entirely when the flag is off (zero hot-path overhead).
                if _CORELOG:
                    col = getattr(RPCKVServer, '_server_timing_collector', None)
                    if col is not None:
                        col.append((_handler_ms, _total_ms))
                        # Log inline every 320 layers (10 decode steps x 32 layers)
                        # since the P-side execute_model may run far less often than
                        # the RPC handler (P-side idles between prefills while D-side
                        # decodes and fires attention requests every layer).
                        if len(col) >= 320:
                            import math as _m
                            handlers = sorted(s[0] for s in col)
                            totals = sorted(s[1] for s in col)
                            n = len(handlers)
                            h_mean = sum(handlers) / n
                            t_mean = sum(totals) / n
                            h_pct = (h_mean / t_mean * 100.0) if t_mean > 0 else 0.0
                            logger.info(
                                "[NELSSA][SERVER-TIMING] n=%d srv_handler: "
                                "mean=%.3f(%.0f%% of total) med=%.3f p99=%.3f | "
                                "srv_total: mean=%.3f med=%.3f p99=%.3f",
                                n, h_mean, h_pct, handlers[n // 2],
                                handlers[min(n - 1, int(_m.ceil(0.99 * n)) - 1)],
                                t_mean, totals[n // 2],
                                totals[min(n - 1, int(_m.ceil(0.99 * n)) - 1)])
                            # Per-stage breakdown of srv_handler from the engine's
                            # stage collector (deserialize / dict_loop / buf_prep /
                            # gather / fused) to find where the handler time lives.
                            stage_parts = []
                            try:
                                from vllm.v1.nelssa.cpu_attention import (
                                    RPCAttentionEngine)
                                sc = getattr(RPCAttentionEngine,
                                             '_server_stage_collector', None)
                                if sc:
                                    stage_names = ['deserialize_q',
                                                   'deserialize_ids',
                                                   'dict_loop', 'buf_prep',
                                                   'gather', 'fused']
                                    for sn in stage_names:
                                        vals = [s.get(sn) for s in sc
                                                if s.get(sn) is not None]
                                        if not vals:
                                            continue
                                        vs = sorted(vals)
                                        m = sum(vals) / len(vals)
                                        pct = (m / h_mean * 100.0
                                               ) if h_mean > 0 else 0.0
                                        stage_parts.append(
                                            f"{sn}: mean={m:.3f}({pct:.0f}%) "
                                            f"med={vs[len(vs) // 2]:.3f} "
                                            f"p99={vs[min(len(vs) - 1, int(_m.ceil(0.99 * len(vs))) - 1)]:.3f}")
                                    sc.clear()
                            except Exception:
                                pass
                            if stage_parts:
                                logger.info(
                                    "[NELSSA][SERVER-STAGE] n=%d %s", n,
                                    " | ".join(stage_parts))
                            col.clear()

                return SparseAttentionBatchResponse(
                    attention_outputs_data=output_data_list,
                    success=True,
                ).serialize()
            else:
                # Handle single request (original logic)
                request = SparseAttentionRequest.deserialize(message["data"])

                # Call custom handler (from gpu_model_runner.py)
                if self.sparse_attention_handler:
                    output_tensor = await self.sparse_attention_handler(
                        request=request,
                    )
                else:
                    raise RuntimeError(
                        "No sparse attention handler registered. "
                        "This should be set by gpu_model_runner.py"
                    )

                output_data = tensor_to_bytes(output_tensor)

                return SparseAttentionResponse(
                    attention_output_data=output_data,
                    success=True,
                ).serialize()

        except Exception as e:
            # Determine request_id for logging
            req_id = "unknown"
            try:
                if is_batch:
                    req_id = batch_request.request_ids[0] if hasattr(batch_request, 'request_ids') and batch_request.request_ids else "unknown"
                else:
                    req_id = request.request_id

                logger.error(
                    f"[NELSSA][RPC-SERVER] Attention computation FAILED: "
                    f"req_id={req_id}, error={e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
            except:
                pass

            # Return appropriate error response based on request type
            if is_batch:
                return SparseAttentionBatchResponse(
                    attention_outputs_data=[],
                    success=False,
                    error_message=str(e),
                ).serialize()
            else:
                return self._create_error_response(str(e))


    async def _handle_cleanup(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle cleanup request from Decode Worker."""
        try:
            cleanup_request = CleanupRequest.deserialize(message["data"])
            request_id = cleanup_request.request_id

            logger.info(f"[RPC Server] Received cleanup request for request_id={request_id}")

            # Call the cleanup handler (from gpu_model_runner.py)
            if self.sparse_attention_handler:
                # The handler should have a cleanup method or we call a separate cleanup function
                # For now, we assume the handler has a _cleanup_sparse_kv_cache method
                if hasattr(self.sparse_attention_handler, '__self__'):
                    runner = self.sparse_attention_handler.__self__
                    if hasattr(runner, '_cleanup_sparse_kv_cache'):
                        runner._cleanup_sparse_kv_cache(request_id)
                        logger.info(f"[RPC Server] Cleaned up CPU KV buffers for request_id={request_id}")
                        return CleanupResponse(success=True).serialize()
                    else:
                        return self._create_error_response("Runner does not have _cleanup_sparse_kv_cache method")
                else:
                    return self._create_error_response("Could not access runner from handler")
            else:
                return self._create_error_response("No sparse attention handler registered")

        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            return self._create_error_response(str(e))


    async def _handle_slot_free(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            req = SlotFreeRequest.deserialize(message["data"])
            request_id, slot = req.request_id, req.slot
            
            handler = self.sparse_attention_handler
            handler.__self__._free_nelssa_meta_slot(request_id, slot=slot)
            logger.info(f"[RPC Server][NELSSA] Freed metadata staging slot={slot} for request_id={request_id}")
            
            return SlotFreeResponse(success=True).serialize()

        except Exception as e:
            logger.error(f"Error in slot_free: {e}")
            return self._create_error_response(str(e))


    def _create_error_response(self, error_msg: str) -> dict[str, Any]:
        return SparseAttentionResponse(
            attention_output_data=b"",
            success=False,
            error_message=error_msg,
        ).serialize()


    def register_sparse_attention_handler(
        self,
        handler: Callable,
    ) -> None:
        self.sparse_attention_handler = handler
        logger.info("Registered sparse attention handler")
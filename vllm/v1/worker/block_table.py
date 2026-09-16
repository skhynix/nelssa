# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        # NELSSA: per-request compaction state for the [head | tail | generated]
        # 3-region layout. head/tail_blocks let the slot-mapping kernel apply
        # the layout every step; tail_start_pos is the block-aligned logical
        # first position of the tail region (aligned down so tail blocks have
        # no empty slots).
        self.nelssa_compacted: np.ndarray = np.zeros(max_num_reqs, dtype=np.int32)
        self.nelssa_head_blocks: np.ndarray = np.zeros(max_num_reqs, dtype=np.int32)
        self.nelssa_tail_blocks: np.ndarray = np.zeros(max_num_reqs, dtype=np.int32)
        self.nelssa_tail_start_pos: np.ndarray = np.zeros(
            max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    def clear_middle_blocks(self, row_idx: int, head_blocks: int, tail_blocks: int,
                            tail_start_pos: int = -1) -> None:
        """Rearrange block table to [head | tail | generated...], zeroing middle.
        """
        total_blocks = self.num_blocks_per_row[row_idx]
        if total_blocks == 0:
            return

        bt = self.block_table.np[row_idx]
        keep_blocks = head_blocks + tail_blocks

        if tail_start_pos < 0:
            tail_start_pos = total_blocks * self.block_size - tail_blocks * self.block_size

        # Compacted rows keep a fixed [head | tail | generated...] layout;
        # just refresh the counts and leave the table untouched.
        if self.nelssa_compacted[row_idx]:
            self.nelssa_head_blocks[row_idx] = head_blocks
            self.nelssa_tail_blocks[row_idx] = tail_blocks
            self.nelssa_tail_start_pos[row_idx] = tail_start_pos
            return

        # First-time compaction: move tail_blks to right after head, zero the rest.
        tail_start_idx = total_blocks - tail_blocks
        bt[head_blocks : head_blocks + tail_blocks] = bt[tail_start_idx:total_blocks]
        bt[keep_blocks:total_blocks] = 0
        self.num_blocks_per_row[row_idx] = keep_blocks

        self.nelssa_compacted[row_idx] = 1
        self.nelssa_head_blocks[row_idx] = head_blocks
        self.nelssa_tail_blocks[row_idx] = tail_blocks
        self.nelssa_tail_start_pos[row_idx] = tail_start_pos

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        prompt_len: torch.Tensor | None = None,
        nelssa_long_mask: torch.Tensor | None = None,
    ) -> None:
        """Compute slot mapping for all tokens.

        Args:
            num_reqs: Number of requests in the batch
            query_start_loc: Cumulative token indices per request
            positions: Absolute positions for each token
            prompt_len: Prompt length per request (for position re-mapping in decode phase)
            nelssa_long_mask: Whether each request has sparse KV enabled (for block rearrangement detection)
        """
        num_tokens = positions.shape[0]
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

        # Move prompt_len to GPU if provided
        prompt_len_gpu = None
        if prompt_len is not None:
            prompt_len_gpu = prompt_len.to(device=self.device, non_blocking=True)

        # Move nelssa_long_mask to GPU if provided
        nelssa_long_mask_gpu = None
        if nelssa_long_mask is not None:
            nelssa_long_mask_gpu = nelssa_long_mask.to(device=self.device)

        # NELSSA compaction state (0 for non-compacted rows -> standard mapping).
        _head_blk = torch.from_numpy(
            self.nelssa_head_blocks[:num_reqs]).to(self.device)
        _tail_blk = torch.from_numpy(
            self.nelssa_tail_blocks[:num_reqs]).to(self.device)
        _compacted_gpu = torch.from_numpy(
            self.nelssa_compacted[:num_reqs]).to(self.device)
        _tail_start_pos = torch.from_numpy(
            self.nelssa_tail_start_pos[:num_reqs]).to(self.device)

        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            PROMPT_LEN=prompt_len_gpu,
            IS_LONG_REQ=nelssa_long_mask_gpu,
            NELSSA_HEAD_BLOCKS=_head_blk,
            NELSSA_TAIL_BLOCKS=_tail_blk,
            NELSSA_COMPACTED=_compacted_gpu,
            NELSSA_TAIL_START_POS=_tail_start_pos,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

        # Debug: Log slot mapping per request to verify no collisions
        # slot_mapping_cpu = self.slot_mapping.gpu[:num_tokens].cpu()
        # for i in range(num_reqs):
        #     start = query_start_loc[i].item()
        #     end = query_start_loc[i + 1].item() if i + 1 < num_reqs + 1 else num_tokens
        #     block_ids = self.block_table.np[i, :self.num_blocks_per_row[i]][:10]
        #     logger.info(f"[BlockTable] req_idx={i}: block_ids={block_ids} slot_range=[{start}:{end}] slots={slot_mapping_cpu[start:end].tolist()[:10]}")


    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)
        # Ensure GPU copy is complete before compute_slot_mapping uses the data
        torch.cuda.synchronize()

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [
                cdiv(max_model_len, block_size * total_cp_world_size)
                for block_size in block_sizes
            ]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
            )
            for block_size, kernel_block_size, max_num_blocks_per_req in zip(
                block_sizes, kernel_block_sizes, max_num_blocks
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        prompt_len: torch.Tensor | None = None,
        nelssa_long_mask: torch.Tensor | None = None,
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions, prompt_len, nelssa_long_mask)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]


@triton.jit
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    PROMPT_LEN,  # [num_reqs], int32 - prompt length per request
    IS_LONG_REQ,  # [num_reqs], bool - whether sparse KV is enabled for each request
    NELSSA_HEAD_BLOCKS,  # [num_reqs], int32 - head block count per request
    NELSSA_TAIL_BLOCKS,  # [num_reqs], int32 - tail block count per request
    NELSSA_COMPACTED,  # [num_reqs], int32 - 1 if row already compacted
    NELSSA_TAIL_START_POS,  # [num_reqs], int32 - block-aligned tail start pos
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride

    # Get prompt_len for this request
    if PROMPT_LEN is not None:
        req_prompt_len = tl.load(PROMPT_LEN + req_idx).to(tl.int64)
    else:
        req_prompt_len = end_idx  # Fallback

    # NELSSA compaction state: [head | tail | generated...]. tail_start_pos is
    # block-aligned (down) so tail blocks have no empty slots; using the stored
    # value keeps this mapping aligned with clear_middle_blocks.
    head_blocks = tl.load(NELSSA_HEAD_BLOCKS + req_idx).to(tl.int64)
    tail_blocks = tl.load(NELSSA_TAIL_BLOCKS + req_idx).to(tl.int64)
    is_compacted = tl.load(NELSSA_COMPACTED + req_idx) != 0
    head_size = head_blocks * virtual_block_size
    tail_start_pos = tl.load(NELSSA_TAIL_START_POS + req_idx).to(tl.int64)

    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)

        if is_compacted:
            # Compacted: tail+generated share one contiguous run from head_size, so
            # the first generated tokens fill the last tail block's empty slots
            # (instead of jumping to a fresh block) — keeps seqlen_k free of gaps.
            tail_tokens = req_prompt_len - tail_start_pos
            is_compact_tail_or_gen = pos >= tail_start_pos
            compact_offset = pos - tail_start_pos  # 0-based within tail+gen run

            head_logical = pos
            tailgen_logical = head_size + compact_offset

            logical_slot = tl.where(is_compact_tail_or_gen,
                                    tailgen_logical, head_logical)
            logical_block_index = logical_slot // virtual_block_size
            virtual_block_offsets = logical_slot % virtual_block_size
        else:
            # Non-compacted row: standard paged mapping (prefill, short reqs).
            logical_block_index = pos // virtual_block_size
            virtual_block_offsets = pos % virtual_block_size

        block_numbers = tl.load(block_table_ptr + row_offset + logical_block_index).to(
            tl.int64
        )
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
import math

import numpy as np
import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_func
from retroinfer_kernels import (
    ThreadPool,
    WaveBufferCPU,
    batch_gemm_softmax,
    gather_copy_and_concat,
    gather_copy_and_scatter,
    gather_copy_vectors,
)
from weighted_flash_decoding import weighted_flash_decoding

from .cache import KV_Cache
from .kmeans import segment_k_means


class nelssa_cache(KV_Cache):
    """
    A class representing the KV Cache of NELSSA.
    Integrated with NelssaClient for PNM-based distributed KV caching.
    """

    def __init__(
        self,
        valid_start,  # numpy array of valid start positions for each sample in the batch
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        static_pattern_start: int,
        static_pattern_end: int,
        core: int,
        n_centroids: int,
        n_segment: int,
        pages_per_cluster: int,  # 1 cluster = 2 pages = 2 * 8 vectors
        retrieval_budget: float,
        estimation_budget: float,
        cache_ratio: float,  # ratio of cache size to sequence length
        buffer_cluster_num: int,  # number of clusters in the buffer
        use_cuda_graph: bool,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int,
        nelssa_client,  # Injected NelssaClient instance
    ) -> None:
        super().__init__(
            layer_num,
            batch_size,
            max_length,
            num_key_value_heads,
            num_heads,
            head_dim,
            dtype,
            layer_mapping,
            prefill_bsz,
            num_gpus,
            model_size,
        )

        # Use the injected NelssaClient
        self.nelssa_client = nelssa_client

        self.device_list = sorted(
            set(self.layer_mapping.values()), key=lambda x: int(x.split(":")[-1])
        )

        # constant values
        self.RSQRT_DIM = 1.0 / math.sqrt(self.head_dim)
        self.DTYPE_MIN = torch.finfo(self.dtype).min

        self.valid_start_list = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = 8
        avg_cluster_size = pages_per_cluster * self.page_size
        self.UPDATE_SEGMENT = 1024  # update segment size
        self.UPDATE_CENTROIDS = max(
            round(self.UPDATE_SEGMENT / avg_cluster_size) // 8 * 8, 8
        )  # must be divisible by 8
        self.UPDATE_NPROBE = max(
            round(self.UPDATE_CENTROIDS * retrieval_budget), 1
        )  # update retrieve zone size per segment
        self.UPDATE_ES = max(
            round(self.UPDATE_CENTROIDS * estimation_budget), 1
        )  # update estimation zone size per segment

        # whether to build index when prefilling, update index when decoding
        self.input_length = self.max_length - max_new_length
        actual_gen_len = (
            max_new_length - 1
        )  # exclude the first token generated during prefilling
        assert actual_gen_len >= 0, (
            f"Decoding generation length({actual_gen_len}) should be larger than or equal to 0"
        )
        if self.input_length <= 0:
            raise ValueError(
                f"input length({self.input_length}) should be larger than 0"
            )
        elif self.input_length < self.static_pattern_total + self.UPDATE_SEGMENT:
            # input length is too short, no need to build index during prefilling
            self.build_index_when_prefilling = False
            # update index when decoding, depends on whether input + output length exceed UPDATE_SEGMENT
            self.will_update_index = (
                self.input_length - self.static_pattern_total + actual_gen_len
            ) > self.UPDATE_SEGMENT
            # set steady zone size, cpu kv cache size and index update parameters
            if self.will_update_index:
                self.static_stride = self.static_pattern_total + self.UPDATE_SEGMENT
                self.list_stride = (
                    (self.input_length - self.static_pattern_total + actual_gen_len - 1)
                    // self.UPDATE_SEGMENT
                ) * self.UPDATE_SEGMENT
                self.n_centroids_new = (
                    (self.input_length - self.static_pattern_total + actual_gen_len - 1)
                    // self.UPDATE_SEGMENT
                ) * self.UPDATE_CENTROIDS
                self.nprobe_new = (
                    (self.input_length - self.static_pattern_total + actual_gen_len - 1)
                    // self.UPDATE_SEGMENT
                ) * self.UPDATE_NPROBE
            else:
                # fall back to full attention, all KV stores in steady zone
                self.static_stride = self.input_length + actual_gen_len
                self.list_stride = 0
                self.n_centroids_new = 0
                self.nprobe_new = 0
        else:
            self.build_index_when_prefilling = True
            # update index when decoding, depends on whether output length exceed UPDATE_SEGMENT
            self.will_update_index = actual_gen_len > self.UPDATE_SEGMENT
            # set steady zone size, cpu kv cache size and index update parameters
            if self.will_update_index:
                self.static_stride = self.UPDATE_SEGMENT + self.static_pattern_total
                self.list_stride = (
                    ((actual_gen_len - 1) // self.UPDATE_SEGMENT) * self.UPDATE_SEGMENT
                    + self.input_length
                    - self.static_pattern_total
                )
                self.n_centroids_new = (
                    (actual_gen_len - 1) // self.UPDATE_SEGMENT
                ) * self.UPDATE_CENTROIDS
                self.nprobe_new = (
                    (actual_gen_len - 1) // self.UPDATE_SEGMENT
                ) * self.UPDATE_NPROBE
            else:
                self.static_stride = actual_gen_len + self.static_pattern_total
                self.list_stride = self.input_length - self.static_pattern_total
                self.n_centroids_new = 0
                self.nprobe_new = 0

        # steady zone keys & values
        self.steady_zone_keys = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.static_stride, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            )
            for ldx in range(self.layer_num)
        ]
        self.steady_zone_values = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.static_stride, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            )
            for ldx in range(self.layer_num)
        ]

        # index parameters
        self.n_segment = n_segment
        self.n_centroids = n_centroids if self.build_index_when_prefilling else 0
        assert self.n_centroids % math.lcm(8, self.n_segment) == 0, (
            f"n_centroids({self.n_centroids}) should be divisible by LCM of 8 and n_segment({self.n_segment})"
        )
        # retrieve zone size (count by clusters)
        self.nprobe = max(round(self.n_centroids * retrieval_budget), 1)
        self.nprobe = min(self.nprobe, self.n_centroids)
        print("!! (self.n_centroids", self.n_centroids)
        print("!! retrieval_budget", retrieval_budget)
        print("!! self.nprobe", self.nprobe)
        # estimation zone size (count by clusters)
        self.es_cluster_num = min(
            round(self.n_centroids * estimation_budget), self.n_centroids - self.nprobe
        )
        # retrieve zone + estimation zone size
        self.max_compute_cluster_num = self.es_cluster_num + self.nprobe
        assert self.max_compute_cluster_num <= self.n_centroids, (
            f"max_compute_cluster_num({self.max_compute_cluster_num}) should <= n_centroids({self.n_centroids})"
        )
        print(
            f"Initial n_centroids: {self.n_centroids}, nprobe: {self.nprobe}, es_cluster_num: {self.es_cluster_num}"
        )

        # CUDA graphs
        self.use_cuda_graph = use_cuda_graph
        if self.will_update_index:  # need update when decoding
            if self.use_cuda_graph:
                print(
                    "Index will be updated during decoding, so CUDA Graph will be disabled."
                )
            self.use_cuda_graph = False
        elif (
            not self.build_index_when_prefilling
        ):  # not build index during prefilling and not update during decoding
            if self.use_cuda_graph:
                print(
                    "Input + output length too small, fall back to full attention, so CUDA Graph will be disabled."
                )
            self.use_cuda_graph = False
        if self.use_cuda_graph:
            self.topk_cudagraphs = [
                torch.cuda.CUDAGraph() for _ in range(self.layer_num)
            ]
            if self.es_cluster_num > 0:
                self.es_cudagraphs = [
                    torch.cuda.CUDAGraph() for _ in range(self.layer_num)
                ]
            self.attn_cudagraphs = [
                torch.cuda.CUDAGraph() for _ in range(self.layer_num)
            ]
            self.update_cudagraphs = [
                torch.cuda.CUDAGraph() for _ in range(self.layer_num)
            ]

        # calculate the GPU block cache size and compute buffer size (count by pages)
        cache_cluster_num = (
            round((self.n_centroids + self.n_centroids_new) * cache_ratio)
            if cache_ratio > 0.0
            else (self.nprobe + self.nprobe_new) * 3
        )
        self.cache_size = cache_cluster_num * pages_per_cluster
        self.buffer_size = (
            max(buffer_cluster_num, (self.nprobe + self.nprobe_new) * 4)
            * pages_per_cluster
        )
        print(f"Cache pages: {self.cache_size}, Buffer pages: {self.buffer_size}")

        # whether to pre-allocate GPU cache and buffer before prefilling
        self.allocated = self.pre_allocate_decision()

        # initialize thread pool
        self.thread_pool = ThreadPool(core)
        thread_pool_pointer = self.thread_pool.get()
        # initialize the Wave Buffer
        self.wave_buffer = [
            WaveBufferCPU(
                self.batch_size,
                self.kv_head,
                self.head_dim,
                self.nprobe,
                self.nprobe_new,
                self.page_size,
                self.n_centroids + self.n_centroids_new,
                self.buffer_size,
                self.cache_size,
                core,
                thread_pool_pointer,
            )
            for _ in range(self.layer_num)
        ]

        # pin memory for hit cluster indices (unit == page)
        self.hit_unit_idices = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes_cumsum = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_num_units = [
            torch.zeros(
                (self.batch_groups), dtype=torch.int32, pin_memory=True
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory for missing cluster indices (unit == page)
        self.miss_unit_idices = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes_cumsum = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_num_units = [
            torch.zeros(
                (self.batch_groups), dtype=torch.int32, pin_memory=True
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory for cache update cluster indices (unit == page)
        self.update_buffer_indices = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_unit_sizes = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_cache_indices = [
            torch.zeros(
                (self.batch_groups, self.buffer_size),
                dtype=torch.int32,
                pin_memory=True,
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_num_units = [
            torch.zeros(
                (self.batch_groups), dtype=torch.int32, pin_memory=True
            ).contiguous()
            for _ in range(self.layer_num)
        ]
        # store searched TopK cluster IDs
        self.cluster_ids = torch.empty(
            (self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True
        ).contiguous()

        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_indices(
                self.hit_unit_idices[ldx],
                self.hit_unit_sizes[ldx],
                self.hit_unit_sizes_cumsum[ldx],
                self.hit_num_units[ldx],
                self.miss_unit_idices[ldx],
                self.miss_unit_sizes[ldx],
                self.miss_unit_sizes_cumsum[ldx],
                self.miss_num_units[ldx],
                self.update_buffer_indices[ldx],
                self.update_unit_sizes[ldx],
                self.update_cache_indices[ldx],
                self.update_num_units[ldx],
                self.cluster_ids,
            )

        if self.allocated:  # allocate GPU block cache and meta index
            self.cache_keys, self.cache_values = [], []
            self.centroids, self.value_sum, self.centroids_mask, self.cluster_size = (
                [],
                [],
                [],
                [],
            )
            for ldx in range(self.layer_num):
                self.cache_keys.append(
                    torch.zeros(
                        (
                            self.batch_size,
                            self.kv_head,
                            self.cache_size,
                            self.page_size,
                            self.head_dim,
                        ),
                        dtype=self.dtype,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
                self.cache_values.append(
                    torch.zeros(
                        (
                            self.batch_size,
                            self.kv_head,
                            self.cache_size,
                            self.page_size,
                            self.head_dim,
                        ),
                        dtype=self.dtype,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
                self.centroids.append(
                    torch.zeros(
                        (
                            self.batch_size * self.kv_head,
                            self.n_centroids,
                            self.head_dim,
                        ),
                        dtype=self.dtype,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
                self.value_sum.append(
                    torch.zeros(
                        (
                            self.batch_size * self.kv_head,
                            self.n_centroids,
                            self.head_dim,
                        ),
                        dtype=self.dtype,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
                self.centroids_mask.append(
                    torch.zeros(
                        (self.batch_size * self.kv_head, self.n_centroids),
                        dtype=torch.bool,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
                self.cluster_size.append(
                    torch.zeros(
                        (self.batch_size * self.kv_head, self.n_centroids),
                        dtype=self.dtype,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
        else:  # allocate meta index in CPU, will move to GPU after prefilling
            self.centroids = [
                torch.zeros(
                    (self.batch_size * self.kv_head, self.n_centroids, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                ).contiguous()
                for ldx in range(self.layer_num)
            ]
            self.value_sum = [
                torch.zeros(
                    (self.batch_size * self.kv_head, self.n_centroids, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                ).contiguous()
                for ldx in range(self.layer_num)
            ]
            self.centroids_mask = [
                torch.zeros(
                    (self.batch_size * self.kv_head, self.n_centroids),
                    dtype=torch.bool,
                    device="cpu",
                ).contiguous()
                for ldx in range(self.layer_num)
            ]
            self.cluster_size = [
                torch.zeros(
                    (self.batch_size * self.kv_head, self.n_centroids),
                    dtype=self.dtype,
                    device="cpu",
                ).contiguous()
                for ldx in range(self.layer_num)
            ]

        # layer-share cpu pin memory, transfer gpu keys & values to cpu for segmented clustering
        if self.build_index_when_prefilling:
            self.offload_keys = torch.empty(
                (
                    self.prefill_bsz * self.kv_head,
                    self.input_length - self.static_pattern_total,
                    self.head_dim,
                ),
                dtype=self.dtype,
                pin_memory=True,
            ).contiguous()
            self.offload_values = torch.empty(
                (
                    self.prefill_bsz * self.kv_head,
                    self.input_length - self.static_pattern_total,
                    self.head_dim,
                ),
                dtype=self.dtype,
                pin_memory=True,
            ).contiguous()

        # layer-share cpu pin memory, offload update keys & values to cpu for segmented clustering
        if self.will_update_index:
            self.offload_update_keys = torch.empty(
                (self.batch_size * self.kv_head, self.UPDATE_SEGMENT, self.head_dim),
                dtype=self.dtype,
                pin_memory=True,
            ).contiguous()
            self.offload_update_values = torch.empty(
                (self.batch_size * self.kv_head, self.UPDATE_SEGMENT, self.head_dim),
                dtype=self.dtype,
                pin_memory=True,
            ).contiguous()

        # allocate cpu pin memory to store organized keys & values
        self.list_keys, self.list_values = [], []
        for _ in range(self.layer_num):
            self.list_keys.append(
                torch.empty(
                    (self.batch_size, self.kv_head, self.list_stride, self.head_dim),
                    dtype=self.dtype,
                    pin_memory=True,
                ).contiguous()
            )
            self.list_values.append(
                torch.empty(
                    (self.batch_size, self.kv_head, self.list_stride, self.head_dim),
                    dtype=self.dtype,
                    pin_memory=True,
                ).contiguous()
            )

        # set keys & values pointers in the wave buffer
        for ldx in range(self.layer_num):
            if self.build_index_when_prefilling:
                self.wave_buffer[ldx].set_kv(
                    self.list_keys[ldx],
                    self.list_values[ldx],
                    self.offload_keys,
                    self.offload_values,
                )
            elif self.will_update_index:
                self.wave_buffer[ldx].set_kv(
                    self.list_keys[ldx],
                    self.list_values[ldx],
                    self.offload_update_keys,
                    self.offload_update_values,
                )
            else:
                self.placeholder = torch.empty(
                    (self.kv_head, 0, self.head_dim), dtype=self.dtype, pin_memory=True
                )
                self.wave_buffer[ldx].set_kv(
                    self.list_keys[ldx],
                    self.list_values[ldx],
                    self.placeholder,
                    self.placeholder,
                )

        # create multi-streams and events for async offloading
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        for device_idx in self.device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()

        # set decoding attention function
        self.attn_func = self.dense_attention

    def pre_allocate_decision(self):
        """Decide whether to pre-allocate GPU cache and buffers before prefilling"""
        # estimate GPU memory consumption for cache and buffers
        self.esitimate_gpu_memory = (
            2
            * self.layer_num
            * self.batch_size
            * self.kv_head
            * (self.cache_size * self.page_size + self.n_centroids + self.static_stride)
            * self.head_dim
            * 2
        )
        self.esitimate_gpu_memory += (
            2
            * self.batch_size
            * self.kv_head
            * (self.buffer_size * self.page_size + self.static_stride)
            * self.head_dim
            * 2
        )
        self.esitimate_gpu_memory += (
            2 * self.batch_size * self.kv_head * self.es_cluster_num * self.head_dim * 2
        )
        self.esitimate_gpu_memory += (
            6 * self.batch_size * self.kv_head * self.group_size * self.n_centroids * 2
        )
        self.esitimate_gpu_memory /= 1024 * 1024 * 1024
        # print(f"Estimate GPU memory consumption for cache and buffers: {self.esitimate_gpu_memory:.4f} GB")
        return self.free_memory > self.esitimate_gpu_memory * 1.5

    def allocate_computation_buffer(self):
        """Allocate layer-share buffers, dict for different GPUs"""
        (
            self.gemm_o_dict,
            self.softmax_o_dict,
            self.norm_dict,
            self.sum_dict,
            self.dist_dict,
        ) = ({}, {}, {}, {}, {})
        self.cI_dict, self.cV_dict = {}, {}
        self.es_centroids_dict, self.es_value_sum_dict, self.es_cluster_size_dict = (
            {},
            {},
            {},
        )
        (
            self.execution_buffer_keys_dict,
            self.execution_buffer_values_dict,
            self.valid_lengths_dict,
        ) = ({}, {}, {})
        self.static_len_tensor_dict = {}
        if self.use_cuda_graph:
            self.query_buffer_dict = {}
            self.es_out_dict, self.es_lse_dict = {}, {}
            self.attn_out_dict = {}

        for device_idx in self.device_list:
            # for batch_gemm_softmax kernel
            self.gemm_o_dict[device_idx] = torch.zeros(
                (self.batch_size, self.kv_head, self.group_size, self.n_centroids),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()
            self.softmax_o_dict[device_idx] = torch.zeros(
                (self.batch_size * self.kv_head, self.group_size, self.n_centroids),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()
            self.norm_dict[device_idx] = torch.zeros(
                (
                    self.batch_size * self.kv_head,
                    self.group_size,
                    (self.n_centroids + 256 - 1) // 256,
                ),
                device=device_idx,
                dtype=torch.float32,
            ).contiguous()
            self.sum_dict[device_idx] = torch.zeros(
                (
                    self.batch_size * self.kv_head,
                    self.group_size,
                    (self.n_centroids + 256 - 1) // 256,
                ),
                device=device_idx,
                dtype=torch.float32,
            ).contiguous()
            self.dist_dict[device_idx] = torch.zeros(
                (self.batch_size * self.kv_head, self.n_centroids),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()

            # for topk
            self.cI_dict[device_idx] = torch.zeros(
                (self.batch_size * self.kv_head, self.max_compute_cluster_num),
                device=device_idx,
                dtype=torch.int64,
            ).contiguous()
            self.cV_dict[device_idx] = torch.zeros(
                (self.batch_size * self.kv_head, self.max_compute_cluster_num),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()

            # estimation zone
            self.es_centroids_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.es_value_sum_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.es_cluster_size_dict[device_idx] = torch.zeros(
                (self.batch_groups, 1, 1, self.es_cluster_num),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()

            # execution buffer
            self.execution_buffer_keys_dict[device_idx] = torch.zeros(
                (
                    self.batch_groups,
                    self.buffer_size * self.page_size + self.static_stride,
                    1,
                    self.head_dim,
                ),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.execution_buffer_values_dict[device_idx] = torch.zeros(
                (
                    self.batch_groups,
                    self.buffer_size * self.page_size + self.static_stride,
                    1,
                    self.head_dim,
                ),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.valid_lengths_dict[device_idx] = torch.zeros(
                (self.batch_groups), dtype=torch.int32, device=device_idx
            ).contiguous()
            self.static_len_tensor_dict[device_idx] = torch.tensor(
                self.static_pattern_total, dtype=torch.int32, device=device_idx
            )

            # allocate buffers used when enable CUDA graphs
            if self.use_cuda_graph:
                self.query_buffer_dict[device_idx] = torch.zeros(
                    (self.batch_groups, 1, self.group_size, self.head_dim),
                    dtype=self.dtype,
                    device=device_idx,
                ).contiguous()
                if self.es_cluster_num > 0:
                    self.es_out_dict[device_idx] = torch.zeros(
                        (self.batch_groups, 1, self.group_size, self.head_dim),
                        dtype=self.dtype,
                        device=device_idx,
                    ).contiguous()
                    self.es_lse_dict[device_idx] = torch.zeros(
                        (self.batch_groups, self.group_size, 1),
                        dtype=torch.float32,
                        device=device_idx,
                    ).contiguous()
                else:
                    self.es_out_dict[device_idx] = None
                    self.es_lse_dict[device_idx] = None
                self.attn_out_dict[device_idx] = torch.zeros(
                    (self.batch_size, 1, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=device_idx,
                ).contiguous()

        self.execution_stride = self.buffer_size * self.page_size + self.static_stride

        # point to the buffer of current layer's device
        self.cI = self.cI_dict[self.layer_mapping[str(0)]]
        self.static_len_tensor = self.static_len_tensor_dict[self.layer_mapping[str(0)]]
        if self.use_cuda_graph:
            self.query_buffer = self.query_buffer_dict[self.layer_mapping[str(0)]]
            self.attn_out = self.attn_out_dict[self.layer_mapping[str(0)]]
        else:
            self.gemm_o = self.gemm_o_dict[self.layer_mapping[str(0)]]
            self.softmax_o = self.softmax_o_dict[self.layer_mapping[str(0)]]
            self.norm = self.norm_dict[self.layer_mapping[str(0)]]
            self.sum = self.sum_dict[self.layer_mapping[str(0)]]
            self.dist = self.dist_dict[self.layer_mapping[str(0)]]
            self.cV = self.cV_dict[self.layer_mapping[str(0)]]
            self.es_centroids = self.es_centroids_dict[self.layer_mapping[str(0)]]
            self.es_value_sum = self.es_value_sum_dict[self.layer_mapping[str(0)]]
            self.es_cluster_size = self.es_cluster_size_dict[self.layer_mapping[str(0)]]
            self.execution_buffer_keys = self.execution_buffer_keys_dict[
                self.layer_mapping[str(0)]
            ]
            self.execution_buffer_values = self.execution_buffer_values_dict[
                self.layer_mapping[str(0)]
            ]
            self.valid_lengths = self.valid_lengths_dict[self.layer_mapping[str(0)]]

    def prepare_cache(self):
        """Ensure GPU cache and buffers are allocated before decoding"""
        if self.build_index_when_prefilling:
            # sync the last batch of the last layer
            torch.cuda.synchronize()
            self.wave_buffer[self.layer_num - 1].construction_sync()
            # clear temp memory
            self.clusters_cpu, self.cluster_size_cpu = None, None
            self.temp_keys, self.temp_values = None, None
            torch.cuda.empty_cache()

        if not self.allocated:  # allocate GPU cache and buffers after prefilling
            self.cache_keys, self.cache_values = [], []
            for ldx in range(self.layer_num):
                self.cache_keys.append(
                    torch.zeros(
                        (
                            self.batch_size,
                            self.kv_head,
                            self.cache_size,
                            self.page_size,
                            self.head_dim,
                        ),
                        dtype=self.dtype,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
                self.cache_values.append(
                    torch.zeros(
                        (
                            self.batch_size,
                            self.kv_head,
                            self.cache_size,
                            self.page_size,
                            self.head_dim,
                        ),
                        dtype=self.dtype,
                        device=self.layer_mapping[str(ldx)],
                    ).contiguous()
                )
                # move meta index to GPU
                self.centroids[ldx] = (
                    self.centroids[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                )
                self.value_sum[ldx] = (
                    self.value_sum[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids_mask[ldx] = (
                    self.centroids_mask[ldx]
                    .to(self.layer_mapping[str(ldx)])
                    .contiguous()
                )
                self.cluster_size[ldx] = (
                    self.cluster_size[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                )
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()

    def prefill_update_kv_cache(
        self, query_states, key_states, value_states, layer_idx, start_bdx
    ):
        """
        Update the key & value cache per layer during prefilling.
        Args:
            query_states: [bsz, seq_len, head_num, head_dim]
            key_states: [bsz, seq_len, group_num, head_dim]
            value_states: [bsz, seq_len, group_num, head_dim]
            layer_idx: layer index
            start_bdx: start batch index
        """
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz <= self.prefill_bsz, (
            f"Prefilling batch size ({bsz}) should <= {self.prefill_bsz}."
        )
        assert seq_len <= self.input_length, (
            f"seq_len({seq_len}) should <= input_length({self.input_length})"
        )
        # assert group_num == self.kv_head, f"kv_head({self.kv_head}) should equal to group_num({group_num})"
        # assert head_dim == self.head_dim, f"head_dim({head_dim}) should equal to self.head_dim({self.head_dim})"

        valid_start = self.valid_start_list[start_bdx]

        if self.build_index_when_prefilling:
            # sync for the previous layer and batch finish their page organization
            if layer_idx > 0:
                self.wave_buffer[layer_idx - 1].construction_sync()
            elif start_bdx > 0:  # layer_idx == 0
                self.wave_buffer[self.layer_num - 1].construction_sync()

            # store in `self` to avoid deleting when async offload to CPU, shape: (bsz*group_num, seq_len, dim)
            self.temp_keys = (
                key_states[
                    :,
                    valid_start + self.static_pattern_start : seq_len
                    - self.static_pattern_end,
                    :,
                    :,
                ]
                .transpose(1, 2)
                .reshape(bsz * self.kv_head, -1, self.head_dim)
                .contiguous()
            )
            self.temp_values = (
                value_states[
                    :,
                    valid_start + self.static_pattern_start : seq_len
                    - self.static_pattern_end,
                    :,
                    :,
                ]
                .transpose(1, 2)
                .reshape(bsz * self.kv_head, -1, self.head_dim)
                .contiguous()
            )
            self.mainevents[self.layer_mapping[str(layer_idx)]].record()

            # async offload keys & values to CPU
            valid_length = seq_len - self.static_pattern_total - valid_start
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(layer_idx)]].wait()
                if valid_length == self.offload_keys.shape[1]:
                    self.offload_keys[: bsz * self.kv_head, :, :].copy_(
                        self.temp_keys, non_blocking=True
                    )
                    self.offload_values[: bsz * self.kv_head, :, :].copy_(
                        self.temp_values, non_blocking=True
                    )
                else:  # loop to preserve pinned for fast copy
                    for i in range(bsz * self.kv_head):
                        self.offload_keys[i, :valid_length, :].copy_(
                            self.temp_keys[i], non_blocking=True
                        )
                        self.offload_values[i, :valid_length, :].copy_(
                            self.temp_values[i], non_blocking=True
                        )
                self.copyevents[self.layer_mapping[str(layer_idx)]].record()

            # copy steady zone KV
            end_bdx = start_bdx + bsz
            self.steady_zone_keys[layer_idx][
                start_bdx:end_bdx, :, : self.static_pattern_start, :
            ] = key_states[
                :, valid_start : valid_start + self.static_pattern_start, :, :
            ].transpose(1, 2)
            self.steady_zone_keys[layer_idx][
                start_bdx:end_bdx,
                :,
                self.static_pattern_start : self.static_pattern_total,
                :,
            ] = key_states[
                :, seq_len - self.static_pattern_end : seq_len, :, :
            ].transpose(1, 2)
            self.steady_zone_values[layer_idx][
                start_bdx:end_bdx, :, : self.static_pattern_start, :
            ] = value_states[
                :, valid_start : valid_start + self.static_pattern_start, :, :
            ].transpose(1, 2)
            self.steady_zone_values[layer_idx][
                start_bdx:end_bdx,
                :,
                self.static_pattern_start : self.static_pattern_total,
                :,
            ] = value_states[
                :, seq_len - self.static_pattern_end : seq_len, :, :
            ].transpose(1, 2)

            # compute key mean, shape (bsz*group_num, 1, head_dim)
            mean_key = torch.mean(self.temp_keys, dim=1, keepdim=True)

            # segmented clustering
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=self.temp_keys - mean_key,  # centering to 0
                value=self.temp_values,
                num_centroids=self.n_centroids,
                num_segments=self.n_segment,
            )

            # PNM Server Offloading
            # k_cache: (bsz, kv_head, seq_len, dim) -> we need to organize it by clusters first or send raw?
            # According to the plan, the PNM server expects organized cache.
            # For now, we send the raw key/value states and the cluster info so the server can organize it.
            # Based on NelssaClient.send_kv_cache signature:
            # send_kv_cache(layer_idx, k_cache_tensor, v_cache_tensor, size_cache_tensor, current_n_clusters)

            # We need k_cache_tensor and v_cache_tensor to be [bsz, kv_heads, seq_len, dim]
            # and size_cache_tensor to be [bsz * kv_heads, n_clusters]

            # Prepare tensors for offloading
            # temp_keys is [bsz * kv_head, seq_len, dim], reshape to [bsz, kv_head, seq_len, dim]
            # k_offload = (
            #     self.temp_keys.view(bsz, self.kv_head, -1, self.head_dim)
            #     .to("cpu")
            #     .contiguous()
            # )
            # v_offload = (
            #     self.temp_values.view(bsz, self.kv_head, -1, self.head_dim)
            #     .to("cpu")
            #     .contiguous()
            # )
            # s_offload = _cluster_size.to(
            #     "cpu"
            # ).contiguous()  # [bsz * kv_head, n_centroids]

            # self.nelssa_client.send_kv_cache(
            #     layer_idx=layer_idx,
            #     k_cache_tensor=k_offload,
            #     v_cache_tensor=v_offload,
            #     size_cache_tensor=s_offload,
            #     current_n_clusters=self.n_centroids,
            # )

            # copy meta index
            self.centroids[layer_idx][
                start_bdx * self.kv_head : end_bdx * self.kv_head, :, :
            ].copy_(_centroids + mean_key)  # (bsz*group_num, n_centroids, dim)
            self.value_sum[layer_idx][
                start_bdx * self.kv_head : end_bdx * self.kv_head, :, :
            ].copy_(_value_sum)  # (bsz*group_num, n_centroids, dim)
            self.centroids_mask[layer_idx][
                start_bdx * self.kv_head : end_bdx * self.kv_head, :
            ].copy_(_cluster_size == 0)  # (bsz*group_num, n_centroids)
            self.cluster_size[layer_idx][
                start_bdx * self.kv_head : end_bdx * self.kv_head, :
            ].copy_(_cluster_size.to(self.dtype))  # (bsz*group_num, n_centroids)

            # cluster results will be used to organize the offload KV cache
            self.cluster_size_cpu = (
                _cluster_size.cpu().contiguous()
            )  # (bsz*group_num, n_centroids)
            self.clusters_cpu = (
                _clusters.cpu().contiguous()
            )  # (bsz*group_num, n_centroids, max_cluster_size)
        else:  # do not build index during prefilling
            assert valid_start == 0, (
                "Requests in the same batch should have the same length."
            )
            end_bdx = start_bdx + bsz
            # copy input KV to steady zone
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(
                key_states.transpose(1, 2)
            )
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(
                value_states.transpose(1, 2)
            )

        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len

            if self.build_index_when_prefilling:
                if self.use_cuda_graph:
                    self.attn_func = self.sparse_attention_with_cudagraph
                else:
                    self.attn_func = self.sparse_attention
            else:
                self.static_pattern_total = seq_len

        return (
            key_states[:, valid_start:, :, :],
            value_states[:, valid_start:, :, :],
        )  # ignore mask tokens, shape: (bsz, seq_len, kv_head, dim)

    def sync(self, layer_idx, start_bdx):
        """Wait async offloading on copystream -> organize KV on wave buffer"""
        if self.build_index_when_prefilling:
            # wait for offload finish
            self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()
            # async organize kv
            self.wave_buffer[layer_idx].async_construction(
                self.clusters_cpu,  # (bsz*group_num, n_centroids, max_cluster_size)
                self.cluster_size_cpu,  # (bsz*group_num, n_centroids)
                start_bdx,
            )
            self.wave_buffer[layer_idx].construction_sync()
            if start_bdx == self.batch_size - 1 * self.prefill_bsz:
                self._send_layer_to_nelssa(layer_idx)

    def _send_layer_to_nelssa(self, layer_idx: int):
        try:
            k_tensor_full = self.list_keys[
                layer_idx
            ]  # (batch_size, kv_head, input_length_update, head_dim) = (1, 8, 3778, 128)
            v_tensor_full = self.list_values[layer_idx]
            s_tensor_gpu = self.cluster_size[
                layer_idx
            ]  # (batch_size*kv_head, n_centroids) = (8, 256)

            s_valid = s_tensor_gpu.to(device="cpu", dtype=torch.int32).contiguous()
            # 일부 클러스터는 비어있기 때문에, 실제로 사용 중인 클러스터(actual_clsters)만 전송
            # current_n을 실제 tensor length로 맞춤... s_valid shape: [KV_Heads, Actual_Clusters]

            k_valid = k_tensor_full.to(
                device="cpu", dtype=self.list_keys[layer_idx].dtype
            ).contiguous()
            v_valid = v_tensor_full.to(
                device="cpu", dtype=self.list_keys[layer_idx].dtype
            ).contiguous()
            # k_flat = k_valid.view(-1)
            # v_flat = v_valid.view(-1)
            # s_flat = s_valid.view(-1)
            # if layer_idx == 0:
            #     non_zero_cnt = torch.count_nonzero(k_flat).item()
            #     msg = f"[send l0] elements: {k_flat.numel()}, Non-zero: {non_zero_cnt}, mean: {k_flat.mean().item():.4f}"
            #     print(msg, flush=True)
            #     if non_zero_cnt == 0:
            #         print("@!#$@$%@ critical warning: key tensor is empty", flush=True)
            #     print(f"[Send L0] Actual Clusters: {current_n} (Config: {self.n_centroids})", flush=True)
            #     print(f"[Send L0] Size Bytes: {s_flat.nbytes} (Expected: {8 * current_n * 4})", flush=True)
            # currnet_n까지 함께 실어서 rdma 전송
            if layer_idx == 0:
                print("k_cache_tensor.shape : ", k_valid.shape)
            self.nelssa_client.send_kv_cache(
                layer_idx, k_valid, v_valid, s_valid, self.n_centroids
            )
            print("k_tensor_full shape : ", k_tensor_full.shape)
            print("v_tensor_full shape : ", v_tensor_full.shape)
            print("s_tensor_gpu shape  : ", s_tensor_gpu.shape)
        except Exception as e:
            print(f"[NELSSA] Send Error: {e}", flush=True)
            raise

    def _update_kv_cache(self):
        """Update KV cache when generate tokens exceed UPDATE_SEGMENT"""
        self.nprobe += self.UPDATE_NPROBE
        self.cluster_ids = torch.empty(
            (self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True
        ).contiguous()

        for ldx in range(self.layer_num):
            torch.cuda.set_device(self.layer_mapping[str(ldx)])
            # extract update segment, shape: (batch_size*kv_head, UPDATE_SEGMENT, head_dim)
            update_keys = (
                self.steady_zone_keys[ldx][
                    :,
                    :,
                    self.static_pattern_start : self.static_pattern_total
                    - self.static_pattern_end,
                    :,
                ]
                .clone()
                .reshape(self.batch_groups, self.UPDATE_SEGMENT, self.head_dim)
                .contiguous()
            )
            update_values = (
                self.steady_zone_values[ldx][
                    :,
                    :,
                    self.static_pattern_start : self.static_pattern_total
                    - self.static_pattern_end,
                    :,
                ]
                .clone()
                .reshape(self.batch_groups, self.UPDATE_SEGMENT, self.head_dim)
                .contiguous()
            )
            self.mainevents[self.layer_mapping[str(ldx)]].record()

            # move local window
            self.steady_zone_keys[ldx][
                :,
                :,
                self.static_pattern_start : self.static_pattern_start
                + self.static_pattern_end,
                :,
            ] = self.steady_zone_keys[ldx][
                :,
                :,
                self.static_pattern_total
                - self.static_pattern_end : self.static_pattern_total,
                :,
            ]
            self.steady_zone_values[ldx][
                :,
                :,
                self.static_pattern_start : self.static_pattern_start
                + self.static_pattern_end,
                :,
            ] = self.steady_zone_values[ldx][
                :,
                :,
                self.static_pattern_total
                - self.static_pattern_end : self.static_pattern_total,
                :,
            ]

            # async offload
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(ldx)]].wait()
                self.offload_update_keys.copy_(update_keys, non_blocking=True)
                self.offload_update_values.copy_(update_values, non_blocking=True)
                self.copyevents[self.layer_mapping[str(ldx)]].record()

            # compute key mean, shape (batch_size*kv_head, 1, head_dim)
            mean_key = torch.mean(update_keys, dim=1, keepdim=True)

            # segmented k-means
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=update_keys
                - mean_key,  # centering to 0, (batch_size*kv_head, UPDATE_SEGMENT, dim)
                value=update_values,  # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                num_centroids=self.UPDATE_CENTROIDS,
                num_segments=1,
            )
            _centroids += mean_key
            # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.UPDATE_CENTROIDS

            # append to meta index
            self.centroids[ldx] = torch.cat(
                (self.centroids[ldx], _centroids), dim=1
            )  # (batch_size*kv_head, new_n_centroids, dim)
            self.value_sum[ldx] = torch.cat(
                (self.value_sum[ldx], _value_sum), dim=1
            )  # (batch_size*kv_head, new_n_centroids, dim)
            self.centroids_mask[ldx] = torch.cat(
                (self.centroids_mask[ldx], _cluster_size == 0), dim=1
            )  # (batch_size*kv_head, new_n_centroids)
            self.cluster_size[ldx] = torch.cat(
                (self.cluster_size[ldx], _cluster_size.to(self.dtype)), dim=1
            )  # (batch_size*kv_head, new_n_centroids)
            # assert self.centroids[ldx].shape[-2] == self.value_sum[ldx].shape[-2] == self.centroids_mask[ldx].shape[-1] == self.cluster_size[ldx].shape[-1] == self.n_centroids + self.UPDATE_CENTROIDS

            # update wave buffer
            self.copyevents[self.layer_mapping[str(ldx)]].synchronize()
            self.wave_buffer[ldx].update_kv(
                self.offload_update_keys,  # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                self.offload_update_values,  # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                _clusters.cpu().contiguous(),  # (batch_size*kv_head, UPDATE_CENTROIDS, max_cluster_size)
                _cluster_size.cpu().contiguous(),  # (batch_size*kv_head, UPDATE_CENTROIDS)
                self.cluster_ids,  # (batch_size*kv_head, new_nprobe)
            )

        # reset current device (layer 0)
        torch.cuda.set_device(self.layer_mapping[str(0)])
        # switch to sparse attention, and update index will disable cudagraph
        assert not self.use_cuda_graph, "CUDA Graph does not support index updating."
        self.attn_func = self.sparse_attention

        # update n_centroids, es_cluster_num
        self.n_centroids += self.UPDATE_CENTROIDS
        self.es_cluster_num += self.UPDATE_ES
        self.max_compute_cluster_num += self.UPDATE_NPROBE + self.UPDATE_ES

        # re-allocate layer-share buffers
        for device_idx in self.device_list:
            self.gemm_o_dict[device_idx] = torch.zeros(
                (self.batch_size, self.kv_head, self.group_size, self.n_centroids),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()
            self.softmax_o_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.group_size, self.n_centroids),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()
            self.norm_dict[device_idx] = torch.zeros(
                (
                    self.batch_groups,
                    self.group_size,
                    (self.n_centroids + 256 - 1) // 256,
                ),
                device=device_idx,
                dtype=torch.float32,
            ).contiguous()
            self.sum_dict[device_idx] = torch.zeros(
                (
                    self.batch_groups,
                    self.group_size,
                    (self.n_centroids + 256 - 1) // 256,
                ),
                device=device_idx,
                dtype=torch.float32,
            ).contiguous()
            self.dist_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.n_centroids),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()
            self.cI_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.max_compute_cluster_num),
                device=device_idx,
                dtype=torch.int64,
            ).contiguous()
            self.cV_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.max_compute_cluster_num),
                device=device_idx,
                dtype=self.dtype,
            ).contiguous()
            self.es_centroids_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.es_value_sum_dict[device_idx] = torch.zeros(
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()
            self.es_cluster_size_dict[device_idx] = torch.zeros(
                (self.batch_groups, 1, 1, self.es_cluster_num),
                dtype=self.dtype,
                device=device_idx,
            ).contiguous()

        # set pointers to current device (layer 0)
        self.gemm_o = self.gemm_o_dict[self.layer_mapping[str(0)]]
        self.softmax_o = self.softmax_o_dict[self.layer_mapping[str(0)]]
        self.norm = self.norm_dict[self.layer_mapping[str(0)]]
        self.sum = self.sum_dict[self.layer_mapping[str(0)]]
        self.dist = self.dist_dict[self.layer_mapping[str(0)]]
        self.cI = self.cI_dict[self.layer_mapping[str(0)]]
        self.cV = self.cV_dict[self.layer_mapping[str(0)]]
        self.es_centroids = self.es_centroids_dict[self.layer_mapping[str(0)]]
        self.es_value_sum = self.es_value_sum_dict[self.layer_mapping[str(0)]]
        self.es_cluster_size = self.es_cluster_size_dict[self.layer_mapping[str(0)]]

        # reset static pattern length
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        print(
            f"nprobe: {self.nprobe}, es_cluster_num: {self.es_cluster_num}, max_compute_cluster_num: {self.max_compute_cluster_num}, n_centroids: {self.n_centroids}"
        )

    def allocate_test_buffer(self):
        """
        Test 모드에서 sparse_attention_only 와 compute_using_gpu 를 위한 전용 버퍼 할당.
        두 함수가 서로 다른 버퍼를 사용하여 교차 오염을 방지.
        """
        device = self.device_list[0]  # 단일 GPU 가정

        # NELSSA path 전용 (sparse_attention_only)
        self.gemm_o_nelssa = torch.zeros(
            (self.batch_size, self.kv_head, self.group_size, self.n_centroids),
            device=device,
            dtype=self.dtype,
        ).contiguous()
        self.softmax_o_nelssa = torch.zeros(
            (self.batch_groups, self.group_size, self.n_centroids),
            device=device,
            dtype=self.dtype,
        ).contiguous()
        self.norm_nelssa = torch.zeros(
            (self.batch_groups, self.group_size, (self.n_centroids + 255) // 256),
            device=device,
            dtype=torch.float32,
        ).contiguous()
        self.sum_nelssa = torch.zeros(
            (self.batch_groups, self.group_size, (self.n_centroids + 255) // 256),
            device=device,
            dtype=torch.float32,
        ).contiguous()
        self.dist_nelssa = torch.zeros(
            (self.batch_groups, self.n_centroids),
            device=device,
            dtype=self.dtype,
        ).contiguous()

        # GPU path 전용 (compute_using_gpu)
        self.gemm_o_gpu = torch.zeros(
            (self.batch_size, self.kv_head, self.group_size, self.n_centroids),
            device=device,
            dtype=self.dtype,
        ).contiguous()
        self.softmax_o_gpu = torch.zeros(
            (self.batch_groups, self.group_size, self.n_centroids),
            device=device,
            dtype=self.dtype,
        ).contiguous()
        self.norm_gpu = torch.zeros(
            (self.batch_groups, self.group_size, (self.n_centroids + 255) // 256),
            device=device,
            dtype=torch.float32,
        ).contiguous()
        self.sum_gpu = torch.zeros(
            (self.batch_groups, self.group_size, (self.n_centroids + 255) // 256),
            device=device,
            dtype=torch.float32,
        ).contiguous()
        self.dist_gpu = torch.zeros(
            (self.batch_groups, self.n_centroids),
            device=device,
            dtype=self.dtype,
        ).contiguous()

    def compare_test_buffers(self, verbose=False):
        """
        NELSSA path 와 GPU path 의 테스트 버퍼 상태를 비교하여 차이이를 반환.

        Args:
            verbose: True 면 상세 출력, False 면 최대 차이만 반환

        Returns:
            dict: 각 버퍼별 max_diff 와 mean_diff
        """
        results = {}

        buffer_pairs = [
            ("gemm_o", self.gemm_o_nelssa, self.gemm_o_gpu),
            ("softmax_o", self.softmax_o_nelssa, self.softmax_o_gpu),
            ("norm", self.norm_nelssa, self.norm_gpu),
            ("sum", self.sum_nelssa, self.sum_gpu),
            ("dist", self.dist_nelssa, self.dist_gpu),
        ]

        for name, buf_nelssa, buf_gpu in buffer_pairs:
            diff = (buf_nelssa - buf_gpu).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            results[name] = {
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "shape": buf_nelssa.shape,
            }

            if verbose:
                print(
                    f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                    f"shape={buf_nelssa.shape}"
                )

        return results

    def decode_update_kv_cache(
        self,
        key_states,  # (bsz, seq_len(=1), group_num, dim)
        value_states,  # (bsz, seq_len(=1), group_num, dim)
        layer_idx,
    ):
        # index update when generate tokens exceed UPDATE_SEGMENT
        if (
            self.static_pattern_total
            == self.static_pattern_start + self.static_pattern_end + self.UPDATE_SEGMENT
        ):
            self._update_kv_cache()

        # append newly generated token to the steady zone
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = (
            key_states[:, 0, :, :]
        )
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = (
            value_states[:, 0, :, :]
        )

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None  # not use the return value

    def dense_attention(self, queries, layer_idx, static_len):
        """
        Full Attention
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
            self.steady_zone_keys[layer_idx].view(
                self.batch_groups, -1, 1, self.head_dim
            ),
            self.steady_zone_values[layer_idx].view(
                self.batch_groups, -1, 1, self.head_dim
            ),
            previous_out=None,
            previous_lse=None,
            cache_seqlens=static_len,
            return_softmax_lse=False,
        )
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)

    def sparse_attention(self, queries, layer_idx, static_len):
        """
        Sparse Attention
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        self.static_len_tensor.fill_(static_len)

        # Softmax(QC^T) -> [batch_size*group_num, group_size, n_centroids]
        batch_gemm_softmax(
            queries,
            self.centroids[layer_idx],
            self.gemm_o,
            self.norm,
            self.sum,
            self.softmax_o,
            self.batch_groups,
            self.group_size,
            self.n_centroids,
            self.head_dim,
            self.RSQRT_DIM,
            0,
        )
        torch.sum(
            self.softmax_o, dim=1, out=self.dist
        )  # Merge groups -> [batch_size*group_num, n_centroids]
        self.dist.masked_fill_(
            self.centroids_mask[layer_idx], self.DTYPE_MIN
        )  # mask empty clusters
        torch.topk(
            self.dist,
            self.max_compute_cluster_num,
            dim=-1,
            largest=True,
            sorted=True,
            out=(self.cV, self.cI),
        )
        self.cluster_ids.copy_(
            self.cI[..., : self.nprobe]
        )  # copy the topk cluster ids to the CPU pin memory

        # PNM Server Sparse Attention Computation
        # The server computes attention for the retrieved clusters.
        # queries: [bsz, 1, num_heads, dim]
        # cluster_ids: [batch_groups, nprobe] (already pinned)

        # We use async call to PNM server
        # result_tensor shape: [batch_groups, 1, group_size, dim + 2]
        # where +2 is for LSE statistics to merge with steady zone.
        queries_cpu = queries.to(
            device="cpu"
        ).contiguous()  # dtype : torch.float16, shape [bsz, 1, num_heads, head_dim]
        ids_flat = self.cluster_ids.to(
            device="cpu"
        ).contiguous()  # TODO : cluster ids dtype torch.int64
        self.nelssa_client.execute_decode_batched_async(
            layer_idx=layer_idx,
            bsz=self.batch_size,
            queries_tensor=queries_cpu,
            cluster_ids_tensor=ids_flat,
        )

        # estimation zone attention computation
        if self.es_cluster_num > 0:
            gather_copy_vectors(
                self.centroids[layer_idx],
                self.es_centroids,
                self.value_sum[layer_idx],
                self.es_value_sum,
                self.cluster_size[layer_idx],
                self.es_cluster_size,
                self.cI,
                self.batch_groups,
                self.n_centroids,
                self.es_cluster_num,
                self.max_compute_cluster_num,
                self.nprobe,
                self.es_cluster_num,
            )

            es_out, es_lse = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
                self.es_centroids,  # [batch_size*group_num, es_cluster_num, 1, dim]
                self.es_value_sum,  # [batch_size*group_num, es_cluster_num, 1, dim]
                self.es_cluster_size,  # [batch_size*group_num, 1, 1, es_cluster_num]
                previous_out=None,
                previous_lse=None,
                return_softmax_lse=True,
            )
        else:
            es_out, es_lse = None, None

        # steady zone
        static_len = (
            self.static_pattern_total
            if layer_idx == self.layer_num - 1
            else self.static_pattern_total + 1
        )

        ### Steady Zone Attention
        s_keys = self.steady_zone_keys[layer_idx][
            ..., : self.static_len_tensor, :
        ].contiguous()
        s_vals = self.steady_zone_values[layer_idx][
            ..., : self.static_len_tensor, :
        ].contiguous()
        # FlashAttention 입력 요구사항에 맞게 차원 변경: [batch, seq_len, num_heads, head_dim]
        # batch_groups * group_size를 batch 차원으로 생각하고 처리할 수 있도록 조정 필요
        q_fa = queries.view(self.batch_size, 1, self.num_heads, self.head_dim)
        k_fa = s_keys.permute(0, 2, 1, 3).contiguous()  # [B, L, KV, D]
        v_fa = s_vals.permute(0, 2, 1, 3).contiguous()  # [B, L, KV, D]
        # Dao-AILab의 flash_attn을 호출하여 Attention 값과 LSE를 한 번에 연산 (O(N^2) 메모리 방지)
        steady_out, steady_lse, _ = flash_attn_func(
            q_fa,
            k_fa,
            v_fa,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),  # 명시적 scaling
            return_attn_probs=True,
        )
        # steady_lse: [B, num_heads, 1] → squeeze → [B, H]
        # H = num_heads = kv_head * group_size
        steady_lse = steady_lse.squeeze(-1)  # [B, H]

        # FlashAttn 은 GQA 를 내부 처리하지만, LSE 는 각 query head 별로 반환됨
        # [B, H] 를 [B*KV, G] 로 변환: H = KV * G 이므로
        # [B, KV*G] → [B, KV, G] → [B*KV, G]
        steady_lse = steady_lse.view(
            self.batch_size, self.kv_head, self.group_size
        )  # [B, KV, G]
        steady_lse = steady_lse.reshape(self.batch_groups, self.group_size)  # [B*KV, G]

        # Native 출력 shape 맞춤: [B*KV, 1, G, 1]
        steady_lse = steady_lse.unsqueeze(1).unsqueeze(-1)  # [B*KV, 1, G, 1]

        # steady_out: [B, 1, H, D] -> [B, 1, KV*G, D] -> [B, KV, G, D] -> [B*KV, 1, G, D]
        steady_out = steady_out.squeeze(1)  # [B, H, D]
        steady_out = steady_out.view(
            self.batch_size, self.kv_head, self.group_size, self.head_dim
        )  # [B, KV, G, D]
        steady_out = steady_out.reshape(
            self.batch_groups, 1, self.group_size, self.head_dim
        )  # [B*KV, 1, G, D]
        ###

        # Since execute_decode_batched_async in the current NelssaClient.py doesn't
        # return the result directly but uses poll_doorbell, we need to poll it.
        # Let's assume for now we need to call poll_doorbell to get the final tensor.
        pnm_attn_out = self.nelssa_client.poll_doorbell(layer_idx, queries)

        # Merge pnm_attn_out and steady_zone_out
        # This is where Task #3 (Merging) comes in.
        # For now, we implement a simple merge logic or placeholder.

        # Merge logic:
        # pnm_attn_out: [batch_groups, 1, group_size, dim + 2]
        # steady_zone_out: (attn_out, lse)

        # pnm_attn_out = pnm_attn_out.to(
        #     device=self.layer_mapping[str(layer_idx)]
        # )  # TODO
        # final_out = self.merge_attention_results(pnm_attn_out, steady_zone_out)

        opt = True
        if not opt:
            # r_raw = pnm_attn_out.to(
            #     device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
            # )
            # # r_raw = r_raw_flat.view(self.batch_groups, 1, self.group_size, self.head_dim + 2)

            # retrieval_out = r_raw[..., : self.head_dim]
            # r_stats = r_raw[..., self.head_dim :]

            # r_sum = r_stats[..., 0:1]
            # r_max = r_stats[..., 1:2]

            # r_sum = torch.clamp(r_sum, min=1e-9)
            # retrieval_lse = r_max + torch.log(r_sum)

            # # 5. Merge Results
            # results_to_merge = [
            #     (retrieval_out, retrieval_lse),  # edit: disable retrieval zone
            #     (es_out, es_lse),  # edit: disable estimation zone
            #     (steady_out, steady_lse),
            # ]
            # final_output = self._merge_results_with_lse(results_to_merge, layer_idx)

            # # if final_output is None:
            # #     final_output = torch.zeros_like(queries, dtype=torch.float32)
            # # else:
            # #     final_output = final_output.to(dtype=torch.float32)

            # final_output = final_output.view(
            #     self.batch_size, 1, self.num_heads, self.head_dim
            # ).to(torch.float16)

            pnm_attn_out = pnm_attn_out.to(
                device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
            )

            # 새 format: [attn: B*H*D][sum: B*H][max: B*H]
            attn_size = self.batch_size * self.num_heads * self.head_dim  # 4096
            num_heads_total = self.batch_size * self.num_heads  # 32

            retrieval_out = pnm_attn_out[:attn_size].view(
                self.batch_size, self.num_heads, 1, self.head_dim
            )
            r_sum = pnm_attn_out[attn_size : attn_size + num_heads_total].view(
                self.batch_groups, 1, self.group_size, 1
            )
            r_max = pnm_attn_out[attn_size + num_heads_total :].view(
                self.batch_groups, 1, self.group_size, 1
            )

            r_sum = torch.clamp(r_sum, min=1e-9)
            retrieval_lse = r_max + torch.log(r_sum)

            results_to_merge = [
                (retrieval_out, retrieval_lse),  # edit: disable retrieval zone
                (es_out, es_lse),  # edit: disable estimation zone
                (steady_out, steady_lse),
            ]
            final_output = self._merge_results_with_lse(results_to_merge, layer_idx)
            final_output = final_output.view(
                self.batch_size, 1, self.num_heads, self.head_dim
            ).to(torch.float16)
        else:
            r_raw = pnm_attn_out.to(
                device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
            )
            final_output = self.triton_merge_wrapper(
                r_raw, es_out, es_lse, steady_out, steady_lse, self.head_dim
            )
            final_output = final_output.view(
                self.batch_size, 1, self.num_heads, self.head_dim
            )

        # r_raw = pnm_attn_out.to(
        #     device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
        # )
        # final_output = self.triton_merge_wrapper(
        #     r_raw, es_out, es_lse, steady_out, steady_lse, self.head_dim
        # )
        # final_output = final_output.view(
        #     self.batch_size, 1, self.num_heads, self.head_dim
        # )

        return final_output.view(self.batch_size, 1, self.num_heads, self.head_dim)

    def sparse_attention_verify(self, queries, layer_idx, static_len):
        """
        Sparse Attention
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        self.static_len_tensor.fill_(static_len)

        # Softmax(QC^T) -> [batch_size*group_num, group_size, n_centroids]
        batch_gemm_softmax(
            queries,
            self.centroids[layer_idx],
            self.gemm_o,
            self.norm,
            self.sum,
            self.softmax_o,
            self.batch_groups,
            self.group_size,
            self.n_centroids,
            self.head_dim,
            self.RSQRT_DIM,
            0,
        )
        torch.sum(
            self.softmax_o, dim=1, out=self.dist
        )  # Merge groups -> [batch_size*group_num, n_centroids]
        self.dist.masked_fill_(
            self.centroids_mask[layer_idx], self.DTYPE_MIN
        )  # mask empty clusters
        torch.topk(
            self.dist,
            self.max_compute_cluster_num,
            dim=-1,
            largest=True,
            sorted=True,
            out=(self.cV, self.cI),
        )
        self.cluster_ids.copy_(
            self.cI[..., : self.nprobe]
        )  # copy the topk cluster ids to the CPU pin memory
        # print("cluster shape : ", self.cluster_ids.shape)

        # PNM Server Sparse Attention Computation
        # The server computes attention for the retrieved clusters.
        # queries: [bsz, 1, num_heads, dim]
        # cluster_ids: [batch_groups, nprobe] (already pinned)

        # We use async call to PNM server
        # result_tensor shape: [batch_groups, 1, group_size, dim + 2]
        # where +2 is for LSE statistics to merge with steady zone.
        queries_cpu = queries.to(
            device="cpu"
        ).contiguous()  # dtype : torch.float16, shape [bsz, 1, num_heads, head_dim]
        ids_flat = self.cluster_ids.to(
            device="cpu"
        ).contiguous()  # TODO : cluster ids dtype torch.int64
        self.nelssa_client.execute_decode_batched_async(
            layer_idx=layer_idx,
            bsz=self.batch_size,
            queries_tensor=queries_cpu,
            cluster_ids_tensor=ids_flat,
        )

        # estimation zone attention computation
        if self.es_cluster_num > 0:
            gather_copy_vectors(
                self.centroids[layer_idx],
                self.es_centroids,
                self.value_sum[layer_idx],
                self.es_value_sum,
                self.cluster_size[layer_idx],
                self.es_cluster_size,
                self.cI,
                self.batch_groups,
                self.n_centroids,
                self.es_cluster_num,
                self.max_compute_cluster_num,
                self.nprobe,
                self.es_cluster_num,
            )

            es_out, es_lse = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
                self.es_centroids,  # [batch_size*group_num, es_cluster_num, 1, dim]
                self.es_value_sum,  # [batch_size*group_num, es_cluster_num, 1, dim]
                self.es_cluster_size,  # [batch_size*group_num, 1, 1, es_cluster_num]
                previous_out=None,
                previous_lse=None,
                return_softmax_lse=True,
            )
        else:
            es_out, es_lse = None, None

        # steady zone
        static_len = (
            self.static_pattern_total
            if layer_idx == self.layer_num - 1
            else self.static_pattern_total + 1
        )

        opt = False
        if not opt:
            s_keys = self.steady_zone_keys[layer_idx][
                ..., : self.static_len_tensor, :
            ]  # (batch_size, kv_head, static_len, head_dim)
            s_vals = self.steady_zone_values[layer_idx][
                ..., : self.static_len_tensor, :
            ]
            s_keys = s_keys.contiguous().view(self.batch_groups, 1, -1, self.head_dim)
            s_vals = s_vals.contiguous().view(self.batch_groups, 1, -1, self.head_dim)
            # Native attention
            q_pt = queries.view(self.batch_groups, self.group_size, 1, self.head_dim)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (
                torch.matmul(q_pt, s_keys.transpose(-2, -1)) * scale
            )  # [B*KV, Group, 1, Len] (Broadcasting 1->Group)
            s_max = torch.max(scores, dim=-1, keepdim=True)[0]
            s_exp = torch.exp(scores - s_max)
            s_sum = torch.sum(s_exp, dim=-1, keepdim=True)
            s_sum = torch.clamp(s_sum, min=1e-9)
            probs = s_exp / s_sum
            s_out = torch.matmul(probs, s_vals)
            steady_out = s_out.permute(0, 2, 1, 3)
            steady_lse = (
                (s_max + torch.log(s_sum)).permute(0, 2, 1, 3).to(torch.float32)
            )
        else:
            s_keys = self.steady_zone_keys[layer_idx][
                ..., : self.static_len_tensor, :
            ].contiguous()
            s_vals = self.steady_zone_values[layer_idx][
                ..., : self.static_len_tensor, :
            ].contiguous()
            # FlashAttention 입력 요구사항에 맞게 차원 변경: [batch, seq_len, num_heads, head_dim]
            # batch_groups * group_size를 batch 차원으로 생각하고 처리할 수 있도록 조정 필요
            q_fa = queries.view(self.batch_size, 1, self.num_heads, self.head_dim)
            k_fa = s_keys.permute(0, 2, 1, 3).contiguous()  # [B, L, KV, D]
            v_fa = s_vals.permute(0, 2, 1, 3).contiguous()  # [B, L, KV, D]
            # Dao-AILab의 flash_attn을 호출하여 Attention 값과 LSE를 한 번에 연산 (O(N^2) 메모리 방지)
            # steady_out, steady_lse, _ = flash_attn_func(
            #     q_fa, k_fa, v_fa, return_attn_probs=True
            # )
            # steady_out = steady_out.transpose(1, 2).reshape(
            #     self.batch_groups, self.group_size, 1, self.head_dim
            # )
            # steady_lse = (
            #     steady_lse.unsqueeze(-1)
            #     .transpose(1, 2)
            #     .view(self.batch_groups, 1, self.group_size, 1)
            # )
            steady_out, steady_lse, _ = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),  # 명시적 scaling
                return_attn_probs=True,
            )
            # steady_lse: [B, num_heads, 1] → squeeze → [B, H]
            # H = num_heads = kv_head * group_size
            steady_lse = steady_lse.squeeze(-1)  # [B, H]

            # FlashAttn 은 GQA 를 내부 처리하지만, LSE 는 각 query head 별로 반환됨
            # [B, H] 를 [B*KV, G] 로 변환: H = KV * G 이므로
            # [B, KV*G] → [B, KV, G] → [B*KV, G]
            steady_lse = steady_lse.view(
                self.batch_size, self.kv_head, self.group_size
            )  # [B, KV, G]
            steady_lse = steady_lse.reshape(
                self.batch_groups, self.group_size
            )  # [B*KV, G]

            # Native 출력 shape 맞춤: [B*KV, 1, G, 1]
            steady_lse = steady_lse.unsqueeze(1).unsqueeze(-1)  # [B*KV, 1, G, 1]

            # steady_out: [B, 1, H, D] -> [B, 1, KV*G, D] -> [B, KV, G, D] -> [B*KV, 1, G, D]
            steady_out = steady_out.squeeze(1)  # [B, H, D]
            steady_out = steady_out.view(
                self.batch_size, self.kv_head, self.group_size, self.head_dim
            )  # [B, KV, G, D]
            steady_out = steady_out.reshape(
                self.batch_groups, 1, self.group_size, self.head_dim
            )  # [B*KV, 1, G, D]

        # Since execute_decode_batched_async in the current NelssaClient.py doesn't
        # return the result directly but uses poll_doorbell, we need to poll it.
        # Let's assume for now we need to call poll_doorbell to get the final tensor.
        pnm_attn_out = self.nelssa_client.poll_doorbell(layer_idx, queries)

        # Steady Zone Attention Computation (GPU)
        # We still need to compute attention for the steady zone locally.
        # steady_zone_out = weighted_flash_decoding(
        #     queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
        #     self.steady_zone_keys[layer_idx].view(
        #         self.batch_groups, -1, 1, self.head_dim
        #     ),
        #     self.steady_zone_values[layer_idx].view(
        #         self.batch_groups, -1, 1, self.head_dim
        #     ),
        #     previous_out=None,
        #     previous_lse=None,
        #     cache_seqlens=static_len,
        #     return_softmax_lse=True,
        # )

        # Merge pnm_attn_out and steady_zone_out
        # This is where Task #3 (Merging) comes in.
        # For now, we implement a simple merge logic or placeholder.

        # Merge logic:
        # pnm_attn_out: [batch_groups, 1, group_size, dim + 2]
        # steady_zone_out: (attn_out, lse)

        # pnm_attn_out = pnm_attn_out.to(
        #     device=self.layer_mapping[str(layer_idx)]
        # )  # TODO
        # final_out = self.merge_attention_results(pnm_attn_out, steady_zone_out)

        opt = False
        if not opt:
            # r_raw = pnm_attn_out.to(
            #     device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
            # )
            # # r_raw = r_raw_flat.view(self.batch_groups, 1, self.group_size, self.head_dim + 2)

            # retrieval_out = r_raw[..., : self.head_dim]
            # r_stats = r_raw[..., self.head_dim :]

            # r_sum = r_stats[..., 0:1]
            # r_max = r_stats[..., 1:2]

            # r_sum = torch.clamp(r_sum, min=1e-9)
            # retrieval_lse = r_max + torch.log(r_sum)

            # # 5. Merge Results
            # results_to_merge = [
            #     (retrieval_out, retrieval_lse),  # edit: disable retrieval zone
            #     (es_out, es_lse),  # edit: disable estimation zone
            #     (steady_out, steady_lse),
            # ]
            # final_output = self._merge_results_with_lse(results_to_merge, layer_idx)

            # # if final_output is None:
            # #     final_output = torch.zeros_like(queries, dtype=torch.float32)
            # # else:
            # #     final_output = final_output.to(dtype=torch.float32)

            # final_output = final_output.view(
            #     self.batch_size, 1, self.num_heads, self.head_dim
            # ).to(torch.float16)

            pnm_attn_out = pnm_attn_out.to(
                device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
            )

            # 새 format: [attn: B*H*D][sum: B*H][max: B*H]
            attn_size = self.batch_size * self.num_heads * self.head_dim  # 4096
            num_heads_total = self.batch_size * self.num_heads  # 32

            retrieval_out = pnm_attn_out[:attn_size].view(
                self.batch_size, self.num_heads, 1, self.head_dim
            )
            r_sum = pnm_attn_out[attn_size : attn_size + num_heads_total].view(
                self.batch_groups, 1, self.group_size, 1
            )
            r_max = pnm_attn_out[attn_size + num_heads_total :].view(
                self.batch_groups, 1, self.group_size, 1
            )

            r_sum = torch.clamp(r_sum, min=1e-9)
            retrieval_lse = r_max + torch.log(r_sum)

            results_to_merge = [
                (retrieval_out, retrieval_lse),  # edit: disable retrieval zone
                (es_out, es_lse),  # edit: disable estimation zone
                (steady_out, steady_lse),
            ]
            final_output = self._merge_results_with_lse(results_to_merge, layer_idx)
            final_output = final_output.view(
                self.batch_size, 1, self.num_heads, self.head_dim
            ).to(torch.float16)
        else:
            r_raw = pnm_attn_out.to(
                device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
            )
            final_output = self.triton_merge_wrapper(
                r_raw, es_out, es_lse, steady_out, steady_lse, self.head_dim
            )
            final_output = final_output.view(
                self.batch_size, 1, self.num_heads, self.head_dim
            )

        # r_raw = pnm_attn_out.to(
        #     device=self.layer_mapping[str(layer_idx)], dtype=torch.float32
        # )
        # final_output = self.triton_merge_wrapper(
        #     r_raw, es_out, es_lse, steady_out, steady_lse, self.head_dim
        # )
        # final_output = final_output.view(
        #     self.batch_size, 1, self.num_heads, self.head_dim
        # )

        return final_output.view(self.batch_size, 1, self.num_heads, self.head_dim)

    def _merge_results_with_lse(self, results, layer_idx):
        """
        batch 2 example shape
        Retrieval out shape : torch.Size([16, 1, 4, 128])
        Retrieval lse shape : torch.Size([16, 1, 4, 1])
        Estimation out shape : torch.Size([16, 1, 4, 128])
        Estimation lse shape : torch.Size([16, 4, 1])
        Steady out shape : torch.Size([16, 4, 1, 128])
        Steady lse shape : torch.Size([16, 1, 4, 1])
        """

        valid_results = []
        for name, res in zip(["Retrieval", "Estimation", "Steady"], results):
            if res is not None and res[0] is not None:
                valid_results.append((name, res[0], res[1]))

        if not valid_results:
            return None

        target_out_shape = (self.batch_groups, 1, self.group_size, self.head_dim)
        target_lse_shape = (self.batch_groups, 1, self.group_size, 1)

        normalized_lses = []
        normalized_outs = []

        for name, out, lse in valid_results:
            out = out.to(dtype=torch.float32)
            lse = lse.to(dtype=torch.float32)

            lse = torch.nan_to_num(lse, nan=float("-inf"))
            out = torch.nan_to_num(out, nan=0.0)

            # numel이 같으면 강제로 shape 맞춰서 병합
            if out.shape != target_out_shape:
                if out.numel() == np.prod(target_out_shape):
                    out = out.view(target_out_shape)
                else:
                    continue

            if lse.shape != target_lse_shape:
                if lse.numel() == np.prod(target_lse_shape):
                    lse = lse.view(target_lse_shape)
                else:
                    continue

            normalized_lses.append(lse)
            normalized_outs.append(out)
        if not normalized_lses:
            return None

        # Global Merge (Online Softmax Logic)
        all_lses = torch.cat(normalized_lses, dim=-1)
        global_lse = torch.logsumexp(all_lses, dim=-1, keepdim=True)

        final_out = 0.0
        for i in range(len(normalized_outs)):
            # Weight = exp(local_lse - global_lse)
            log_weight = normalized_lses[i] - global_lse
            weight = torch.exp(log_weight)
            weight = torch.nan_to_num(weight, nan=0.0)
            final_out += normalized_outs[i] * weight

        return final_out

    @triton.jit
    def fused_pnm_merge_kernel(
        # Pointers
        r_out_ptr,
        r_sum_ptr,
        r_max_ptr,
        e_out_ptr,
        e_lse_ptr,
        s_out_ptr,
        s_lse_ptr,
        out_ptr,
        # Strides
        stride_r_row,
        stride_e_row,
        stride_s_row,
        stride_out_row,
        # Constants
        HEAD_DIM: tl.constexpr,
    ):
        """
        Triton kernel for merging PNM attention results with steady/estimation zones.

        Args:
            r_out_ptr: (N, HEAD_DIM) - PNM attention output
            r_sum_ptr: (N,) - PNM sum statistics
            r_max_ptr: (N,) - PNM max statistics
            e_out_ptr: (N, HEAD_DIM) - Estimation zone output
            e_lse_ptr: (N,) - Estimation zone LSE
            s_out_ptr: (N, HEAD_DIM) - Steady zone output
            s_lse_ptr: (N,) - Steady zone LSE
            out_ptr: (N, HEAD_DIM) - Output tensor
            stride_*: Strides for each tensor
            HEAD_DIM: Head dimension (constexpr)
        """
        # pid is 0 ~ N-1 (batch_groups * group_size = total query heads)
        pid = tl.program_id(axis=0)
        # Calculate row start addresses
        r_row_start = r_out_ptr + pid * stride_r_row
        e_row_start = e_out_ptr + pid * stride_e_row
        s_row_start = s_out_ptr + pid * stride_s_row
        out_row_start = out_ptr + pid * stride_out_row
        # Offsets [0, 1, 2, ..., HEAD_DIM-1]
        offsets = tl.arange(0, HEAD_DIM)
        # -----------------------------------------------------------
        # 1. Load PNM data and compute LSE
        #    r_out: (N, HEAD_DIM), r_sum: (N,), r_max: (N,)
        # -----------------------------------------------------------
        r_out = tl.load(r_row_start + offsets)
        r_sum = tl.load(r_sum_ptr + pid)
        r_max = tl.load(r_max_ptr + pid)
        r_sum = tl.maximum(r_sum, 1e-9)
        r_lse = r_max + tl.math.log(r_sum)
        # NaN handling
        r_lse = tl.where(r_lse != r_lse, float("-inf"), r_lse)
        r_out = tl.where(r_out != r_out, 0.0, r_out)
        # -----------------------------------------------------------
        # 2. Load Estimation & Steady data
        # -----------------------------------------------------------
        e_out = tl.load(e_row_start + offsets)
        e_lse = tl.load(e_lse_ptr + pid)
        e_lse = tl.where(e_lse != e_lse, float("-inf"), e_lse)
        e_out = tl.where(e_out != e_out, 0.0, e_out)
        s_out = tl.load(s_row_start + offsets)
        s_lse = tl.load(s_lse_ptr + pid)
        s_lse = tl.where(s_lse != s_lse, float("-inf"), s_lse)
        s_out = tl.where(s_out != s_out, 0.0, s_out)
        # -----------------------------------------------------------
        # 3. Compute Global LSE (logsumexp)
        # -----------------------------------------------------------
        # m = max(r_lse, e_lse, s_lse)
        m = tl.maximum(r_lse, tl.maximum(e_lse, s_lse))
        # exp(x - m)
        exp_r = tl.exp(r_lse - m)
        exp_e = tl.exp(e_lse - m)
        exp_s = tl.exp(s_lse - m)
        sum_exp = exp_r + exp_e + exp_s
        global_lse = m + tl.math.log(sum_exp)
        # -----------------------------------------------------------
        # 4. Compute weights and final output
        # -----------------------------------------------------------
        w_r = tl.exp(r_lse - global_lse)
        w_e = tl.exp(e_lse - global_lse)
        w_s = tl.exp(s_lse - global_lse)
        w_r = tl.where(w_r != w_r, 0.0, w_r)
        w_e = tl.where(w_e != w_e, 0.0, w_e)
        w_s = tl.where(w_s != w_s, 0.0, w_s)
        final_out = (w_r * r_out) + (w_e * e_out) + (w_s * s_out)
        # Store result to VRAM
        tl.store(out_row_start + offsets, final_out)

    def triton_merge_wrapper(
        self, r_raw_flat, e_out, e_lse, s_out, s_lse, head_dim=128
    ):
        """
        Triton kernel wrapper for merging PNM attention results with steady/estimation zones.

        PNM new structure: r_raw_flat is a 1D flat tensor [attn_all][sum_all][max_all]
        - attn_all: batch_size * num_heads * head_dim elements
        - sum_all: batch_size * num_heads elements
        - max_all: batch_size * num_heads elements
        """
        # 1. Split 1D flat tensor from PNM into separate sections
        # Current PNM structure: [attn_results][sum_values][max_values]
        attn_size = self.batch_size * self.num_heads * head_dim
        num_heads_total = self.batch_size * self.num_heads

        # Split each section
        attn_flat = r_raw_flat[:attn_size]
        sum_flat = r_raw_flat[attn_size : attn_size + num_heads_total]
        max_flat = r_raw_flat[attn_size + num_heads_total :]

        # 2. Convert to (N, head_dim) format
        # N = batch_groups * group_size = batch_size * num_heads
        r_out = attn_flat.view(-1, head_dim).contiguous()  # (N, head_dim)
        r_sum = sum_flat.view(-1).contiguous()  # (N,)
        r_max = max_flat.view(-1).contiguous()  # (N,)

        # 3. Make existing tensors contiguous
        e_out = e_out.view(-1, head_dim).contiguous()
        e_lse = e_lse.view(-1).contiguous()
        s_out = s_out.view(-1, head_dim).contiguous()
        s_lse = s_lse.view(-1).contiguous()

        N = r_out.shape[0]  # batch_groups * group_size
        out = torch.empty_like(e_out)

        # 4. 1D Grid setup (create blocks for each row)
        grid = (N,)
        self.fused_pnm_merge_kernel[grid](
            r_out,
            r_sum,
            r_max,
            e_out,
            e_lse,
            s_out,
            s_lse,
            out,
            r_out.stride(0),
            e_out.stride(0),
            s_out.stride(0),
            out.stride(0),
            HEAD_DIM=head_dim,
            num_warps=4,
        )
        return out

    def merge_attention_results(self, pnm_out, steady_out):
        """
        Merges attention results from PNM server and local steady zone.
        pnm_out: [batch_groups, 1, group_size, dim + 2]
        steady_out: (steady_attn_out, steady_lse)
        """
        # pnm_out[:, :, :, -2:] contains the LSE statistics for the PNM part.
        # Based on the PNM server implementation, let's assume the last 2 elements are LSE and something else,
        # or it's a specific LSE value. Usually, LSE is a single scalar per head.
        # If dim + 2, let's assume pnm_out[..., -1] is the LSE.

        pnm_val = pnm_out[..., :-2]
        pnm_lse = pnm_out[..., -1:]  # Using the last element as LSE

        steady_val, steady_lse = (
            steady_out  # steady_lse shape: [batch_groups, group_size, 1]
        )

        # Ensure shapes match for broadcasting
        # pnm_lse: [batch_groups, 1, group_size, 1]
        # steady_lse: [batch_groups, group_size, 1] -> reshape to [batch_groups, 1, group_size, 1]
        steady_lse = steady_lse.unsqueeze(1)
        # 1. steady_lse의 shape 확인 및 변환
        # steady_lse: [batch_group, 1, group_size, 1] -> [bsz, 1, num_head, 1]
        if pnm_lse.dim() == 4 and steady_lse.shape[0] != pnm_lse.shape[0]:
            # batch_group(8) * group_size(4) = num_head(32)
            # view를 사용하여 [1, 1, 32, 1] 형태로 변경
            # bsz = pnm_lse.shape[0]
            pnm_lse = steady_lse.view(self.batch_groups, 1, -1, 1)
            pnm_val = steady_val.view(self.batch_groups, 1, -1, self.head_dim)

        # Log-Sum-Exp trick for numerical stability
        # max_lse = max(LSE_pnm, LSE_steady)
        max_lse = torch.max(pnm_lse, steady_lse)

        # weights = exp(LSE - max_lse)
        w_pnm = torch.exp(pnm_lse - max_lse)
        w_steady = torch.exp(steady_lse - max_lse)

        # final_out = (val_pnm * w_pnm + val_steady * w_steady) / (w_pnm + w_steady)
        # steady_val: [batch_groups, 1, group_size, dim]
        final_out = (pnm_val * w_pnm + steady_val * w_steady) / (
            w_pnm + w_steady + 1e-6
        )

        # TODO : pnm out is float32
        return final_out.to(dtype=self.dtype)

    def sparse_attention_only(
        self, queries: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        lnum = getattr(self, "layer_num", 32)
        bsz = self.batch_size
        target_dtype = queries.dtype
        target_device = queries.device
        retrieval_out, retrieval_lse = None, None
        print(queries.shape, queries.dtype)
        # 1. Centroid Search
        # queries = queries.view(self.batch_groups, 1, self.group_size, self.head_dim)
        batch_gemm_softmax(
            A=queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
            B=self.centroids[layer_idx],
            D=self.gemm_o_nelssa,
            Norm=self.norm_nelssa,
            Sum=self.sum_nelssa,
            Softmax=self.softmax_o_nelssa,
            batch_count=self.batch_groups,
            m=self.group_size,
            n=self.n_centroids,
            k=self.head_dim,
            alpha=self.RSQRT_DIM,
            beta=0.0,
        )
        dist = torch.sum(self.softmax_o_nelssa, dim=1)
        dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
        cI_all = torch.topk(
            dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True
        )[1]
        r_ids = cI_all[..., : self.nprobe].contiguous()
        # 2. Retrieval Zone (NELSSA)
        queries = queries.view(
            self.batch_size, 1, self.num_heads, self.head_dim
        ).contiguous()
        if True:
            # opt=True
            # if opt:
            #     with torch.cuda.stream(self.nelssa_stream):
            #         self.offload_ids.copy_(r_ids, non_blocking=True)
            #         self.cpu_queries_buffer.copy_(queries, non_blocking=True)
            #         self.nelssa_client.execute_decode_batched_async(
            #             layer_idx=layer_idx, bsz = bsz, queries_tensor=self.cpu_queries_buffer, cluster_ids_tensor=self.offload_ids
            #         )
            # else:
            ids_flat = r_ids.view(-1, self.nprobe).to(device="cpu").contiguous()
            queries_cpu = queries.to(device="cpu")
            self.nelssa_client.execute_decode_batched_async(
                layer_idx=layer_idx,
                bsz=bsz,
                queries_tensor=queries_cpu,
                cluster_ids_tensor=ids_flat,
            )
            # ids_flat = r_ids.view(-1, self.nprobe).contiguous().to(device='cpu')
            # queries_cpu = queries.view(self.batch_size, 1, self.num_heads, self.head_dim).to(dtype=torch.float32, device='cpu')
            # self.offload_ids.copy_(r_ids)
            # self.cpu_queries_buffer.copy_(queries_cpu)
            # r_raw_flat = self.nelssa_client.execute_decode_batched(
            #     layer_idx=layer_idx, bsz = bsz, queries=queries, cluster_ids=ids_flat
            # )
            # self.nelssa_client.execute_decode_batched_async(
            # layer_idx=layer_idx, bsz = bsz, queries_tensor=self.cpu_queries_buffer, cluster_ids_tensor=self.offload_ids
            # )
            # self.offload_ids.copy_(r_ids, non_blocking=True)
            # self.cpu_queries_buffer.copy_(queries, non_blocking=True)
            # self.nelssa_client.execute_decode_batched_async(
            # layer_idx=layer_idx, bsz = bsz, queries_tensor=self.cpu_queries_buffer, cluster_ids_tensor=self.offload_ids
            # )

            # r_raw_flat = self.nelssa_client.poll_doorbell(
            #     layer_idx=layer_idx, queries_tensor=queries
            # )
            # print("! r_raw", r_raw_flat.shape)

            # queries = queries.view(self.batch_groups, 1, self.group_size, self.head_dim)
            # r_raw_flat = r_raw_flat.to(device=target_device, dtype=torch.float32)
            # r_raw = r_raw_flat.view(
            #     self.batch_groups, 1, self.group_size, self.head_dim + 2
            # )
            # retrieval_out = r_raw[..., : self.head_dim]
            # r_stats = r_raw[..., self.head_dim :]
            # r_sum = r_stats[..., 0:1]
            # r_max = r_stats[..., 1:2]
            # r_sum = torch.clamp(r_sum, min=1e-9)
            # retrieval_lse = r_max + torch.log(r_sum)
            # print("! r", r_raw_flat.shape)
            # print("! sum ", r_sum.shape)
            # print("! max ", r_max.shape)

            r_raw_flat = self.nelssa_client.poll_doorbell(
                layer_idx=layer_idx, queries_tensor=queries
            )

            r_raw_flat = r_raw_flat.to(device=target_device, dtype=torch.float32)

            # 새 format: [attn: B*H*D][sum: B*H][max: B*H]
            attn_size = self.batch_size * self.num_heads * self.head_dim  # 4096
            num_heads_total = self.batch_size * self.num_heads  # 32

            retrieval_out = r_raw_flat[:attn_size].view(
                self.batch_size, self.num_heads, 1, self.head_dim
            )
            r_sum = r_raw_flat[attn_size : attn_size + num_heads_total].view(
                self.batch_groups, 1, self.group_size, 1
            )
            r_max = r_raw_flat[attn_size + num_heads_total :].view(
                self.batch_groups, 1, self.group_size, 1
            )
            print("! retrieval_out", retrieval_out.shape)
            print("! r_sum", r_sum.shape)
            print("! r_max", r_max.shape)

            r_sum = torch.clamp(r_sum, min=1e-9)
            retrieval_lse = r_max + torch.log(r_sum)

        # Test 비교용 retrieval 출력 저장
        self.wave_buffer[layer_idx].sync()
        self.last_nelssa_retrieval = retrieval_out.clone()
        final_output = retrieval_out.view(bsz, 1, self.num_heads, self.head_dim)
        final_output = final_output.to(torch.float16)
        return final_output, retrieval_lse, r_ids

    def compute_async(self, hidden_states, layer_idx):
        """
        [Dummy] NELSSA full compute path - 아직 구현되지 않음
        """
        # TODO: implement full NELSSA compute path
        return hidden_states

    def compute_gpu(self, hidden_states, layer_idx):
        """
        [Dummy] GPU compute path - 아직 구현되지 않음
        """
        # TODO: implement GPU compute path
        return hidden_states

    def compute_using_gpu(self, queries, layer_idx):
        """
        정합성 검증을 위해 Estimation 결과와 Retrieval+Steady 결과를 분리하여 반환하는 함수
        """
        torch.set_printoptions(sci_mode=False)

        # Search for TopK centroids (기존 코드와 동일)
        batch_gemm_softmax(
            queries,
            self.centroids[layer_idx],
            self.gemm_o_gpu,
            self.norm_gpu,
            self.sum_gpu,
            self.softmax_o_gpu,
            self.batch_groups,
            self.group_size,
            self.n_centroids,
            self.head_dim,
            self.RSQRT_DIM,
            0,
        )
        dist = torch.sum(self.softmax_o_gpu, dim=1)
        dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
        cI = torch.topk(
            dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True
        )[1]
        self.cluster_ids.copy_(cI[..., : self.nprobe])

        # ------------------------------------------------------------------------
        # [Step B] Execution Buffer 준비 (기존과 동일)
        # ------------------------------------------------------------------------
        self.wave_buffer[layer_idx].batch_access()
        gather_copy_and_concat(
            self.steady_zone_keys[layer_idx],
            self.list_keys[layer_idx],
            self.cache_keys[layer_idx],
            self.execution_buffer_keys,
            self.steady_zone_values[layer_idx],
            self.list_values[layer_idx],
            self.cache_values[layer_idx],
            self.execution_buffer_values,
            self.miss_unit_idices[layer_idx],
            self.miss_unit_sizes[layer_idx],
            self.miss_unit_sizes_cumsum[layer_idx],
            self.miss_num_units[layer_idx],
            self.hit_unit_idices[layer_idx],
            self.hit_unit_sizes[layer_idx],
            self.hit_unit_sizes_cumsum[layer_idx],
            self.hit_num_units[layer_idx],
            self.valid_lengths,
            self.batch_groups,
            self.static_stride,
            self.list_stride,
            self.cache_stride,
            self.execution_stride,
            self.buffer_size,
            self.static_len_tensor,
        )
        # 1. Retrieval Zone의 실제 유효 길이 계산
        # 전체 유효 길이(valid_lengths)에는 Steady Zone 길이(static_len)가 포함되어 있으므로 이를 뺍니다.
        retrieval_lengths = self.valid_lengths - self.static_len_tensor
        # 2. Execution Buffer에서 Steady zone 부분 잘라내기 (Slicing)
        # shape: [batch_groups, buffer_size, 1, head_dim]
        # dim=1 (Sequence length 축)에서 static_len 인덱스부터 끝까지만 선택합니다.
        # .contiguous()를 붙여 메모리 연속성을 보장하는 것이 안전합니다.
        retrieval_keys_only = self.execution_buffer_keys[
            :, self.static_len_tensor :, :, :
        ].contiguous()
        retrieval_values_only = self.execution_buffer_values[
            :, self.static_len_tensor :, :, :
        ].contiguous()

        # ------------------------------------------------------------------------
        # [Step C] Retrieval + Steady Zone 연산 결과 분리 (핵심 변경 부분)
        # ------------------------------------------------------------------------
        # previous_out=es_out을 넣지 않고 None을 넣어 "순수 Execution Buffer 결과"만 계산
        retrieval_out, retrieval_lse = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
            retrieval_keys_only,
            retrieval_values_only,
            previous_out=None,  # <--- 중요: 여기서 병합하지 않음
            previous_lse=None,
            cache_seqlens=retrieval_lengths,
            return_softmax_lse=True,  # LSE 반환 필요
        )
        # Test 비교용 retrieval 출력 저장
        self.last_gpu_retrieval = retrieval_out.clone()

        # Cache Update (기존과 동일)
        self.wave_buffer[layer_idx].sync()
        # gather_copy_and_scatter(
        #     self.execution_buffer_keys,
        #     self.cache_keys[layer_idx],
        #     self.execution_buffer_values,
        #     self.cache_values[layer_idx],
        #     self.update_buffer_indices[layer_idx],
        #     self.update_unit_sizes[layer_idx],
        #     self.update_cache_indices[layer_idx],
        #     self.update_num_units[layer_idx],
        #     self.batch_groups,
        #     self.execution_stride,
        #     self.cache_stride,
        #     self.buffer_size,
        #     self.static_len_tensor,
        # )
        return (
            retrieval_out.view(self.batch_size, 1, self.num_heads, self.head_dim),
            retrieval_lse,
            self.cluster_ids,
        )

    def sparse_attention_with_cudagraph(self, queries, layer_idx, static_len):
        """
        Sparse Attention with CUDA graph
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        self.static_len_tensor.fill_(static_len)
        self.query_buffer.copy_(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
            non_blocking=True,
        )

        # get topk clusters
        self.topk_cudagraphs[layer_idx].replay()
        self.cluster_ids.copy_(self.cI[..., : self.nprobe])  # GPU -> CPU pin memory

        # estimation zone attention computation
        if self.es_cluster_num > 0:
            self.es_cudagraphs[layer_idx].replay()

        # access cache and submit cache update jobs to thread pool
        self.wave_buffer[layer_idx].batch_access()

        # compute attention for retrieve zone and steady zone, merge estimation zone results
        self.attn_cudagraphs[layer_idx].replay()

        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        # admit pages from execution buffer to GPU cache
        self.update_cudagraphs[layer_idx].replay()

        return self.attn_out

    def capture_cuda_graph(self):
        """Capture CUDA Graph"""
        if not self.use_cuda_graph:
            return

        print("Capture CUDA graph ...")
        for layer_idx in range(self.layer_num):
            with torch.cuda.device(self.layer_mapping[str(layer_idx)]):
                capture_stream = torch.cuda.Stream(
                    device=self.layer_mapping[str(layer_idx)]
                )

                # TopK search CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(
                    self.topk_cudagraphs[layer_idx], stream=capture_stream
                ):
                    batch_gemm_softmax(
                        self.query_buffer_dict[self.layer_mapping[str(layer_idx)]],
                        self.centroids[layer_idx],
                        self.gemm_o_dict[self.layer_mapping[str(layer_idx)]],
                        self.norm_dict[self.layer_mapping[str(layer_idx)]],
                        self.sum_dict[self.layer_mapping[str(layer_idx)]],
                        self.softmax_o_dict[self.layer_mapping[str(layer_idx)]],
                        self.batch_groups,
                        self.group_size,
                        self.n_centroids,
                        self.head_dim,
                        self.RSQRT_DIM,
                        0,
                    )
                    torch.sum(
                        self.softmax_o_dict[self.layer_mapping[str(layer_idx)]],
                        dim=1,
                        out=self.dist_dict[self.layer_mapping[str(layer_idx)]],
                    )
                    self.dist_dict[self.layer_mapping[str(layer_idx)]].masked_fill_(
                        self.centroids_mask[layer_idx], self.DTYPE_MIN
                    )
                    torch.topk(
                        self.dist_dict[self.layer_mapping[str(layer_idx)]],
                        self.max_compute_cluster_num,
                        dim=-1,
                        largest=True,
                        sorted=True,
                        out=(
                            self.cV_dict[self.layer_mapping[str(layer_idx)]],
                            self.cI_dict[self.layer_mapping[str(layer_idx)]],
                        ),
                    )

                # Estimation zone CUDA graph
                if self.es_cluster_num > 0:
                    torch.cuda.synchronize()
                    with torch.cuda.graph(
                        self.es_cudagraphs[layer_idx], stream=capture_stream
                    ):
                        gather_copy_vectors(
                            self.centroids[layer_idx],
                            self.es_centroids_dict[self.layer_mapping[str(layer_idx)]],
                            self.value_sum[layer_idx],
                            self.es_value_sum_dict[self.layer_mapping[str(layer_idx)]],
                            self.cluster_size[layer_idx],
                            self.es_cluster_size_dict[
                                self.layer_mapping[str(layer_idx)]
                            ],
                            self.cI_dict[self.layer_mapping[str(layer_idx)]],
                            self.batch_groups,
                            self.n_centroids,
                            self.es_cluster_num,
                            self.max_compute_cluster_num,
                            self.nprobe,
                            self.es_cluster_num,
                        )
                        # TODO: add output API in this kernel
                        es_out, es_lse = weighted_flash_decoding(
                            self.query_buffer_dict[self.layer_mapping[str(layer_idx)]],
                            self.es_centroids_dict[self.layer_mapping[str(layer_idx)]],
                            self.es_value_sum_dict[self.layer_mapping[str(layer_idx)]],
                            self.es_cluster_size_dict[
                                self.layer_mapping[str(layer_idx)]
                            ],
                            previous_out=None,
                            previous_lse=None,
                            return_softmax_lse=True,
                        )
                        self.es_out_dict[self.layer_mapping[str(layer_idx)]].copy_(
                            es_out, non_blocking=True
                        )
                        self.es_lse_dict[self.layer_mapping[str(layer_idx)]].copy_(
                            es_lse, non_blocking=True
                        )

                # Retrieval and Steady zone CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(
                    self.attn_cudagraphs[layer_idx], stream=capture_stream
                ):
                    gather_copy_and_concat(
                        self.steady_zone_keys[layer_idx],
                        self.list_keys[layer_idx],
                        self.cache_keys[layer_idx],
                        self.execution_buffer_keys_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        self.steady_zone_values[layer_idx],
                        self.list_values[layer_idx],
                        self.cache_values[layer_idx],
                        self.execution_buffer_values_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        self.miss_unit_idices[layer_idx],
                        self.miss_unit_sizes[layer_idx],
                        self.miss_unit_sizes_cumsum[layer_idx],
                        self.miss_num_units[layer_idx],
                        self.hit_unit_idices[layer_idx],
                        self.hit_unit_sizes[layer_idx],
                        self.hit_unit_sizes_cumsum[layer_idx],
                        self.hit_num_units[layer_idx],
                        self.valid_lengths_dict[self.layer_mapping[str(layer_idx)]],
                        self.batch_groups,
                        self.static_stride,
                        self.list_stride,
                        self.cache_stride,
                        self.execution_stride,
                        self.buffer_size,
                        self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]],
                    )
                    # TODO: add output API in this kernel
                    attn_out = weighted_flash_decoding(
                        self.query_buffer_dict[self.layer_mapping[str(layer_idx)]],
                        self.execution_buffer_keys_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        self.execution_buffer_values_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        previous_out=self.es_out_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        previous_lse=self.es_lse_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        cache_seqlens=self.valid_lengths_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        return_softmax_lse=False,
                    )
                    self.attn_out_dict[self.layer_mapping[str(layer_idx)]].copy_(
                        attn_out.view(
                            self.batch_size, 1, self.num_heads, self.head_dim
                        ),
                        non_blocking=True,
                    )

                # Cache update CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(
                    self.update_cudagraphs[layer_idx], stream=capture_stream
                ):
                    gather_copy_and_scatter(
                        self.execution_buffer_keys_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        self.cache_keys[layer_idx],
                        self.execution_buffer_values_dict[
                            self.layer_mapping[str(layer_idx)]
                        ],
                        self.cache_values[layer_idx],
                        self.update_buffer_indices[layer_idx],
                        self.update_unit_sizes[layer_idx],
                        self.update_cache_indices[layer_idx],
                        self.update_num_units[layer_idx],
                        self.batch_groups,
                        self.execution_stride,
                        self.cache_stride,
                        self.buffer_size,
                        self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]],
                    )

                torch.cuda.synchronize()

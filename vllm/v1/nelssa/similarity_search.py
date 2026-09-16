# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Centroid-based similarity search for NELSSA KV cache retrieval."""

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.ops.sparse_kv_similarity_search import (
    batched_centroid_search,
    get_topk_cluster_ids,
)

logger = init_logger(__name__)

# NELSSA: load the C++ clustering extension (vllm_clustering_ext, same .so as
# reorganize/gather) for the C++ similarity-search path. _SIM_EXT is None if the
# extension is missing, in which case perform() falls back to the legacy Python path.
_SIM_EXT = None
try:
    import sys as _sys
    if "vllm_clustering_ext" in _sys.modules:
        # clustering.py already loaded the .so under this name; reuse it. The
        # .so's only PyInit matches this build-time name, so loading under any
        # other name (e.g. "..._sim") raises "does not define module export
        # function". Reusing avoids a second exec and the name mismatch.
        _SIM_EXT = _sys.modules["vllm_clustering_ext"]
        _src = "sys.modules (shared with clustering.py)"
    else:
        _ext_dir = os.path.dirname(os.path.abspath(__file__))
        _so_files = [f for f in os.listdir(_ext_dir)
                     if f.startswith("vllm_clustering_ext") and f.endswith(".so")]
        if _so_files:
            import importlib.util
            _so_path = os.path.join(_ext_dir, _so_files[0])
            _spec = importlib.util.spec_from_file_location(
                "vllm_clustering_ext", _so_path)
            _SIM_EXT = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_SIM_EXT)
            _src = _so_files[0]
    if _SIM_EXT is not None and hasattr(_SIM_EXT, "perform_similarity_search"):
        logger.info("[NELSSA] C++ Similarity Search Extension loaded from "
                    "%s (perform_similarity_search)", _src)
    elif _SIM_EXT is not None:
        # Loaded but the symbol is missing -> old .so without the new fn.
        logger.warning("[NELSSA] C++ Extension loaded but has no "
                       "perform_similarity_search symbol (rebuild the .so). "
                       "Using legacy Python path.")
        _SIM_EXT = None
    else:
        logger.warning("[NELSSA] C++ Clustering Extension (.so) not found; "
                       "similarity search will use the legacy Python path.")
except Exception as e:
    logger.warning("[NELSSA] C++ Similarity Search Extension not available: "
                    "%s. Using legacy Python path.", e)
    _SIM_EXT = None


class SimilaritySearchEngine:
    @staticmethod
    def perform(
        query_states: torch.Tensor,
        layer_idx: int,
        req_ids: list[str],
        nprobe: float,
        is_pd_disaggregated: bool,
        get_remote_request_id_fn,
        nelssa_cluster_metadata: dict | None = None,
        nelssa_long_mask_tensor: torch.Tensor | None = None,
        nelssa_long_mask_cpu_list: list[int] | None = None,
        query_start_loc: torch.Tensor | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        nelssa_centroid_staging: torch.Tensor | None = None,
        nelssa_cluster_size_staging: torch.Tensor | None = None,
        nelssa_meta_slot_by_req: dict | None = None,
        sim_cache: dict | None = None,
        sim_persistent_bufs: dict | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """ Returns: Dict mapping request_id to {'cluster_ids', 'topk_indices', 'topk_values'} """
        # Infer dimensions if not provided
        if head_dim is None:
            head_dim = query_states.shape[-1]
        if num_heads is None:
            num_heads = query_states.shape[1] if query_states.dim() > 2 else query_states.shape[1] // head_dim
        if num_kv_heads is None:
            num_kv_heads = num_heads

        if query_start_loc is None:
            return {}

        group_size = num_heads // num_kv_heads
        rsqrt_dim = 1.0 / (head_dim ** 0.5)

        # NELSSA C++ path (default): one C++ call for per-req loop + stack +
        # batched_centroid_search + extraction, removing host-side Python
        # dispatch overhead. NELSSA_SIM_CPP=0 reverts to the legacy Python path.
        # Bit-identical in op sequence/dtype to batched_centroid_search.
        #
        # sim_cache (stashed on the per-step nelssa_forward_context): when
        # present, layer 0 computes everything constant across the 32 layers of
        # a decode step (per-req pre-filter, buffer_key/remote-id mapping, CPU
        # index tensors, pre-stacked centroids) and caches it under 'pf';
        # layers 1..31 reuse 'pf' and only feed per-layer query_states/layer_idx.
        # The cache lives on nelssa_forward_context (rebuilt every step), so
        # it is auto-invalidated between steps.
        _use_cpp = (os.environ.get("NELSSA_SIM_CPP", "1") == "1"
                    and _SIM_EXT is not None)
        if _use_cpp:
            return SimilaritySearchEngine._perform_cpp(
                query_states=query_states,
                layer_idx=layer_idx,
                req_ids=req_ids,
                nprobe=nprobe,
                is_pd_disaggregated=is_pd_disaggregated,
                get_remote_request_id_fn=get_remote_request_id_fn,
                nelssa_cluster_metadata=nelssa_cluster_metadata,
                nelssa_long_mask_cpu_list=nelssa_long_mask_cpu_list,
                nelssa_long_mask_tensor=nelssa_long_mask_tensor,
                query_start_loc=query_start_loc,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                nelssa_centroid_staging=nelssa_centroid_staging,
                nelssa_cluster_size_staging=nelssa_cluster_size_staging,
                nelssa_meta_slot_by_req=nelssa_meta_slot_by_req,
                sim_cache=sim_cache,
                sim_persistent_bufs=sim_persistent_bufs,
            )

        # NELSSA: long-mask and query_start_loc are for Python control-flow
        # only (skip non-long / multi-token), not GPU compute — so use
        # pre-snapshotted CPU data taken once per step to avoid a per-layer
        # .cpu() drain of the GPU mask (32 drains/step). Fall back to the GPU
        # tensors only if the caller didn't supply the CPU snapshots. The
        # mask/loc are fixed for the whole step, so one snapshot serves all
        # 32 layers.
        if nelssa_long_mask_cpu_list is not None:
            long_mask_list = nelssa_long_mask_cpu_list
        elif nelssa_long_mask_tensor is not None:
            long_mask_cpu = (nelssa_long_mask_tensor.bool().cpu()
                             if nelssa_long_mask_tensor.is_cuda
                             else nelssa_long_mask_tensor.bool())
            long_mask_list = long_mask_cpu.tolist()
        else:
            long_mask_list = None
        if query_start_loc.is_cuda:
            query_start_loc_cpu = query_start_loc.cpu()
        else:
            query_start_loc_cpu = query_start_loc
        # int list for O(1) Python indexing below (no per-access GPU sync).
        qsl = query_start_loc_cpu.tolist()

        results: dict[str, dict[str, torch.Tensor]] = {}
        long_req_ids: list[str] = []
        query_inputs: list[torch.Tensor] = []
        centroids_list: list[torch.Tensor] = []
        cluster_sizes_list: list[torch.Tensor] = []
        nprobe_list: list[int] = []
        topk_list: list[int] = []

        for req_idx, req_id in enumerate(req_ids):
            # Map request ID to buffer key (P/D mode uses remote ID)
            buffer_key = (
                get_remote_request_id_fn(req_id) if is_pd_disaggregated else None
            ) or req_id

            # Skip non-long requests (P/D mode) or requests without metadata
            if is_pd_disaggregated:
                if long_mask_list is None or not long_mask_list[req_idx]:
                    continue
            elif not _has_valid_metadata(nelssa_cluster_metadata, buffer_key, layer_idx):
                if layer_idx == 0:
                    logger.warning(
                        "[NELSSA][SIMILARITY] No metadata for %s layer %d, skipping",
                        buffer_key[:20], layer_idx
                    )
                continue

            # Extract query (decode step has exactly 1 token). Index the CPU
            # list snapshot taken once at the top — no per-access GPU sync.
            start_idx = qsl[req_idx]
            end_idx = qsl[req_idx + 1]
            if (end_idx - start_idx) != 1:
                continue

            query_input = _extract_query(query_states[start_idx], num_kv_heads, group_size, head_dim)

            # Get centroids and n_centroids
            if is_pd_disaggregated:
                meta = nelssa_cluster_metadata.get(buffer_key, {}).get(layer_idx)
            else:
                meta = nelssa_cluster_metadata[buffer_key][layer_idx]

            centroids = meta.get('centroids') if meta else None
            if centroids is None or centroids.numel() == 0:
                if layer_idx == 0:
                    logger.warning(
                        "[NELSSA][SIMILARITY] No centroids for %s layer %d, using DUMMY",
                        buffer_key[:20], layer_idx
                    )
                # Dummy result for requests without metadata
                nprobe_int = max(1, round(nprobe))
                dummy_ids = torch.arange(nprobe_int, dtype=torch.int32).unsqueeze(0).expand(num_kv_heads, -1)
                results[buffer_key] = {
                    'cluster_ids': dummy_ids,
                    'cluster_ids_cpu_tensor': dummy_ids.cpu().int(),
                    'topk_indices': dummy_ids,
                    'topk_values': torch.ones_like(dummy_ids, dtype=torch.float32),
                }
                continue

            # Accumulate for batched search
            long_req_ids.append(buffer_key)
            query_inputs.append(query_input)
            centroids_list.append(centroids)
            # NELSSA: use the precomputed empty-cluster mask ('cluster_size_mask',
            # built once at staging) instead of recomputing `== 0` per layer —
            # removes one GPU kernel per layer. Fall back only for older metadata.
            _csm = meta.get('cluster_size_mask')
            if _csm is None:
                _csm = (meta['cluster_size'] == 0)
            cluster_sizes_list.append(_csm)
            req_nprobe = max(round(meta['n_centroids'] * nprobe), 1)
            nprobe_list.append(req_nprobe)
            # NELSSA: request exactly nprobe from topk instead of 2*nprobe + slice.
            # The extra candidates (RetrievalAttention's "estimation zone") aren't
            # consumed here, and torch.topk(sorted=True)[:nprobe] == topk(nprobe), so
            # this halves the topk work (launch-overhead-dominated at batch=1).
            topk_list.append(req_nprobe)

        if not long_req_ids:
            return results

        # Batch similarity search with padding
        results.update(_batched_search(
            query_inputs, centroids_list, cluster_sizes_list,
            nprobe_list, topk_list, long_req_ids, layer_idx,
            num_kv_heads, rsqrt_dim,
        ))

        return results

    @staticmethod
    def _perform_cpp(
        query_states: torch.Tensor,
        layer_idx: int,
        req_ids: list[str],
        nprobe: float,
        is_pd_disaggregated: bool,
        get_remote_request_id_fn,
        nelssa_cluster_metadata: dict | None = None,
        nelssa_long_mask_cpu_list: list[int] | None = None,
        nelssa_long_mask_tensor: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        nelssa_centroid_staging: torch.Tensor | None = None,
        nelssa_cluster_size_staging: torch.Tensor | None = None,
        nelssa_meta_slot_by_req: dict | None = None,
        sim_cache: dict | None = None,
        sim_persistent_bufs: dict | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """C++ path for perform(). Mirrors the legacy perform() + _batched_search
        logic but runs the per-req loop + stack + GPU op sequence + extraction in
        one C++ call (_SIM_EXT.perform_similarity_search) to eliminate the
        host-side Python dispatch overhead.

        Python-only concerns stay here: buffer_key/remote-id mapping, the
        has-metadata check, dummy-results for reqs without centroids. C++ only
        sees the already-eligible long reqs (pure-tensor).

        sim_cache: when supplied, the layer-0 call computes EVERYTHING constant
        across the 32 layers of a decode step and caches it under 'pf'; layers
        1..31 reuse it. See perform()'s docstring for the invariant. When None
        (legacy callers), the pre-filter runs every layer (the old behavior).

        CORRECTNESS CAVEAT: centroids themselves are per-layer (each layer's
        keys are k-means'd separately, see clustering.py), so the prestacked
        path's centroids_prestacked CANNOT be cached — it must be rebuilt per
        layer. Only the staging path is fully cacheable (C++ indexes
        centroid_staging[slot, layer_idx] per layer). For the prestacked path
        (single-GPU, or P/D when a req lacks a slot) we cache the layer-invariant
        parts (qsl, long_mask, buffer_keys, n_centroids, slots) and rebuild
        ONLY the per-layer prestacked centroids each call."""
        # ---- per-step pre-filter (cached across layers when sim_cache given) ----
        # The long_mask list, qsl, buffer_keys, n_centroids, slot eligibility are
        # identical across the 32 layers of a decode step (depend on step batch,
        # not layer_idx). Compute once at layer 0 ('pf') and reuse — avoids the
        # per-layer host prep (tensor construction + pre-filter dict loop). 'pf'
        # lives on the step-scoped nelssa_forward_context so it auto-rebuilds.
        pf = sim_cache.get('pf') if sim_cache is not None else None
        if pf is None:
            pf = SimilaritySearchEngine._build_prefilter(
                req_ids=req_ids,
                nprobe=nprobe,
                is_pd_disaggregated=is_pd_disaggregated,
                get_remote_request_id_fn=get_remote_request_id_fn,
                nelssa_cluster_metadata=nelssa_cluster_metadata,
                nelssa_long_mask_cpu_list=nelssa_long_mask_cpu_list,
                nelssa_long_mask_tensor=nelssa_long_mask_tensor,
                query_start_loc=query_start_loc,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                nelssa_centroid_staging=nelssa_centroid_staging,
                nelssa_cluster_size_staging=nelssa_cluster_size_staging,
                nelssa_meta_slot_by_req=nelssa_meta_slot_by_req,
                max_n_centroids=(sim_persistent_bufs.get('scores').size(2)
                              if sim_persistent_bufs and sim_persistent_bufs.get('scores', 0).numel() > 0
                              else None),
                pad_mask_buf=sim_persistent_bufs.get('pad_mask') if sim_persistent_bufs else None,
            )
            if sim_cache is not None:
                sim_cache['pf'] = pf

        # Dummy results for reqs without centroids are layer-independent and
        # were folded into pf['dummy_results'] at build time.
        results: dict[str, dict[str, torch.Tensor]] = dict(pf['dummy_results'])

        if pf['num_long'] == 0:
            return results

        # Per-layer centroids for the PRESTACKED path only (single-GPU, or P/D
        # fallback when a req lacks a staging slot). Centroids are per-layer
        # (clustering runs k-means per layer), so they cannot be cached. The
        # staging path skips this — C++ indexes centroid_staging[slot,layer_idx]
        # per layer, which already holds the right slice.
        centroids_prestacked = pf['centroids_prestacked']
        cluster_size_prestacked = pf['cluster_size_prestacked']
        if not pf['use_staging_batch'] and pf['needs_per_layer_restack']:
            # Move 3: pass the persistent batch buffers so _restack_per_layer
            # fills them directly; C++ then reads them as-is (no double copy).
            pb = sim_persistent_bufs or {}
            centroids_prestacked, cluster_size_prestacked = \
                SimilaritySearchEngine._restack_per_layer(
                    pf=pf, layer_idx=layer_idx,
                    nelssa_cluster_metadata=nelssa_cluster_metadata,
                    num_kv_heads=num_kv_heads,
                    centroids_batch_buf=pb.get('centroids_batch'),
                    cluster_size_batch_buf=pb.get('cluster_size_batch'))

        # ---- call C++ (per-layer: only query_states + layer_idx vary) ----
        # Persistent scratch/output buffers (CUDA-Graph path). When provided
        # the C++ op sequence uses _out variants into these fixed-address
        # buffers (zero allocation). When None, C++ falls back to allocating.
        pb = sim_persistent_bufs or {}
        topk_indices_flat, topk_values_flat, nprobe_per_req, topk_per_req, long_req_idx_out = (
            _SIM_EXT.perform_similarity_search(
                query_states,
                pf['qsl_cpu_t'],
                pf['long_mask_cpu'],
                num_heads, num_kv_heads, head_dim,
                float(nprobe),
                pf['centroid_staging'],
                pf['cluster_size_staging'],
                pf['slots_cpu'],
                centroids_prestacked,
                cluster_size_prestacked,
                pf['n_centroids_cpu'],
                pf['pad_mask_bh'],
                pb.get('scores', torch.empty(0)),
                pb.get('sm', torch.empty(0)),
                pb.get('dist', torch.empty(0)),
                pb.get('topk_values', torch.empty(0)),
                pb.get('topk_indices', torch.empty(0)),
                pb.get('query_batch', torch.empty(0)),
                pb.get('centroids_batch', torch.empty(0)),
                pb.get('cluster_size_batch', torch.empty(0)),
                pb.get('cmask', torch.empty(0)),
                layer_idx,
                pf['num_layers'],
            )
        )

        # ---- reconstruct the per-req dict (views, no copy) ----
        import time as _time
        _dtoh_ms = 0.0
        buffer_keys = pf['buffer_keys']
        out_list = long_req_idx_out.tolist()
        for j, req_idx in enumerate(out_list):
            base = j * num_kv_heads
            tk = int(topk_per_req[j].item())
            npr = int(nprobe_per_req[j].item())
            ri = topk_indices_flat[base:base + num_kv_heads, :tk]
            rv = topk_values_flat[base:base + num_kv_heads, :tk]
            cluster_ids = ri[:, :npr]   # == get_topk_cluster_ids (view)
            # DtoH the contiguous full-width slice and cut on the CPU (the [:npr]
            # slice is non-contiguous — .cpu() on it would materialize first).
            _ri_full = topk_indices_flat[base:base + num_kv_heads]   # contiguous [kvH, max_k]
            _t0 = _time.perf_counter()
            cluster_ids_cpu_tensor = _ri_full.cpu()[:, :npr].to(torch.int32)
            _dtoh_ms += (_time.perf_counter() - _t0) * 1e3
            results[buffer_keys[j]] = {
                'cluster_ids': cluster_ids,
                'cluster_ids_cpu_tensor': cluster_ids_cpu_tensor,
                'topk_indices': ri,
                'topk_values': rv,
            }

        # Record the DtoH wait time for this layer on the runner's collector so
        # the periodic ATTN-TIMING log can attribute the SimSearch CPU/GPU gap.
        _sim_dtoh_collector = getattr(
            SimilaritySearchEngine, '_sim_dtoh_collector', None)
        if _sim_dtoh_collector is not None:
            _sim_dtoh_collector.append(_dtoh_ms)

        return results

    @staticmethod
    def _build_prefilter(
        req_ids: list[str],
        nprobe: float,
        is_pd_disaggregated: bool,
        get_remote_request_id_fn,
        nelssa_cluster_metadata: dict | None,
        nelssa_long_mask_cpu_list: list[int] | None,
        nelssa_long_mask_tensor: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        num_kv_heads: int,
        head_dim: int,
        nelssa_centroid_staging: torch.Tensor | None,
        nelssa_cluster_size_staging: torch.Tensor | None,
        nelssa_meta_slot_by_req: dict | None,
        max_n_centroids: int | None = None,
        pad_mask_buf: torch.Tensor | None = None,
    ) -> dict:
        """Compute everything constant across the 32 layers of a decode step.

        Returns a dict with the pre-filter outputs + ready-to-pass C++ input
        tensors. Layer-independent: long_mask, qsl, buffer_key/remote-id
        mapping, per-req n_centroids, staging-slot eligibility, dummy results
        for centroid-less reqs, and (single-GPU / fallback) pre-stacked
        centroids/cluster_size. The C++ call still needs per-layer
        query_states + layer_idx, which the caller supplies."""
        # Resolve the long-mask CPU list (same logic as perform's top).
        if nelssa_long_mask_cpu_list is not None:
            long_mask_list = nelssa_long_mask_cpu_list
        elif nelssa_long_mask_tensor is not None:
            long_mask_cpu = (nelssa_long_mask_tensor.bool().cpu()
                             if nelssa_long_mask_tensor.is_cuda
                             else nelssa_long_mask_tensor.bool())
            long_mask_list = long_mask_cpu.tolist()
        else:
            long_mask_list = None
        if query_start_loc.is_cuda:
            query_start_loc_cpu = query_start_loc.cpu()
        else:
            query_start_loc_cpu = query_start_loc
        qsl = query_start_loc_cpu.tolist()

        # ---- Python pre-filter: build the eligible long-req list ----
        # buffer_key / has-metadata / dummy-results are string/dict concerns
        # that belong in Python. C++ only gets the pure-tensor view.
        dummy_results: dict[str, dict[str, torch.Tensor]] = {}
        long_req_indices: list[int] = []      # req_idx into req_ids
        buffer_keys: list[str] = []
        n_centroids_list: list[int] = []
        # single-GPU (no staging buffer): collect per-req centroid/cluster_size
        # tensors to pre-stack. P/D skips these (C++ indexes staging by slot).
        cent_list_single: list[torch.Tensor] = []
        csize_list_single: list[torch.Tensor] = []

        use_staging = (nelssa_centroid_staging is not None
                       and nelssa_centroid_staging.numel() > 0)
        # All eligible long reqs must have a staging slot to use the staging
        # path. If any lacks one, fall back to the prestack (legacy) path for
        # the whole batch (see the per-req slot_ok note below).
        all_slots_ok = True

        for req_idx, req_id in enumerate(req_ids):
            buffer_key = (
                get_remote_request_id_fn(req_id) if is_pd_disaggregated else None
            ) or req_id

            # Skip non-long (P/D) or without-metadata (single-GPU).
            if is_pd_disaggregated:
                if long_mask_list is None or not long_mask_list[req_idx]:
                    continue
            elif not _has_valid_metadata(nelssa_cluster_metadata, buffer_key, 0):
                continue

            # decode == 1 token
            start_idx = qsl[req_idx]
            end_idx = qsl[req_idx + 1]
            if (end_idx - start_idx) != 1:
                continue

            # Fetch centroids/metadata; handle dummy (no centroids) in Python.
            # NOTE: for the cached pre-filter we probe layer 0's metadata as a
            # proxy for "this req has centroids at all". The per-layer meta is
            # fetched lazily inside C++ via the staging/prestacked tensors, so
            # we don't need per-layer meta here.
            if is_pd_disaggregated:
                meta = (nelssa_cluster_metadata.get(buffer_key, {})
                        if nelssa_cluster_metadata else {}).get(0)
            else:
                meta = (nelssa_cluster_metadata.get(buffer_key, {})
                        .get(0)) if nelssa_cluster_metadata else None
            centroids = meta.get('centroids') if meta else None
            if centroids is None or centroids.numel() == 0:
                # dummy result (matches legacy perform's dummy path)
                nprobe_int = max(1, round(nprobe)) if meta else 1
                dummy_ids = torch.arange(
                    nprobe_int, dtype=torch.int32
                ).unsqueeze(0).expand(num_kv_heads, -1)
                dummy_results[buffer_key] = {
                    'cluster_ids': dummy_ids,
                    'cluster_ids_cpu_tensor': dummy_ids.cpu().int(),       #  raw-bytes RPC path.
                    'topk_indices': dummy_ids,
                    'topk_values': torch.ones_like(dummy_ids, dtype=torch.float32),
                }
                continue

            long_req_indices.append(req_idx)
            buffer_keys.append(buffer_key)
            # staging-path eligibility: a req can use the staging buffer ONLY if
            # its buffer_key has a registered slot in _nelssa_meta_slot_by_req.
            # The slot is registered in _store_received_nelssa_metadata ONLY when
            # kv_transfer_params.remote_slot_idx is present at store time. A req
            # may have centroids in nelssa_cluster_metadata (so the legacy path,
            # which reads the dict clone, works fine) but no slot (e.g. the NIXL
            # READ completed and centroids were cloned into the dict, yet the
            # slot wasn't recorded, or the remote_request_id differs between
            # store time and decode time so the buffer_key differs). Querying a
            # missing slot would KeyError -> crash. So we fall back to the
            # prestack path (which clones from the dict, exactly what legacy
            # does) for the WHOLE batch if any eligible req lacks a slot. This
            # preserves legacy semantics: correctness first, staging is an opt
            # that only applies when every long req has a slot.
            slot_ok = (not use_staging) or (
                nelssa_meta_slot_by_req is not None
                and buffer_key in nelssa_meta_slot_by_req)
            all_slots_ok = all_slots_ok and slot_ok
            # Always collect centroids/cluster_size from the dict so the
            # prestack path is ready if any req lacks a slot. When every req
            # has a slot, the staging path is used instead (n_centroids only).
            cent_list_single.append(centroids)
            _cs = meta.get('cluster_size')
            csize_list_single.append(_cs if _cs is not None else meta['centroids'].new_zeros(centroids.shape[:2]))
            n_centroids_list.append(int(meta.get('n_centroids', centroids.shape[1])))

        num_long = len(long_req_indices)

        # ---- build C++ inputs ----
        long_mask_cpu = torch.tensor(
            [1 if i in set(long_req_indices) else 0 for i in range(len(req_ids))],
            dtype=torch.int8,
        )
        # qsl as int64 CPU tensor for the C++ side (handles int32/int64).
        qsl_cpu_t = torch.tensor(qsl, dtype=torch.int64)
        # n_centroids/slots as int64 so the C++ side skips its per-layer
        # toType(kInt64) conversion (these are layer-invariant, converted once
        # at layer 0 here and reused for layers 1..31).
        n_centroids_cpu = torch.tensor(n_centroids_list, dtype=torch.int64)

        empty = torch.empty(0, dtype=torch.bfloat16)
        empty_i32 = torch.empty(0, dtype=torch.int32)
        empty_i32_cpu = torch.empty(0, dtype=torch.int32)
        empty_i64_cpu = torch.empty(0, dtype=torch.int64)
        # Use the staging path only when every eligible long req has a slot.
        # Otherwise fall back to prestack (clone from the dict, == legacy).
        use_staging_batch = use_staging and all_slots_ok
        if use_staging_batch:
            # slots: map each eligible long req's buffer_key to its staging slot.
            slots_cpu = torch.tensor(
                [nelssa_meta_slot_by_req[bk] for bk in buffer_keys],
                dtype=torch.int64,
            )
            centroid_staging = nelssa_centroid_staging
            cluster_size_staging = nelssa_cluster_size_staging
            centroids_prestacked = empty
            cluster_size_prestacked = empty_i32
            needs_per_layer_restack = False
            max_nc_prestack = 0
        else:
            slots_cpu = empty_i64_cpu
            centroid_staging = empty
            cluster_size_staging = empty_i32
            # pre-stack centroids (padded to max). Used for single-GPU AND for
            # P/D when any req lacks a staging slot (legacy-equivalent clone).
            # NOTE: centroids are per-layer (clustering k-means per layer), so
            # this layer-0 prestack is only a placeholder; _restack_per_layer
            # rebuilds it for each layer_idx. We still build it here so the
            # non-cached path (sim_cache None) and layer 0 work without extra
            # branching, and so max_nc_prestack / padding info is captured.
            max_nc_prestack = max(c.shape[1] for c in cent_list_single)
            num_kv_h = num_kv_heads
            if any(c.shape[1] != max_nc_prestack for c in cent_list_single):
                padded_c, padded_cs = _pad_to_max(
                    cent_list_single, csize_list_single, max_nc_prestack, num_kv_h)
            else:
                padded_c, padded_cs = cent_list_single, csize_list_single
            centroids_prestacked = torch.stack(padded_c, dim=0)
            cluster_size_prestacked = torch.stack(padded_cs, dim=0)
            # num_kv_heads>1 in single-GPU mode → per-layer centroids differ, so
            # we must restack each layer. (When num_long>=1 and not staging, the
            # restack path runs every layer; the layer-0 stack above is reused
            # only when sim_cache is None.)
            needs_per_layer_restack = True

        num_layers = (nelssa_centroid_staging.size(1)
                      if (use_staging_batch and nelssa_centroid_staging is not None)
                      else 1)

        # Layer-invariant pad mask: precompute once here so the 32 per-layer C++
        # calls skip the arange + nc_per_req.to(device) (HtoD) + >= + expand
        # kernels. A centroid position is padded (masked True) iff its column
        # index >= that req's real n_centroids. Depends only on n_centroids_vec
        # (per-req, fixed across layers) and the uniform padded width n_cent
        # (staging width or max_nc_prestack — both fixed per step), so it is
        # identical across the 32 layers and safe to cache. The layer-dependent
        # empty-cluster mask (cluster_size==0) is still computed per layer in C++.
        pad_mask_bh = torch.empty(0, dtype=torch.bool)
        if num_long > 0:
            # Build the pad mask at the persistent buffer width (max_n_centroids)
            # so the C++ cmask path skips its per-layer ones()+copy_ padding.
            if use_staging_batch:
                _n_cent = nelssa_centroid_staging.size(3)
                _dev = nelssa_centroid_staging.device
            else:
                _n_cent = max_nc_prestack
                _dev = (centroids_prestacked.device
                        if centroids_prestacked.numel() > 0 else torch.device('cpu'))
            if max_n_centroids is not None:
                _n_cent = max_n_centroids
            _mask_range = torch.arange(_n_cent, dtype=torch.int64, device=_dev)
            _nc = n_centroids_cpu.to(_dev).to(torch.int64).view(num_long, 1)
            _pad_mask = _mask_range.unsqueeze(0) >= _nc          # [num_long, n_cent]
            _pad_mask_bh = _pad_mask.unsqueeze(1).expand(
                num_long, num_kv_heads, _n_cent)                  # [num_long, kvH, n_cent]
            # Move 2a: write into a fixed-address contiguous persistent buffer so
            # the captured graph can read it (inside eq_out/bitwise_or_) without a
            # materialize kernel.
            if (pad_mask_buf is not None
                    and pad_mask_buf.numel() > 0
                    and pad_mask_buf.size(0) >= num_long
                    and pad_mask_buf.size(2) == _n_cent):
                pad_mask_buf[:num_long].copy_(_pad_mask_bh)
                pad_mask_bh = pad_mask_buf[:num_long]            # contiguous fixed view
            else:
                pad_mask_bh = _pad_mask_bh.contiguous()          # fallback: fresh tensor

        return {
            'dummy_results': dummy_results,
            'num_long': num_long,
            'buffer_keys': buffer_keys,
            'long_mask_cpu': long_mask_cpu,
            'qsl_cpu_t': qsl_cpu_t,
            'n_centroids_cpu': n_centroids_cpu,
            'use_staging_batch': use_staging_batch,
            'slots_cpu': slots_cpu,
            'centroid_staging': centroid_staging,
            'cluster_size_staging': cluster_size_staging,
            'centroids_prestacked': centroids_prestacked,
            'cluster_size_prestacked': cluster_size_prestacked,
            'num_layers': num_layers,
            'needs_per_layer_restack': needs_per_layer_restack,
            'max_nc_prestack': max_nc_prestack,
            'is_pd_disaggregated': is_pd_disaggregated,
            'pad_mask_bh': pad_mask_bh,
        }

    @staticmethod
    def _restack_per_layer(
        pf: dict,
        layer_idx: int,
        nelssa_cluster_metadata: dict | None,
        num_kv_heads: int,
        centroids_batch_buf: torch.Tensor | None = None,
        cluster_size_batch_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild the prestacked centroids/cluster_size for a specific layer.

        Centroids are per-layer (k-means per layer), so the layer-0 prestack in
        pf cannot be reused. Move 3: when persistent batch buffers are provided,
        copy each per-layer centroid/cluster_size directly into the fixed-
        address buffer slices (tail columns masked by the pad mask, no zeroing
        needed) and return empty tensors; C++ then reads the buffers as-is,
        halving the per-layer copy count (4 -> 2)."""
        buffer_keys = pf['buffer_keys']
        is_pd = pf.get('is_pd_disaggregated', False)
        max_nc = pf['max_nc_prestack']
        direct_to_buf = (centroids_batch_buf is not None
                         and centroids_batch_buf.numel() > 0
                         and cluster_size_batch_buf is not None
                         and cluster_size_batch_buf.numel() > 0)
        cent_list: list[torch.Tensor] = []
        csize_list: list[torch.Tensor] = []
        for j, buffer_key in enumerate(buffer_keys):
            if is_pd:
                meta = (nelssa_cluster_metadata.get(buffer_key, {})
                        if nelssa_cluster_metadata else {}).get(layer_idx)
            else:
                meta = (nelssa_cluster_metadata.get(buffer_key, {})
                        .get(layer_idx)) if nelssa_cluster_metadata else None
            centroids = meta.get('centroids') if meta else None
            if centroids is None or centroids.numel() == 0:
                # Should not happen: the layer-0 prefilter already routed
                # centroid-less reqs to dummy_results. If a later layer lacks
                # centroids, fall back to a zero slice (will be fully masked
                # by the pad_mask, so topk picks nothing real — safe).
                _zero = torch.zeros(num_kv_heads, max_nc, 1,
                                    dtype=torch.bfloat16,
                                    device=(centroids.device if centroids is not None
                                            else torch.device('cpu')))
                if direct_to_buf:
                    _dev = centroids_batch_buf.device
                    _zero_g = torch.zeros(num_kv_heads, centroids_batch_buf.size(2), 1,
                                          dtype=torch.bfloat16, device=_dev)
                    centroids_batch_buf[j].copy_(_zero_g)
                    cluster_size_batch_buf[j].zero_()
                else:
                    cent_list.append(_zero)
                    csize_list.append(torch.zeros(num_kv_heads, max_nc, dtype=torch.int32))
                continue
            _cs = meta.get('cluster_size')
            if _cs is None:
                _cs = meta['centroids'].new_zeros(centroids.shape[:2])
            if direct_to_buf:
                # Direct narrow-copy into the fixed buffer; tail columns are
                # masked by the pad mask, no zeroing needed (Move 1b safety net).
                c_buf = centroids_batch_buf[j]           # [kvH, max_n_cent, D]
                if centroids.shape[1] >= c_buf.size(1):
                    c_buf.copy_(centroids)
                else:
                    c_buf.narrow(1, 0, centroids.shape[1]).copy_(centroids)
                cs_buf = cluster_size_batch_buf[j]       # [kvH, max_n_cent]
                if _cs.shape[1] >= cs_buf.size(1):
                    cs_buf.copy_(_cs)
                else:
                    cs_buf.narrow(1, 0, _cs.shape[1]).copy_(_cs)
            else:
                # Legacy torch::stack path (non-graph / no persistent buffers).
                cent_list.append(centroids)
                csize_list.append(_cs)
                # Pad to max_nc if this layer's n_centroids differs (clustering
                # can produce slightly different n_centroids per layer when
                # token counts differ, though in practice they match).
                if centroids.shape[1] < max_nc:
                    c_pad = torch.zeros(num_kv_heads, max_nc, centroids.shape[2],
                                        dtype=centroids.dtype, device=centroids.device)
                    c_pad[:, :centroids.shape[1], :].copy_(centroids)
                    cent_list[-1] = c_pad
                    if _cs.dtype == torch.bool:
                        s_pad = torch.ones(num_kv_heads, max_nc, dtype=_cs.dtype,
                                           device=_cs.device)
                    else:
                        s_pad = torch.zeros(num_kv_heads, max_nc, dtype=_cs.dtype,
                                            device=_cs.device)
                    s_pad[:, :_cs.shape[1]].copy_(_cs)
                    csize_list[-1] = s_pad
        if direct_to_buf:
            # Signal "buffers filled directly" to C++ via empty tensors. C++
            # reads centroids_batch_buf / cluster_size_batch_buf as-is.
            empty_c = torch.empty(0, dtype=torch.bfloat16,
                                  device=centroids_batch_buf.device)
            empty_cs = torch.empty(0, dtype=torch.int32,
                                   device=cluster_size_batch_buf.device)
            return empty_c, empty_cs
        return torch.stack(cent_list, dim=0), torch.stack(csize_list, dim=0)


def _has_valid_metadata(
    metadata: dict | None,
    buffer_key: str,
    layer_idx: int,
) -> bool:
    """Check if clustering metadata exists for the given request and layer."""
    return (
        metadata is not None
        and buffer_key in metadata
        and layer_idx in metadata[buffer_key]
    )


def _extract_query(
    query_state: torch.Tensor,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Reshape query state to [num_kv_heads, 1, group_size, head_dim]."""
    return query_state.view(1, num_kv_heads, group_size, head_dim).permute(1, 0, 2, 3).contiguous()


def _batched_search(
    query_inputs: list[torch.Tensor],
    centroids_list: list[torch.Tensor],
    cluster_sizes_list: list[torch.Tensor],
    nprobe_list: list[int],
    topk_list: list[int],
    long_req_ids: list[str],
    layer_idx: int,
    num_kv_heads: int,
    rsqrt_dim: float,
) -> dict[str, dict[str, torch.Tensor]]:
    results: dict[str, dict[str, torch.Tensor]] = {}

    # Pad centroids/cluster_sizes to max n_centroids
    max_n_centroids = max(c.shape[1] for c in centroids_list)
    if any(c.shape[1] != max_n_centroids for c in centroids_list):
        centroids_list, cluster_sizes_list = _pad_to_max(centroids_list, cluster_sizes_list, max_n_centroids, num_kv_heads)

    # Stack into batch tensors
    query_batch = torch.stack(query_inputs, dim=0)
    centroids_batch = torch.stack(centroids_list, dim=0)
    cluster_sizes_batch = torch.stack(cluster_sizes_list, dim=0)
    max_topk = max(topk_list)

    # Batched centroid search
    topk_indices_batch, topk_values_batch = batched_centroid_search(
        queries=query_batch, centroids=centroids_batch, cluster_sizes=cluster_sizes_batch,
        topk=max_topk, rsqrt_dim=rsqrt_dim,
    )

    # Extract per-request results
    for i, req_id in enumerate(long_req_ids):
        base = i * num_kv_heads
        req_topk = topk_list[i]
        req_topk_indices = topk_indices_batch[base:base + num_kv_heads, :req_topk]
        req_topk_values = topk_values_batch[base:base + num_kv_heads, :req_topk]
        cluster_ids = get_topk_cluster_ids(req_topk_indices, nprobe_list[i])

        results[req_id] = {
            'cluster_ids': cluster_ids,
            # CPU long tensor for the raw-bytes RPC path
            'cluster_ids_cpu_tensor': cluster_ids.cpu().int(),
            'topk_indices': req_topk_indices,
            'topk_values': req_topk_values,
        }

    return results


def _pad_to_max(
    centroids_list: list[torch.Tensor],
    cluster_sizes_list: list[torch.Tensor],
    max_n_centroids: int,
    num_kv_h: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    padded_centroids: list[torch.Tensor] = []
    padded_cluster_sizes: list[torch.Tensor] = []
    head_d = centroids_list[0].shape[2]

    for c, s in zip(centroids_list, cluster_sizes_list):
        if c.shape[1] < max_n_centroids:
            pad_c = torch.zeros(num_kv_h, max_n_centroids, head_d, dtype=c.dtype, device=c.device)
            pad_c[:, :c.shape[1], :].copy_(c)
            # Padded positions aren't real clusters — mark them EMPTY so topk
            # can't pick a padded centroid. The mask is precomputed bool, so fill
            # pad with True (ones) when dtype is bool.
            if s.dtype == torch.bool:
                pad_s = torch.ones(num_kv_h, max_n_centroids, dtype=s.dtype, device=s.device)
            else:
                pad_s = torch.zeros(num_kv_h, max_n_centroids, dtype=s.dtype, device=s.device)
            pad_s[:, :s.shape[1]].copy_(s)
            padded_centroids.append(pad_c)
            padded_cluster_sizes.append(pad_s)
        else:
            padded_centroids.append(c)
            padded_cluster_sizes.append(s)

    return padded_centroids, padded_cluster_sizes
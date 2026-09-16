"""
test_nelssa_node.py

NELSSA 노드 연산 정합성 검증 테스트
- NelssaClient 를 직접 생성하고 nelssa_comm 함수를 사용하여 KV 적재 및 연산
- GPU 에서 간단한 sparse attention 연산으로 결과 비교
- LLM 모델이나 nelssa_cache.py 사용하지 않음
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from nelssa_comm.NelssaClient import NelssaClient


def set_seed(seed=2025):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_random_kv_data(
    batch_size, seq_len, kv_heads, head_dim, dtype=torch.float16, device="cuda"
):
    """
    Random KV 데이터 생성
    """
    key_states = torch.randn(
        (batch_size, seq_len, kv_heads, head_dim), dtype=dtype, device=device
    )
    value_states = torch.randn(
        (batch_size, seq_len, kv_heads, head_dim), dtype=dtype, device=device
    )
    return key_states, value_states


def cluster_kv_data(key_states, value_states, n_centroids, n_segments):
    """
    KV 데이터를 클러스터링
    NELSSA 는 연속적인 토큰 블록을 클러스터로 사용한다고 가정
    예: cluster 0 = tokens [0:16], cluster 1 = tokens [16:32], ...
    """
    batch_size, seq_len, kv_heads, head_dim = key_states.shape

    # Reshape: [batch_size*kv_heads, seq_len, head_dim]
    keys_reshaped = key_states.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )
    values_reshaped = value_states.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )

    # Compute centroids as mean of each chunk
    # Each cluster covers seq_len / n_centroids tokens
    tokens_per_cluster = seq_len // n_centroids
    centroids = []
    clusters = []
    cluster_sizes = []

    for bdx in range(batch_size * kv_heads):
        centroids_list = []
        clusters_list = []
        sizes_list = []

        for c_idx in range(n_centroids):
            start = c_idx * tokens_per_cluster
            end = start + tokens_per_cluster

            # Centroid: mean of tokens in this chunk
            chunk_keys = keys_reshaped[bdx, start:end]  # [tokens_per_cluster, head_dim]
            centroid = chunk_keys.mean(dim=0)  # [head_dim]
            centroids_list.append(centroid)

            # Cluster: token indices in this chunk
            token_indices = torch.arange(
                start, end, dtype=torch.int32, device=key_states.device
            )
            clusters_list.append(token_indices)
            sizes_list.append(tokens_per_cluster)

        centroids.append(torch.stack(centroids_list))  # [n_centroids, head_dim]
        clusters.append(torch.stack(clusters_list))  # [n_centroids, tokens_per_cluster]
        cluster_sizes.append(torch.tensor(sizes_list, dtype=torch.int32, device=key_states.device))

    centroids = torch.stack(centroids)  # [batch_size*kv_heads, n_centroids, head_dim]
    clusters = torch.stack(clusters)  # [batch_size*kv_heads, n_centroids, tokens_per_cluster]
    cluster_sizes = torch.stack(cluster_sizes)  # [batch_size*kv_heads, n_centroids]

    # value_sum: sum of values in each cluster
    value_sum = []
    for bdx in range(batch_size * kv_heads):
        value_sum_list = []
        for c_idx in range(n_centroids):
            start = c_idx * tokens_per_cluster
            end = start + tokens_per_cluster
            chunk_values = values_reshaped[bdx, start:end]
            value_sum_list.append(chunk_values.sum(dim=0))
        value_sum.append(torch.stack(value_sum_list))
    value_sum = torch.stack(value_sum)  # [batch_size*kv_heads, n_centroids, head_dim]

    return centroids, value_sum, clusters, cluster_sizes


def prepare_kv_for_nelssa(key_states, value_states, cluster_size):
    """
    NelssaClient.send_kv_cache 에 맞게 KV 데이터 준비
    """
    # Reshape to [batch_size, kv_heads, seq_len, head_dim]
    k_cache = key_states.transpose(1, 2).contiguous().cpu()
    v_cache = value_states.transpose(1, 2).contiguous().cpu()
    size_cache = cluster_size.contiguous().cpu()

    return k_cache, v_cache, size_cache


def sparse_attention_gpu(
    query, keys, values, centroids, clusters, cluster_size, nprobe
):
    """
    GPU 에서 sparse attention 연산 - 선택된 클러스터의 KV 만 사용

    Args:
        query: [batch_size, 1, num_heads, head_dim]
        keys: [batch_size, seq_len, kv_heads, head_dim]
        values: [batch_size, seq_len, kv_heads, head_dim]
        centroids: [batch_size*kv_heads, n_centroids, head_dim]
        clusters: [batch_size*kv_heads, n_centroids, max_cluster_size] - 토큰 인덱스
        cluster_size: [batch_size*kv_heads, n_centroids]
        nprobe: number of clusters to retrieve

    Returns:
        output: [batch_size, 1, num_heads, head_dim]
        cluster_ids: [batch_size*kv_heads, nprobe] - 선택된 클러스터 ID
    """
    batch_size, _, num_heads, head_dim = query.shape
    _, seq_len, kv_heads, _ = keys.shape
    group_size = num_heads // kv_heads
    n_centroids = centroids.shape[1]

    device = query.device
    scale = 1.0 / math.sqrt(head_dim)

    # Reshape query for grouping: [batch_size*kv_heads, 1, group_size, head_dim]
    query_grouped = query.view(batch_size, 1, kv_heads, group_size, head_dim)
    query_grouped = query_grouped.transpose(1, 2).reshape(
        batch_size * kv_heads, 1, group_size, head_dim
    )

    # Compute similarity with centroids
    similarity = torch.matmul(
        query_grouped,  # [B*H, 1, group_size, head_dim]
        centroids.unsqueeze(1).transpose(-2, -1),  # [B*H, 1, head_dim, n_centroids]
    )  # [B*H, 1, group_size, n_centroids]
    similarity = similarity.squeeze(1)  # [B*H, group_size, n_centroids]

    # Select top-k clusters per kv_head group
    # similarity: [B*H, group_size, n_centroids]
    # NELSSA uses the same cluster_ids for all query groups within a kv_head
    # Use the first group's similarity for cluster selection
    topk_values, topk_indices = torch.topk(similarity[:, 0, :], nprobe, dim=-1)
    # topk_indices: [B*H, nprobe] where B*H = batch_size * kv_heads

    # Debug: Print similarity stats for first batch_group
    print(f"  Debug: similarity[0, 0, :5] = {similarity[0, 0, :5]}")
    print(f"  Debug: topk cluster_ids[0] = {topk_indices[0]}")

    # Reshape KV for gathering: [batch_size*kv_heads, seq_len, head_dim]
    keys_reshaped = keys.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )
    values_reshaped = values.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )

    # Debug: Print first few token values that will be gathered
    print("  Debug: First tokens to be gathered (for bdx=0):")
    for c_idx in range(min(3, nprobe)):
        cluster_id = topk_indices[0, c_idx].item()
        size = cluster_size[0, cluster_id].item()
        if size > 0:
            token_idx = clusters[0, cluster_id, 0].item()
            key_val = keys_reshaped[0, token_idx, :4].tolist()
            print(f"    Cluster {cluster_id}: token={token_idx}, key[:4]={key_val}")

    # Gather KV from selected clusters
    # topk_indices: [B*H, nprobe] where B*H = batch_size * kv_heads
    # Same cluster_ids for all query groups within a kv_head

    batch_groups = batch_size * kv_heads
    all_outputs = []

    # Debug: Track first output for debugging
    first_output_done = False

    for bdx in range(batch_groups):
        # For each batch/head group
        query_b = query_grouped[bdx]  # [1, group_size, head_dim]
        cluster_ids_b = topk_indices[bdx]  # [nprobe] - same for all groups

        for g in range(group_size):
            # Get selected cluster IDs (same for all groups)
            selected_clusters = cluster_ids_b  # [nprobe]

            # Gather tokens from each selected cluster
            cluster_keys = []
            cluster_values = []

            for c_idx in range(nprobe):
                cluster_id = selected_clusters[c_idx].item()
                size = cluster_size[bdx, cluster_id].item()

                # Get tokens from this cluster
                # clusters[bdx, cluster_id, :size] contains token indices
                token_indices = clusters[bdx, cluster_id, :size]

                cluster_k = keys_reshaped[bdx, token_indices]  # [size, head_dim]
                cluster_v = values_reshaped[bdx, token_indices]  # [size, head_dim]

                cluster_keys.append(cluster_k)
                cluster_values.append(cluster_v)

            # Concatenate all selected tokens
            if cluster_keys:
                all_k = torch.cat(cluster_keys, dim=0)  # [total_tokens, head_dim]
                all_v = torch.cat(cluster_values, dim=0)  # [total_tokens, head_dim]

                # Compute attention
                q = query_b[0, g]  # [head_dim]
                attn_scores = (
                    torch.matmul(q.unsqueeze(0), all_k.transpose(0, 1)).squeeze(0)
                    * scale
                )  # [total_tokens]

                attn_probs = torch.softmax(attn_scores, dim=-1)
                output_g = torch.matmul(attn_probs.unsqueeze(0), all_v).squeeze(0)

                # Debug: Print first output details
                if not first_output_done and bdx == 0 and g == 0:
                    print(f"  Debug: query[0,0,:4] = {q[:4]}")
                    print(f"  Debug: all_k shape = {all_k.shape}")
                    print(f"  Debug: all_k[:2,:4] = {all_k[:2, :4]}")
                    print(f"  Debug: attn_scores[:5] = {attn_scores[:5]}")
                    print(f"  Debug: attn_probs[:5] = {attn_probs[:5]}")
                    print(f"  Debug: output_g[:4] = {output_g[:4]}")
                    first_output_done = True

                all_outputs.append(output_g)

    # Stack all outputs
    if all_outputs:
        output_grouped = torch.stack(all_outputs, dim=0).reshape(
            batch_groups, group_size, head_dim
        )
    else:
        output_grouped = torch.zeros(
            batch_groups, group_size, head_dim, device=device, dtype=query.dtype
        )

    # Reshape back: [batch_size, 1, num_heads, head_dim]
    output = output_grouped.view(batch_size, kv_heads, group_size, head_dim)
    output = output.transpose(1, 2).reshape(batch_size, 1, num_heads, head_dim)

    return output, topk_indices


def compare_results(nelssa_output, reference_output, atol=1e-2, rtol=1e-2):
    """
    Nelssa 결과와 reference 결과 비교
    """
    all_passed = True

    nelssa_cpu = nelssa_output.cpu()
    ref_cpu = reference_output.cpu()

    # Debug: Print more details about the mismatch
    print(f"  Debug: NELSSA output[0,0,0,:5] = {nelssa_cpu[0, 0, 0, :5]}")
    print(f"  Debug: GPU ref output[0,0,0,:5] = {ref_cpu[0, 0, 0, :5]}")
    nelssa_mean = nelssa_cpu.mean().item()
    nelssa_std = nelssa_cpu.std().item()
    ref_mean = ref_cpu.mean().item()
    ref_std = ref_cpu.std().item()
    print(f"  Debug: NELSSA stats: mean={nelssa_mean:.6f}, std={nelssa_std:.6f}")
    print(f"  Debug: GPU ref stats: mean={ref_mean:.6f}, std={ref_std:.6f}")

    if not torch.allclose(nelssa_cpu, ref_cpu, atol=atol, rtol=rtol):
        diff = (nelssa_cpu - ref_cpu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        diff_mask = diff > (atol + rtol * ref_cpu.abs())

        print(colored("[FAIL] Attention output mismatch!", "red"))
        print(colored(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}", "red"))

        if diff_mask.any():
            num_mismatched = diff_mask.sum().item()
            total_elements = diff_mask.numel()
            print(
                colored(
                    f"  Mismatched elements: {num_mismatched}/{total_elements} ({100 * num_mismatched / total_elements:.2f}%)",
                    "red",
                )
            )

            # Show some mismatched values
            wrong_indices = torch.nonzero(diff_mask, as_tuple=False)
            num_show = min(5, wrong_indices.shape[0])
            print(colored("  Sample mismatched positions:", "red"))
            for i in range(num_show):
                idx = tuple(wrong_indices[i].tolist())
                nelssa_val = nelssa_cpu[idx]
                ref_val = ref_cpu[idx]
                print(
                    f"    {idx}: NELSSA={nelssa_val:.6f}, Ref={ref_val:.6f}, Diff={abs(nelssa_val - ref_val):.6f}"
                )
        all_passed = False
    else:
        print(
            colored(
                f"[PASS] Attention output match within tolerance (atol={atol}, rtol={rtol})",
                "green",
            )
        )

    return all_passed


def main(args):
    """메인 테스트 함수"""
    print(colored("=" * 80, "cyan"))
    print(colored("NELSSA Node Operation Verification Test", "cyan"))
    print(colored("=" * 80, "cyan"))

    # 설정
    set_seed(args.seed)
    device = args.device
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    batch_size = args.batch_size
    seq_len = args.seq_len
    kv_heads = args.kv_heads
    num_heads = args.num_heads
    head_dim = args.head_dim
    n_centroids = args.n_centroids
    n_segments = args.n_segments
    nprobe = max(int(n_centroids * args.retrieval_budget), 1)

    print("\n[Test Configuration]")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  KV heads: {kv_heads}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  N centroids: {n_centroids}")
    print(f"  N segments: {n_segments}")
    print(f"  N probe: {nprobe}")
    print(f"  Retrieval budget: {args.retrieval_budget}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Device: {device}")

    # Step 1: Generate random KV data
    print(colored("\n[Step 1] Generating random KV data...", "yellow"))
    key_states, value_states = generate_random_kv_data(
        batch_size=batch_size,
        seq_len=seq_len,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    print(f"  Key states shape: {key_states.shape}")
    print(f"  Value states shape: {value_states.shape}")

    # Step 2: Cluster KV data
    print(colored("\n[Step 2] Clustering KV data...", "yellow"))
    centroids, value_sum, clusters, cluster_size = cluster_kv_data(
        key_states=key_states,
        value_states=value_states,
        n_centroids=n_centroids,
        n_segments=n_segments,
    )
    print(f"  Centroids shape: {centroids.shape}")
    print(f"  Clusters shape: {clusters.shape}")
    print(f"  Cluster size shape: {cluster_size.shape}")
    print(f"  Max cluster size: {cluster_size.max().item()}")
    print(f"  Avg cluster size: {cluster_size.float().mean().item():.2f}")

    # Step 3: Prepare KV for Nelssa
    print(colored("\n[Step 3] Preparing KV data for Nelssa...", "yellow"))
    k_cache, v_cache, size_cache = prepare_kv_for_nelssa(
        key_states=key_states,
        value_states=value_states,
        cluster_size=cluster_size,
    )
    print(f"  K cache shape: {k_cache.shape}")
    print(f"  V cache shape: {v_cache.shape}")
    print(f"  Size cache shape: {size_cache.shape}")

    # Step 4: Initialize NelssaClient
    print(colored("\n[Step 4] Initializing NelssaClient...", "yellow"))
    model_config = {
        "num_layers": args.num_layers,
        "num_heads": num_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "n_clusters": n_centroids,
        "cache_unit_size": args.pages_per_cluster * 8,  # page_size = 8
        "n_probe": nprobe,
        "kv_dtype": dtype,
    }
    print(f"  Model config: {model_config}")

    nelssa_client = NelssaClient(
        host=args.pnm_host,
        port=args.pnm_port,
        model_config=model_config,
    )

    # Step 5: Send KV cache to Nelssa
    print(colored("\n[Step 5] Sending KV cache to Nelssa server...", "yellow"))
    try:
        for layer_idx in range(args.num_layers):
            nelssa_client.send_kv_cache(
                layer_idx=layer_idx,
                k_cache_tensor=k_cache.to("cpu").contiguous(),
                v_cache_tensor=v_cache.to("cpu").contiguous(),
                size_cache_tensor=size_cache.to("cpu").contiguous(),
                current_n_clusters=n_centroids,
            )
            print(f"  Layer {layer_idx}: KV cache sent successfully")
        print(colored("  [SUCCESS] All KV caches sent to Nelssa server", "green"))
    except Exception as e:
        print(colored(f"  [ERROR] Failed to send KV cache: {e}", "red"))
        return

    # Step 6: Generate random query
    print(colored("\n[Step 6] Generating random query states...", "yellow"))
    query_states = torch.randn(
        (batch_size, 1, num_heads, head_dim), dtype=dtype, device=device
    )
    print(f"  Query states shape: {query_states.shape}")

    # Step 7: Compute reference sparse attention on GPU
    print(
        colored("\n[Step 7] Computing reference sparse attention on GPU...", "yellow")
    )
    torch.cuda.synchronize()
    ref_start = time.time()

    ref_output, ref_cluster_ids = sparse_attention_gpu(
        query=query_states,
        keys=key_states,
        values=value_states,
        centroids=centroids.to(device),
        clusters=clusters.to(device),
        cluster_size=cluster_size.to(device),
        nprobe=nprobe,
    )

    torch.cuda.synchronize()
    ref_time = time.time() - ref_start

    # Debug: Print GPU reference output sample
    print(f"  Debug: GPU ref_output[0,0,0,:4] = {ref_output[0, 0, 0, :4]}")
    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  Reference cluster IDs shape: {ref_cluster_ids.shape}")
    print(f"  Reference computation time: {ref_time * 1000:.2f} ms")

    # Step 8: Compute Nelssa attention
    print(colored("\n[Step 8] Computing Nelssa attention...", "yellow"))
    try:
        torch.cuda.synchronize()
        nelssa_start = time.time()

        # Debug: Print query and cluster_ids sample
        print(f"  Debug: query_states sample (batch=0, head=0, first 4 dims): {query_states[0, 0, 0, :4]}")
        print(f"  Debug: cluster_ids shape: {ref_cluster_ids.shape}")
        print(f"  Debug: cluster_ids[0] (first batch_group): {ref_cluster_ids[0]}")

        # Prepare cluster IDs for Nelssa
        # ref_cluster_ids: [batch_size * kv_heads, nprobe]
        # Need to reshape to: [batch_size, kv_heads, nprobe]
        cluster_ids_for_nelssa = ref_cluster_ids.view(batch_size, kv_heads, nprobe).to(torch.int32)
        print(f"  Debug: cluster_ids_for_nelssa shape: {cluster_ids_for_nelssa.shape}")

        # Debug: Check total tokens selected
        bdx = 0
        total_tokens = 0
        for c_idx in range(nprobe):
            cluster_id = ref_cluster_ids[bdx, c_idx].item()
            size = cluster_size[bdx, cluster_id].item()
            total_tokens += size
        print(f"  Debug: Total tokens to attend (GPU ref): {total_tokens}")

        # Execute decode via NelssaClient
        nelssa_result = nelssa_client.execute_decode_batched(
            layer_idx=0,
            bsz=batch_size,
            queries=query_states.cpu(),
            cluster_ids=cluster_ids_for_nelssa.cpu(),
        )

        torch.cuda.synchronize()
        nelssa_time = time.time() - nelssa_start

        # Parse result (format: [batch_group, 1, group_size, head_dim + 2])
        # Last 2 dimensions contain LSE values
        batch_group = batch_size * kv_heads
        group_size = num_heads // kv_heads

        nelssa_output = nelssa_result[:, :, :, :-2].to(device)
        nelssa_lse = nelssa_result[:, :, :, -2:].to(device)

        # Reshape to match reference format
        nelssa_output = nelssa_output.view(
            batch_size, kv_heads, group_size, head_dim
        ).to(dtype=torch.float16)
        nelssa_output = nelssa_output.transpose(1, 2).reshape(
            batch_size, 1, num_heads, head_dim
        )

        print(f"  Nelssa output shape: {nelssa_output.shape}")
        print(f"  Nelssa computation time: {nelssa_time * 1000:.2f} ms")
        print(colored("  [SUCCESS] Nelssa attention computed", "green"))

    except Exception as e:
        print(colored(f"  [ERROR] Failed to compute Nelssa attention: {e}", "red"))
        import traceback

        traceback.print_exc()
        return

    # Step 9: Compare results
    print(colored("\n[Step 9] Comparing results...", "yellow"))
    all_passed = compare_results(
        nelssa_output=nelssa_output,
        reference_output=ref_output,
        atol=args.atol,
        rtol=args.rtol,
    )

    if all_passed:
        print(colored("\n" + "=" * 80, "green"))
        print(colored("ALL TESTS PASSED!", "green"))
        print(colored("=" * 80, "green"))
    else:
        print(colored("\n" + "=" * 80, "red"))
        print(colored("SOME TESTS FAILED!", "red"))
        print(colored("=" * 80, "red"))

    # Summary
    print(colored("\n[Summary]", "cyan"))
    print(f"  Random KV data generated: {key_states.shape}")
    print(f"  Clustering completed: {n_centroids} centroids")
    print(f"  Reference attention: {ref_output.shape} ({ref_time * 1000:.2f}ms)")
    print(f"  Nelssa attention: {nelssa_output.shape} ({nelssa_time * 1000:.2f}ms)")
    print(f"  Speedup: {ref_time / nelssa_time:.2f}x")

    print(colored("\nTest completed!", "cyan"))


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Verify NELSSA node operations with random clustered KV data"
    )

    # Data generation parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--kv_heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument(
        "--num_heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")

    # Clustering parameters
    parser.add_argument(
        "--n_centroids", type=int, default=256, help="Number of clusters"
    )
    parser.add_argument(
        "--n_segments", type=int, default=4, help="Number of segments for clustering"
    )
    parser.add_argument(
        "--retrieval_budget", type=float, default=0.1, help="Retrieval budget ratio"
    )
    parser.add_argument(
        "--pages_per_cluster", type=int, default=4, help="Pages per cluster"
    )

    # NelssaClient parameters
    parser.add_argument(
        "--pnm_host", type=str, default="10.0.0.2", help="PNM server host"
    )
    parser.add_argument("--pnm_port", type=int, default=50058, help="PNM server port")
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of layers to test"
    )

    # General parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Data type"
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance for comparison"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-2, help="Relative tolerance for comparison"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
verify_test.py

Random으로 생성한 KV를 클러스터링한 데이터셋으로 검증하는 테스트 스크립트.
- NelssaClient를 생성하고 KV를 적재하거나 연산
- 같은 데이터셋으로 sparse attention한 결과 값과 비교
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

from cache_hub.kmeans import segment_k_means
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

    Args:
        batch_size: 배치 크기
        seq_len: 시퀀스 길이
        kv_heads: KV head 수
        head_dim: head dimension
        dtype: 데이터 타입
        device: 디바이스

    Returns:
        key_states: [batch_size, seq_len, kv_heads, head_dim]
        value_states: [batch_size, seq_len, kv_heads, head_dim]
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

    Args:
        key_states: [batch_size, seq_len, kv_heads, head_dim]
        value_states: [batch_size, seq_len, kv_heads, head_dim]
        n_centroids: 클러스터 수
        n_segments: 세그먼트 수

    Returns:
        centroids: [batch_size*kv_heads, n_centroids, head_dim]
        value_sum: [batch_size*kv_heads, n_centroids, head_dim]
        clusters: [batch_size*kv_heads, n_centroids, max_cluster_size]
        cluster_size: [batch_size*kv_heads, n_centroids]
    """
    batch_size, seq_len, kv_heads, head_dim = key_states.shape

    # Reshape for clustering: [batch_size*kv_heads, seq_len, head_dim]
    keys_reshaped = key_states.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )
    values_reshaped = value_states.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )

    # Centering
    mean_key = torch.mean(keys_reshaped, dim=1, keepdim=True)
    keys_centered = keys_reshaped - mean_key

    # Segment K-means clustering
    centroids, value_sum, clusters, cluster_size = segment_k_means(
        key=keys_centered,
        value=values_reshaped,
        num_centroids=n_centroids,
        num_segments=n_segments,
    )

    return centroids, value_sum, clusters, cluster_size


def prepare_kv_for_nelssa(key_states, value_states, cluster_size, n_centroids):
    """
    NelssaClient.send_kv_cache에 맞게 KV 데이터 준비

    Args:
        key_states: [batch_size, seq_len, kv_heads, head_dim]
        value_states: [batch_size, seq_len, kv_heads, head_dim]
        cluster_size: [batch_size*kv_heads, n_centroids]
        n_centroids: 클러스터 수

    Returns:
        k_cache: [batch_size, kv_heads, seq_len, head_dim] (CPU)
        v_cache: [batch_size, kv_heads, seq_len, head_dim] (CPU)
        size_cache: [batch_size*kv_heads, n_centroids] (CPU)
    """
    batch_size, seq_len, kv_heads, head_dim = key_states.shape

    # Reshape to [batch_size, kv_heads, seq_len, head_dim]
    k_cache = key_states.transpose(1, 2).contiguous().cpu()
    v_cache = value_states.transpose(1, 2).contiguous().cpu()
    size_cache = cluster_size.contiguous().cpu()

    return k_cache, v_cache, size_cache


def compute_sparse_attention_reference(
    query_states,
    key_states,
    value_states,
    centroids,
    cluster_size,
    clusters,
    nprobe,
    retrieval_budget=0.1,
):
    """
    Reference sparse attention computation (GPU-based)
    Uses only the selected clusters' KV for attention (sparse attention)

    Args:
        query_states: [batch_size, 1, num_heads, head_dim]
        key_states: [batch_size, seq_len, kv_heads, head_dim]
        value_states: [batch_size, seq_len, kv_heads, head_dim]
        centroids: [batch_size*kv_heads, n_centroids, head_dim]
        cluster_size: [batch_size*kv_heads, n_centroids]
        clusters: [batch_size*kv_heads, n_centroids, max_cluster_size]
        nprobe: number of clusters to retrieve
        retrieval_budget: retrieval budget ratio

    Returns:
        attn_output: [batch_size, 1, num_heads, head_dim]
        lse: log-sum-exp values
        cluster_ids: selected cluster IDs
    """
    batch_size, _, num_heads, head_dim = query_states.shape
    _, seq_len, kv_heads, _ = key_states.shape
    group_size = num_heads // kv_heads

    device = query_states.device
    dtype = query_states.dtype

    # Reshape query for grouping: [batch_size*kv_heads, 1, group_size, head_dim]
    query_grouped = query_states.view(batch_size, 1, kv_heads, group_size, head_dim)
    query_grouped = query_grouped.transpose(1, 2).reshape(
        batch_size * kv_heads, 1, group_size, head_dim
    )

    # Compute similarity with centroids: [batch_size*kv_heads, group_size, n_centroids]
    n_centroids = centroids.shape[1]

    # Compute dot product similarity
    similarity = torch.matmul(
        query_grouped,  # [B*H, 1, group_size, head_dim]
        centroids.unsqueeze(1).transpose(-2, -1),  # [B*H, 1, head_dim, n_centroids]
    )  # [B*H, 1, group_size, n_centroids]
    similarity = similarity.squeeze(1)  # [B*H, group_size, n_centroids]

    # Select top-k clusters
    topk_values, topk_indices = torch.topk(
        similarity, nprobe, dim=-1
    )  # [B*H, group_size, nprobe]

    # Reshape KV for attention
    keys_reshaped = key_states.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )
    values_reshaped = value_states.transpose(1, 2).reshape(
        batch_size * kv_heads, seq_len, head_dim
    )

    # Gather KV from selected clusters (sparse attention)
    # For each batch*head and each selected cluster, gather the tokens in that cluster
    batch_groups = batch_size * kv_heads
    max_cluster_size = clusters.shape[2]

    # Collect all tokens from selected clusters
    # This is a simplified version - gather tokens from selected clusters
    selected_keys = []
    selected_values = []
    selected_lengths = []

    for b in range(batch_groups):
        # Get unique cluster IDs for this batch*head (across all groups)
        unique_clusters = torch.unique(topk_indices[b])  # [nprobe]

        # Gather tokens from these clusters
        tokens_in_clusters = []
        for cid in unique_clusters:
            cid_int = cid.item()
            size = cluster_size[b, cid_int].item()
            if size > 0:
                # Get token indices in this cluster
                token_indices = clusters[b, cid_int, :size]  # [size]
                tokens_in_clusters.append(token_indices)

        if len(tokens_in_clusters) > 0:
            all_token_indices = torch.cat(tokens_in_clusters)  # [total_tokens]
            # Remove duplicates and sort
            all_token_indices = torch.unique(all_token_indices, sorted=True)

            # Gather keys and values
            sel_keys = keys_reshaped[b, all_token_indices]  # [n_tokens, head_dim]
            sel_values = values_reshaped[b, all_token_indices]  # [n_tokens, head_dim]

            selected_keys.append(sel_keys)
            selected_values.append(sel_values)
            selected_lengths.append(len(all_token_indices))
        else:
            # Fallback: use all tokens
            selected_keys.append(keys_reshaped[b])
            selected_values.append(values_reshaped[b])
            selected_lengths.append(seq_len)

    # Compute attention for each batch*head separately (due to different lengths)
    attn_outputs = []
    lse_list = []

    scale = 1.0 / math.sqrt(head_dim)

    for b in range(batch_groups):
        q = query_grouped[b, 0]  # [group_size, head_dim]
        k = selected_keys[b]  # [n_tokens, head_dim]
        v = selected_values[b]  # [n_tokens, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.t()) * scale  # [group_size, n_tokens]

        # Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_out = torch.matmul(attn_probs, v)  # [group_size, head_dim]
        attn_outputs.append(attn_out)

        # Compute LSE
        lse_val = torch.logsumexp(attn_scores, dim=-1, keepdim=True)  # [group_size, 1]
        lse_list.append(lse_val)

    # Stack results
    attn_output = torch.stack(attn_outputs)  # [B*H, group_size, head_dim]
    lse = torch.stack(lse_list)  # [B*H, group_size, 1]

    # Reshape back: [batch_size, 1, num_heads, head_dim]
    attn_output = attn_output.view(batch_size, kv_heads, group_size, head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(
        batch_size, 1, num_heads, head_dim
    )

    # Reshape LSE: [batch_size, num_heads, 1]
    lse = lse.view(batch_size, kv_heads, group_size, 1)
    lse = lse.transpose(1, 2).reshape(batch_size, num_heads, 1)

    return attn_output, lse, topk_indices


def compare_results(
    nelssa_output,
    reference_output,
    nelssa_lse=None,
    reference_lse=None,
    atol=1e-3,
    rtol=1e-3,
):
    """
    Nelssa 결과와 reference 결과 비교

    Args:
        nelssa_output: Nelssa attention output
        reference_output: Reference attention output
        nelssa_lse: Nelssa LSE values
        reference_lse: Reference LSE values
        atol: absolute tolerance
        rtol: relative tolerance

    Returns:
        bool: True if all tests pass
    """
    all_passed = True

    # Compare attention outputs
    if nelssa_output is not None and reference_output is not None:
        nelssa_cpu = nelssa_output.cpu()
        ref_cpu = reference_output.cpu()

        if not torch.allclose(nelssa_cpu, ref_cpu, atol=atol, rtol=rtol):
            diff = (nelssa_cpu - ref_cpu).abs()
            max_diff = diff.max().item()
            diff_mask = diff > (atol + rtol * ref_cpu.abs())

            print(
                colored(
                    f"[FAIL] Attention output mismatch! Max diff: {max_diff:.6f}", "red"
                )
            )

            if diff_mask.any():
                wrong_indices = torch.nonzero(diff_mask, as_tuple=False)
                num_show = min(10, wrong_indices.shape[0])
                print(
                    colored(
                        f"Number of mismatched elements: {wrong_indices.shape[0]}",
                        "red",
                    )
                )
                print(colored(f"First {num_show} mismatched positions:", "red"))
                for i in range(num_show):
                    idx = wrong_indices[i]
                    nelssa_val = nelssa_cpu[tuple(idx)].item()
                    ref_val = ref_cpu[tuple(idx)].item()
                    print(
                        f"  Position [{idx}]: NELSSA={nelssa_val:.6f}, Ref={ref_val:.6f}, Diff={abs(nelssa_val - ref_val):.6f}"
                    )
            all_passed = False
        else:
            print(
                colored(
                    f"[PASS] Attention output match within tolerance (atol={atol}, rtol={rtol})",
                    "green",
                )
            )

    # Compare LSE values
    if nelssa_lse is not None and reference_lse is not None:
        nelssa_lse_cpu = nelssa_lse.cpu()
        ref_lse_cpu = reference_lse.cpu()

        # Handle dimension mismatch
        if nelssa_lse_cpu.dim() != ref_lse_cpu.dim():
            if nelssa_lse_cpu.dim() == 4 and ref_lse_cpu.dim() == 3:
                nelssa_lse_cpu = nelssa_lse_cpu.squeeze(1)
            elif nelssa_lse_cpu.dim() == 3 and ref_lse_cpu.dim() == 4:
                ref_lse_cpu = ref_lse_cpu.squeeze(1)

        if not torch.allclose(nelssa_lse_cpu, ref_lse_cpu, atol=atol, rtol=rtol):
            diff = (nelssa_lse_cpu - ref_lse_cpu).abs()
            max_diff = diff.max().item()
            print(colored(f"[FAIL] LSE mismatch! Max diff: {max_diff:.6f}", "red"))
            all_passed = False
        else:
            print(colored("[PASS] LSE match within tolerance", "green"))

    return all_passed


def main(args):
    """메인 테스트 함수"""
    print(colored("=" * 80, "cyan"))
    print(colored("NELSSA Verification Test with Random Clustered KV Data", "cyan"))
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
    print(f"  Value sum shape: {value_sum.shape}")
    print(f"  Clusters shape: {clusters.shape}")
    print(f"  Cluster size shape: {cluster_size.shape}")
    print(f"  Max cluster size: {cluster_size.max().item()}")
    print(f"  Min cluster size: {cluster_size.min().item()}")
    print(f"  Avg cluster size: {cluster_size.float().mean().item():.2f}")

    # Step 3: Prepare KV for Nelssa
    print(colored("\n[Step 3] Preparing KV data for Nelssa...", "yellow"))
    k_cache, v_cache, size_cache = prepare_kv_for_nelssa(
        key_states=key_states,
        value_states=value_states,
        cluster_size=cluster_size,
        n_centroids=n_centroids,
    )
    print(f"  K cache shape: {k_cache.shape}")
    print(f"  V cache shape: {v_cache.shape}")
    print(f"  Size cache shape: {size_cache.shape}")

    # Step 4: Initialize NelssaClient (if not skipped)
    nelssa_client = None
    if not args.skip_nelssa:
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

        try:
            nelssa_client = NelssaClient(
                host=args.pnm_host,
                port=args.pnm_port,
                model_config=model_config,
            )
            print(colored("  [SUCCESS] NelssaClient initialized", "green"))
        except Exception as e:
            print(colored(f"  [ERROR] Failed to initialize NelssaClient: {e}", "red"))
            print(colored("  Continuing with reference computation only...", "yellow"))
            nelssa_client = None
    else:
        print(
            colored(
                "\n[Step 4] Skipping NelssaClient initialization (--skip-nelssa)",
                "yellow",
            )
        )

    # Step 5: Send KV cache to Nelssa (if client is available)
    if nelssa_client is not None:
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
            nelssa_client = None

    # Step 6: Generate random query for attention computation
    print(colored("\n[Step 6] Generating random query states...", "yellow"))
    query_states = torch.randn(
        (batch_size, 1, num_heads, head_dim), dtype=dtype, device=device
    )
    print(f"  Query states shape: {query_states.shape}")

    # Step 7: Compute reference sparse attention (GPU-based)
    print(colored("\n[Step 7] Computing reference sparse attention...", "yellow"))
    torch.cuda.synchronize()
    ref_start = time.time()

    ref_output, ref_lse, ref_cluster_ids = compute_sparse_attention_reference(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        centroids=centroids.to(device),
        cluster_size=cluster_size.to(device),
        nprobe=nprobe,
        retrieval_budget=args.retrieval_budget,
    )

    torch.cuda.synchronize()
    ref_time = time.time() - ref_start
    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  Reference LSE shape: {ref_lse.shape}")
    print(f"  Reference cluster IDs shape: {ref_cluster_ids.shape}")
    print(f"  Reference computation time: {ref_time * 1000:.2f} ms")

    # Step 8: Compute Nelssa attention (if client is available)
    nelssa_output = None
    nelssa_lse = None
    nelssa_cluster_ids = None

    if nelssa_client is not None:
        print(colored("\n[Step 8] Computing Nelssa attention...", "yellow"))
        try:
            torch.cuda.synchronize()
            nelssa_start = time.time()

            # Prepare cluster IDs for Nelssa
            # Use the same cluster IDs as reference for fair comparison
            cluster_ids_for_nelssa = ref_cluster_ids.view(batch_size, kv_heads, -1).to(
                torch.int32
            )

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
            )
            nelssa_output = (
                nelssa_output.transpose(1, 2)
                .reshape(batch_size, 1, num_heads, head_dim)
                .to(dtype=torch.float16)
            )

            print(f"  Nelssa output shape: {nelssa_output.shape}")
            print(f"  Nelssa LSE shape: {nelssa_lse.shape}")
            print(f"  Nelssa computation time: {nelssa_time * 1000:.2f} ms")
            print(colored("  [SUCCESS] Nelssa attention computed", "green"))

        except Exception as e:
            print(colored(f"  [ERROR] Failed to compute Nelssa attention: {e}", "red"))
            import traceback

            traceback.print_exc()
            nelssa_output = None
    else:
        print(
            colored(
                "\n[Step 8] Skipping Nelssa attention (client not available)", "yellow"
            )
        )

    # Step 9: Compare results
    print(colored("\n[Step 9] Comparing results...", "yellow"))
    if nelssa_output is not None:
        all_passed = compare_results(
            nelssa_output=nelssa_output,
            reference_output=ref_output,
            nelssa_lse=nelssa_lse,
            reference_lse=ref_lse,
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
    else:
        print(colored("\n[SKIP] Nelssa output not available for comparison", "yellow"))
        print(colored("\n" + "=" * 80, "blue"))
        print(colored("Reference computation completed successfully", "blue"))
        print(colored("=" * 80, "blue"))

    # Summary
    print(colored("\n[Summary]", "cyan"))
    print(f"  Random KV data generated: {key_states.shape}")
    print(f"  Clustering completed: {n_centroids} centroids")
    print(f"  Reference attention: {ref_output.shape}")
    if nelssa_client is not None:
        print("  Nelssa client: Initialized and connected")
        if nelssa_output is not None:
            print(f"  Nelssa attention: {nelssa_output.shape}")
    else:
        print("  Nelssa client: Not available")

    print(colored("\nTest completed!", "cyan"))


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Verify NELSSA with random clustered KV data"
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
    parser.add_argument(
        "--skip_nelssa", action="store_true", help="Skip NelssaClient initialization"
    )

    # General parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Data type"
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--atol", type=float, default=1e-3, help="Absolute tolerance for comparison"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3, help="Relative tolerance for comparison"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

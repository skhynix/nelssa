"""Complex workload test with configurable QPS and long request ratio."""

import os
import argparse
import asyncio
import torch
import time
import json

os.environ['VLLM_USE_RAY_SPMD_WORKER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

from vllm import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.engine.arg_utils import AsyncEngineArgs

selection_ratio=0.018
output_length=1000
long_threshold=100000 

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Complex workload test with configurable QPS and long request ratio")
    parser.add_argument('--flag', type=str, default='False', help='Enable sparse KV offloading (True/False)')
    parser.add_argument('--enable_cpu_attention', type=str, default='True', help='Enable CPU attention computation (True/False)')
    parser.add_argument('--enable_offloading', type=str, default='True', help='Enable KV offloading to CPU (True/False)')
    parser.add_argument('--qps', type=float, default=4.0, help='Queries per second (default: 1.0)')
    parser.add_argument('--long_ratio', type=float, default=0.005, help='Ratio of long requests (0.0-1.0, default: 0.1)')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds (default: 60)')
    parser.add_argument('--long_tokens', type=int, default=125000, help='Token count for long requests (default: 64000)')
    parser.add_argument('--short_tokens_max', type=int, default=64000, help='Min token count for short requests (default: 4000)')
    parser.add_argument('--model', type=str, default='/mnt/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659', help='Model path')
    return parser.parse_args()

def load_trace_lengths(trace_file, max_requests=100):
    """Load request lengths from conversation trace."""
    print(f"Loading trace from {trace_file}...")
    lengths = []
    with open(trace_file, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                input_len = int(obj.get("input_length", 0))
                if input_len >= 10:  # Filter out very short entries
                    lengths.append(input_len)
            except:
                continue
            if len(lengths) >= max_requests:
                break
    print(f"Loaded {len(lengths)} requests from trace")
    return lengths


def build_prompt(tokenizer, target_tokens):
    """Build a prompt with approximately target_tokens tokens."""
    base_sentence = "Hello world. What is today's lunch menu? "
    tokens = []
    while len(tokens) < target_tokens:
        tokens.extend(tokenizer.encode(base_sentence, add_special_tokens=False))
    return tokens[:target_tokens]


async def test_complex_workload(args):
    """Test sparse KV cache with complex workload from trace."""
    # Parse flag arguments (convert string to bool)
    enable_flag = args.flag.lower() == 'true'
    enable_cpu_attention_flag = args.enable_cpu_attention.lower() == 'true'
    enable_offloading_flag = args.enable_offloading.lower() == 'true'
    print("=" * 70)
    print(f"Complex Workload Test: {args.duration}s, {args.qps} req/s, long_ratio={args.long_ratio}")
    print(f"Flags: sparse_kv={enable_flag}, cpu_attention={enable_cpu_attention_flag}, offloading={enable_offloading_flag}")
    print("=" * 70)

    # Calculate short and long request counts
    # Short requests are the base, long requests are added on top (not replacing short)
    num_short_requests = int(args.qps * args.duration)
    num_long_requests = 0
    if args.long_ratio > 0:
        num_long_requests = max(1, int(num_short_requests * args.long_ratio)+1)
    total_requests = num_short_requests + num_long_requests

    # Load trace lengths (only for short requests)
    trace_file = os.path.join(os.path.dirname(__file__), 'conversation_trace.jsonl')
    trace_lengths = load_trace_lengths(trace_file, max_requests=num_short_requests)

    if len(trace_lengths) < num_short_requests:
        print(f"Warning: Only {len(trace_lengths)} trace entries, padding with synthetic data")
        while len(trace_lengths) < num_short_requests:
            trace_lengths.append(100 + len(trace_lengths) * 50)

    # Create short and long request lists
    short_lengths = [min(l, args.short_tokens_max) for l in trace_lengths[:num_short_requests]]
    long_lengths = [args.long_tokens] * num_long_requests

    # Calculate even distribution of long requests across total_requests
    # Long requests are interleaved with short requests for uniform scheduling
    request_lengths = []
    if num_long_requests > 0:
        # Evenly distribute long requests using floating point step
        step = total_requests / num_long_requests
        long_indices = set(int(i * step) for i in range(num_long_requests))

        short_idx = 0
        long_idx = 0
        for i in range(total_requests):
            if i in long_indices:
                request_lengths.append(long_lengths[long_idx])
                long_idx += 1
            else:
                request_lengths.append(short_lengths[short_idx])
                short_idx += 1
    else:
        # No long requests: use all short requests
        request_lengths = short_lengths

    # Build all prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    print(f"\nBuilding {len(request_lengths)} prompts from trace lengths...")
    prompts = []
    for i, length in enumerate(request_lengths):
        prompts.append(build_prompt(tokenizer, length))

    # Engine initialization with sparse KV
    engine_args = AsyncEngineArgs(
        model=args.model,
        enforce_eager=True,
        max_model_len=128000,
        gpu_memory_utilization=0.8,
        distributed_executor_backend='mp',
        enable_prefix_caching=False,  # Disable prefix caching to isolate the issue
        attention_backend='TRITON_ATTN',  # Use Triton Attention backend to avoid FlashAttention CUDA driver issue
    )
    engine_args.nelssa_config = {
        'enabled': enable_flag,
        'first_n': 16,
        'last_m': 64,
        'prompt_length_threshold': long_threshold,
        'enable_cpu_attention': enable_cpu_attention_flag,
        'cpu_attention_kv_fraction': selection_ratio,
        'enable_offloading': enable_offloading_flag,
    }

    vllm_config = engine_args.create_engine_config()

    # Get cache config for memory estimation
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    sparse_kv_config = cache_config.nelssa_config

    print("\n=== Initializing Async Engine with Sparse KV ===")
    executor_class = Executor.get_class(vllm_config)
    llm = AsyncLLM(vllm_config=vllm_config, executor_class=executor_class, log_stats=True)

    # Track results
    results = []

    # Pre-compute buffer sizes for memory estimation
    def compute_memory_estimates():
        # Get total GPU memory
        gpu_total = torch.cuda.mem_get_info()[1] / (1024**3)  # GB

        # GPU memory utilization for KV cache
        gpu_kv_fraction = cache_config.gpu_memory_utilization
        gpu_kv_total = gpu_total * gpu_kv_fraction

        # Estimate KV cache buffer per token
        # For Llama-3.1-8B: hidden_size=4096, num_layers=32, num_kv_heads=8, head_dim=128
        # KV cache per layer per token = 2 * num_kv_heads * head_dim * dtype_size
        hidden_size = model_config.get_hidden_size()
        num_layers = model_config.get_total_num_hidden_layers()
        num_kv_heads = model_config.get_total_num_kv_heads()
        head_dim = model_config.get_head_size()
        dtype_size = 2  # float16/bfloat16

        kv_per_token_bytes = 2 * num_kv_heads * head_dim * dtype_size  # K and V
        kv_per_layer_per_token_gb = kv_per_token_bytes / (1024**3)

        # Total tokens that can fit in GPU KV cache
        max_gpu_tokens = gpu_kv_total / (num_layers * kv_per_layer_per_token_gb)

        # For sparse KV: first_n + last_m on GPU, rest on CPU
        first_n = sparse_kv_config.first_n
        last_m = sparse_kv_config.last_m
        gpu_tokens_per_long_req = first_n + last_m

        # CPU buffer size estimation
        cpu_attention_kv_fraction = sparse_kv_config.cpu_attention_kv_fraction
        # CPU buffer holds the middle tokens that are offloaded

        return {
            'gpu_total_gb': gpu_total,
            'gpu_kv_total_gb': gpu_kv_total,
            'gpu_tokens_per_long_req': gpu_tokens_per_long_req,
            'num_layers': num_layers,
            'kv_per_layer_per_token_gb': kv_per_layer_per_token_gb,
            'cpu_attention_kv_fraction': cpu_attention_kv_fraction,
        }

    memory_estimates = compute_memory_estimates()
    print(f"Memory estimates: GPU KV buffer = {memory_estimates['gpu_kv_total_gb']:.2f}GB")
    print(f"  Per long request: {memory_estimates['gpu_tokens_per_long_req']} tokens on GPU")

    async def run_request(req_id, prompt, scheduled_time, prompt_len):
        # Calculate output tokens based on prompt length (ratio: 1/128, min: 100)
        # output_tokens = prompt_len // 128
        output_tokens = output_length
        req_sampling_params = SamplingParams(temperature=0.0, max_tokens=output_tokens)

        actual_start = time.time()
        generated_tokens = 0
        async for o in llm.generate(prompt=prompt, sampling_params=req_sampling_params, request_id=req_id):
            generated_tokens += 1
        elapsed = time.time() - actual_start
        req_type = "LONG" if prompt_len >= 16000 else "short"
        tpot = elapsed / (prompt_len + generated_tokens) if (prompt_len + generated_tokens) > 0 else 0
        # print(f"[T+{scheduled_time:.1f}s] {req_id} ({req_type}, {prompt_len}+{generated_tokens} tokens) completed in {elapsed:.2f}s (TPOT: {tpot*1000:.2f}ms)")
        results.append({
            'req_id': req_id,
            'scheduled_time': scheduled_time,
            'prompt_len': prompt_len,
            'generated_tokens': generated_tokens,
            'req_type': req_type,
            'latency': elapsed,
            'tpot': tpot
        })

    start_time = time.time()
    tasks = []

    # Calculate sleep interval based on QPS
    sleep_interval = 1.0 / args.qps if args.qps > 0 else 1.0

    for i in range(total_requests):
        req_id = f'req-{i:04d}'
        prompt_len = request_lengths[i]
        prompt = prompts[i]

        task = asyncio.create_task(run_request(req_id, prompt, i * sleep_interval, prompt_len))
        tasks.append(task)

        # Wait before next request
        if i < total_requests - 1:
            await asyncio.sleep(sleep_interval)

    print(f"\n=== All requests submitted, waiting for completion ===")
    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total test duration: {elapsed:.0f} seconds")
    print(f"Total requests: {len(results)}")

    # Group by type
    long_reqs = [r for r in results if r['req_type'] == 'LONG']
    short_reqs = [r for r in results if r['req_type'] == 'short']

    print(f"\nRequest Distribution:")
    print(f"  - Long requests (>={long_threshold} tokens): {len(long_reqs)}")
    print(f"  - Short requests (<{long_threshold} tokens): {len(short_reqs)}")

    print(f"\nToken Statistics:")
    all_tokens = [r['prompt_len'] for r in results]
    print(f"  - Min tokens: {min(all_tokens)}")
    print(f"  - Max tokens: {max(all_tokens)}")
    print(f"  - Avg tokens: {sum(all_tokens)/len(all_tokens):.0f}")
    print(f"  - Total tokens: {sum(all_tokens)}")

    print(f"\nLatency Statistics:")
    if long_reqs:
        long_latencies = [r['latency'] for r in long_reqs]
        print(f"  - Long requests avg latency: {sum(long_latencies)/len(long_latencies):.2f}s")
        print(f"    (min: {min(long_latencies):.2f}s, max: {max(long_latencies):.2f}s)")
    if short_reqs:
        short_latencies = [r['latency'] for r in short_reqs]
        print(f"  - Short requests avg latency: {sum(short_latencies)/len(short_latencies):.2f}s")
        print(f"    (min: {min(short_latencies):.2f}s, max: {max(short_latencies):.2f}s)")

    print(f"\nTPOT (Time Per Output Token) Statistics:")
    # TPOT = total_latency / total_tokens (prompt + generated)
    all_tpot = [r['tpot'] for r in results]
    mean_tpot = sum(all_tpot) / len(all_tpot)
    sorted_tpot = sorted(all_tpot)
    n = len(sorted_tpot)
    median_tpot = sorted_tpot[n // 2] if n % 2 == 1 else (sorted_tpot[n // 2 - 1] + sorted_tpot[n // 2]) / 2

    print(f"  - Mean TPOT: {mean_tpot*1000:.2f}ms")
    print(f"  - Median TPOT: {median_tpot*1000:.2f}ms")
    print(f"    (min: {min(all_tpot)*1000:.2f}ms, max: {max(all_tpot)*1000:.2f}ms)")

    # Total throughput
    total_tokens_processed = sum(r['prompt_len'] + r['generated_tokens'] for r in results)
    overall_throughput = total_tokens_processed / elapsed if elapsed > 0 else 0
    print(f"  - Overall throughput: {overall_throughput:.1f} tokens/sec")

    print(f"\nToken Length Distribution:")
    buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 4096), (4096, 8192), (8192, long_threshold), (long_threshold, 130000)]
    for low, high in buckets:
        count = sum(1 for r in results if low <= r['prompt_len'] < high)
        if count > 0:
            print(f"  - [{low:4d}-{high:4d} tokens]: {count} requests")

    # Sparse KV Cache Statistics (from collected samples)
    print(f"\nSparse KV Cache Statistics:")
    if results:
        long_reqs_with_offload = [r for r in results if r['prompt_len'] >= long_threshold]
        if long_reqs_with_offload:
            # Calculate offloaded tokens per request (prompt_len - first_n - last_m)
            first_n = 16
            last_m = 64
            total_offloaded_tokens = 0
            per_request_offload = []
            for r in long_reqs_with_offload:
                offloaded = max(0, r['prompt_len'] - first_n - last_m)
                total_offloaded_tokens += offloaded
                per_request_offload.append((r['req_id'], offloaded))

            avg_offloaded_tokens = total_offloaded_tokens / len(long_reqs_with_offload)
            # CPU KV cache: offloaded_tokens * num_layers * num_kv_heads * head_dim * 2 (K+V) * dtype_size
            # For Llama-3.1-8B: 32 layers, 8 kv_heads, 128 head_dim, 2 bytes (bfloat16)
            avg_cpu_kv_cache_gb = avg_offloaded_tokens * 32 * 8 * 128 * 2 / (1024**3)
            total_cpu_kv_cache_gb = total_offloaded_tokens * 32 * 8 * 128 * 2 / (1024**3)

            print(f"  - Total long requests: {len(long_reqs_with_offload)}")
            print(f"  - Total offloaded tokens (all long requests): {total_offloaded_tokens:,}")
            print(f"  - Avg offloaded tokens per long request: {avg_offloaded_tokens:,.0f}")
            print(f"  - Avg CPU KV cache per long request: {avg_cpu_kv_cache_gb:.2f}GB")
            print(f"  - Total CPU KV cache memory: {total_cpu_kv_cache_gb:.2f}GB")
            print(f"  - Per-request offload breakdown:")
            for req_id, offloaded in sorted(per_request_offload, key=lambda x: -x[1])[:10]:
                print(f"    {req_id}: {offloaded:,} tokens offloaded")
            if len(per_request_offload) > 10:
                print(f"    ... and {len(per_request_offload) - 10} more requests")

    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(test_complex_workload(args))

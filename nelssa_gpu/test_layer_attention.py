import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from flash_attn import flash_attn_with_kvcache
from termcolor import colored
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from model_hub.llama import LlamaModel
from nelssa_comm.NelssaClient import NelssaClient


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_test_config(model_name, input_len, attn_type, kv_dtype, retrieval_budget):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1] + ".json"
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE) as f:
        original_config = json.load(f)
    n_clusters = max(int(input_len / 16), 1)
    n_segments = max(int(input_len / 8192), 1)
    # compute the nearest multiple of (n_segments*32)
    lower = (n_clusters // (n_segments * 32)) * (n_segments * 32)
    upper = lower + (n_segments * 32)
    n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
    nprobe = max(round(n_clusters * retrieval_budget), 1)
    print("! test n_clusters", n_clusters)
    print("! test retrieval_budget", retrieval_budget)
    print("! test nprobe", nprobe)
    if attn_type == "RetroInfer" or attn_type == "NELSSA":
        original_config[attn_type]["n_centroids"] = n_clusters
        original_config[attn_type]["n_segment"] = n_segments
        original_config[attn_type]["nprobe"] = nprobe
        original_config[attn_type]["cache_cluster_num"] = nprobe * 3
        original_config[attn_type]["max_compute_cluster_num"] = max(
            int(n_clusters / 4), nprobe
        )
        original_config[attn_type]["kv_dtype"] = kv_dtype
        original_config[attn_type]["retrieval_budget"] = retrieval_budget
    original_config["RetroInfer"]["n_centroids"] = n_clusters
    original_config["RetroInfer"]["n_segment"] = n_segments
    original_config["RetroInfer"]["nprobe"] = nprobe
    original_config["RetroInfer"]["cache_cluster_num"] = nprobe * 3
    original_config["RetroInfer"]["max_compute_cluster_num"] = max(
        int(n_clusters / 4), nprobe
    )
    original_config["RetroInfer"]["kv_dtype"] = kv_dtype
    original_config["RetroInfer"]["retrieval_budget"] = retrieval_budget
    return original_config


def write_cache_to_file(layer_idx, start_bdx, key_states, attn_type):
    if layer_idx == 0:
        print(f"[write_cache_to_file] key_states shape : {key_states.shape}")
        file_name = (
            "test_" + attn_type + "_key_cache_before_b" + str(start_bdx) + ".txt"
        )
        with open(file_name, "w") as f:
            for bsz_idx in range(key_states.shape[0]):  # batch size
                for kv_head_idx in range(key_states.shape[2]):  # kv heads
                    f.write(f"batch {bsz_idx}, head{kv_head_idx}\n")
                    # 해당 batch와 head의 모든 sequence length 출력
                    for seqlen_idx in range(key_states.shape[1]):
                        # 각 sequence 위치의 head_dim 값을 쉼표로 구분하여 저장
                        row = key_states[bsz_idx, seqlen_idx, kv_head_idx]
                        row_str = ",".join([f"{x:.6f}" for x in row])
                        f.write(row_str + "\n")
                    f.write("\n")  # 각 head 간에 빈 줄 추가
        print(f"File name({file_name}) write cmpl")


def layer_prefill(llm, cache, layer_idx, start_bdx, hidden_states, attn_type):
    bsz, seq_len, dim = hidden_states.shape
    print(f"layer_idx : {layer_idx}, bsz: {bsz}, seq_len: {seq_len}, dim: {dim}")
    layer = llm.layers[layer_idx]
    # original hidden_states used as residual, clone a new one to process
    temp_hidden_states = hidden_states.clone()
    # chunk for lower memory comsumption
    for start_idx in range(0, seq_len, 8192 // bsz):
        end_idx = min(seq_len, start_idx + 8192 // bsz)
        temp_hidden_states[:, start_idx:end_idx, :] = llm.layernorm(
            temp_hidden_states[:, start_idx:end_idx, :],
            layer.input_layernorm_variance_epsilon,
            layer.input_layernorm_weight,
        )
    query_states, key_states, value_states = llm.wqkv(temp_hidden_states, layer)
    del temp_hidden_states
    torch.cuda.empty_cache()
    query_states, key_states = llm.position_embedd(query_states, key_states)
    query_states = query_states.view(
        bsz, seq_len, llm.num_heads, llm.head_dim
    )  # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
    key_states = key_states.view(bsz, seq_len, llm.num_key_value_heads, llm.head_dim)
    value_states = value_states.view(
        bsz, seq_len, llm.num_key_value_heads, llm.head_dim
    )
    key_states, value_states = cache.prefill_update_kv_cache(
        query_states, key_states, value_states, layer_idx, start_bdx
    )
    # write_cache_to_file(layer_idx, start_bdx, key_states, attn_type)
    torch.cuda.empty_cache()
    temp_attn_out = flash_attn_with_kvcache(
        query_states, key_states, value_states, causal=True
    )
    cache.sync(layer_idx, start_bdx)
    del query_states, key_states, value_states
    torch.cuda.empty_cache()
    hidden_states += llm.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
    del temp_attn_out
    torch.cuda.empty_cache()
    # post attention
    residual = hidden_states.clone()
    # chunk for lower memory comsumption
    for start_idx in range(0, seq_len, 8192 // bsz):
        end_idx = min(seq_len, start_idx + 8192 // bsz)
        hidden_states[:, start_idx:end_idx, :] = llm.layernorm(
            hidden_states[:, start_idx:end_idx, :],
            layer.post_attention_layernorm_variance_epsilon,
            layer.post_attention_layernorm_weight,
        )
        hidden_states[:, start_idx:end_idx, :] = llm.mlp(
            hidden_states[:, start_idx:end_idx, :], layer
        )
    hidden_states += residual
    del residual
    torch.cuda.empty_cache()
    return hidden_states


def save_retroinfer_result(
    batch_size, num_layers, input_len, hidden_states, lse, cluster_ids, device="cuda"
):
    filename = f"retroinfer_batch{batch_size}_layers{num_layers}_input{input_len}.pt"
    filepath = os.path.join(PROJECT_ROOT, "validation", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # GPU -> CPU 이동
    save_dict = {
        "hidden_states": hidden_states.cpu().detach(),
        "lse": lse.cpu().detach() if lse is not None else None,
        "cluster_ids": cluster_ids.cpu().detach() if cluster_ids is not None else None,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "input_len": input_len,
        "timestamp": time.time(),
    }
    torch.save(save_dict, filepath)
    print(colored(f"[SAVE] RetroInfer result saved to {filepath}", "green"))


def compare_with_retroinfer(
    batch_size,
    num_layers,
    input_len,
    hidden_states,
    lse,
    cluster_ids,
    atol=1e-3,
    rtol=1e-3,
):
    filename = f"retroinfer_batch{batch_size}_layers{num_layers}_input{input_len}.pt"
    filepath = os.path.join(PROJECT_ROOT, "validation", filename)
    if not os.path.exists(filepath):
        print(
            colored(
                f"[WARN] No RetroInfer result found for config: batch={batch_size}, layers={num_layers}, input_len={input_len}",
                "yellow",
            )
        )
        return False
    try:
        saved = torch.load(filepath, weights_only=True)
        print(colored(f"[LOAD] Loaded RetroInfer result from {filepath}", "green"))
        # hidden_states 비교
        hidden_states_cpu = hidden_states.cpu()
        saved_hidden_states = saved["hidden_states"]
        print(colored(f"hidden_states shape: {hidden_states_cpu.shape}", "white"))
        print(
            colored(f"saved_hidden_states shape: {saved_hidden_states.shape}", "white")
        )
        if not torch.allclose(
            hidden_states_cpu, saved_hidden_states, atol=atol, rtol=rtol
        ):
            diff = (hidden_states_cpu - saved_hidden_states).abs()
            max_diff = diff.max().item()
            diff_mask = diff > (atol + rtol * saved_hidden_states.abs())
            print(
                colored(
                    f"[FAIL] hidden_states mismatch! Max diff: {max_diff:.6f}", "red"
                )
            )
            # 틀린 위치 출력
            if diff_mask.any():
                # 차원별로 틀린 인덱스 찾기
                wrong_indices = torch.nonzero(diff_mask, as_tuple=False)
                print(
                    colored(
                        f"Number of mismatched elements: {wrong_indices.shape[0]}",
                        "red",
                    )
                )
                # 처음 10개의 틀린 위치만 출력
                num_show = min(10, wrong_indices.shape[0])
                print(
                    colored(
                        f"First {num_show} mismatched positions (batch, 1, n_heads, dim):",
                        "red",
                    )
                )
                for i in range(num_show):
                    idx = wrong_indices[i]
                    # batch_idx, tmp, seq_idx, dim_idx = idx[0].item(), idx[1].item(), idx[2].item(), idx[3].item()
                    # gpu_val = hidden_states_cpu[batch_idx, tmp, seq_idx, dim_idx].item()
                    # retro_val = saved_hidden_states[batch_idx, tmp, seq_idx, dim_idx].item()
                    # print(f"  Position [{batch_idx}, {tmp}, {seq_idx}, {dim_idx}]: GPU={gpu_val:.6f}, RetroInfer={retro_val:.6f}, Diff={abs(gpu_val-retro_val):.6f}")
                    nelssa_val = hidden_states_cpu[tuple(idx)].item()
                    retro_val = saved_hidden_states[tuple(idx)].item()
                    print(
                        f"  Position [{idx}]: NELSSA={nelssa_val:.6f}, RetroInfer={retro_val:.6f}, Diff={abs(nelssa_val - retro_val):.6f}"
                    )
        else:
            print(
                colored(
                    f"[PASS] hidden_states match within tolerance (atol={atol}, rtol={rtol})",
                    "green",
                )
            )
        # lse 비교
        if lse is not None and saved["lse"] is not None:
            lse_cpu = lse.cpu()
            saved_lse = saved["lse"]
            print(
                colored(
                    f"lse_cpu shape: {lse_cpu.shape}, lse_cpu dtype : {lse_cpu.dtype}",
                    "white",
                )
            )
            print(
                colored(
                    f"saved_lse shape: {saved_lse.shape}, saved_lse dtyupe: {saved_lse.dtype}",
                    "white",
                )
            )
            print(colored("unsqueeze saved_lse", "cyan"))
            saved_lse = saved_lse.unsqueeze(1)
            if not torch.allclose(lse_cpu, saved_lse, atol=atol, rtol=rtol):
                diff = (lse_cpu - saved_lse).abs()
                max_diff = diff.max().item()
                diff_mask = diff > (atol + rtol * saved_lse.abs())
                print(colored(f"[FAIL] lse mismatch! Max diff: {max_diff:.6f}", "red"))
                # 틀린 위치 출력
                if diff_mask.any():
                    wrong_indices = torch.nonzero(diff_mask, as_tuple=False)
                    print(
                        colored(
                            f"Number of mismatched elements in LSE: {wrong_indices.shape[0]}",
                            "red",
                        )
                    )
                    # 처음 10개의 틀린 위치만 출력
                    num_show = min(10, wrong_indices.shape[0])
                    print(colored(f"First {num_show} mismatched positions:", "red"))
                    for i in range(num_show):
                        idx = wrong_indices[i]
                        gpu_val = lse_cpu[tuple(idx)].item()
                        retro_val = saved_lse[tuple(idx)].item()
                        print(
                            f"  Position [{idx}]: GPU={gpu_val:.6f}, RetroInfer={retro_val:.6f}, Diff={abs(gpu_val - retro_val):.6f}"
                        )
            else:
                print(colored("[PASS] lse match within tolerance", "green"))
        elif lse is None and saved["lse"] is None:
            print(colored("[INFO] lse not computed in both", "blue"))
        else:
            print(colored("[WARN] lse comparison skipped (one side missing)", "yellow"))
        # cluster_ids 비교
        if cluster_ids is not None and saved["cluster_ids"] is not None:
            cluster_ids_cpu = cluster_ids.cpu()
            saved_cluster_ids = saved["cluster_ids"]
            print(colored(f"cluster_ids_cpu shape: {cluster_ids_cpu.shape}", "white"))
            print(
                colored(f"saved_cluster_ids shape: {saved_cluster_ids.shape}", "white")
            )
            # TODO : cluster ID 64b으로 변경해야함
            cluster_ids_cpu = cluster_ids_cpu.to(dtype=torch.int64)
            if not torch.allclose(cluster_ids_cpu, saved_cluster_ids, atol=0, rtol=0):
                diff = (cluster_ids_cpu - saved_cluster_ids).abs()
                max_diff = diff.max().item()
                diff_mask = diff > 0
                print(
                    colored(
                        f"[FAIL] cluster_ids mismatch! Max diff: {max_diff:.6f}", "red"
                    )
                )
                # 틀린 위치 출력
                if diff_mask.any():
                    wrong_indices = torch.nonzero(diff_mask, as_tuple=False)
                    print(
                        colored(
                            f"Number of mismatched elements in cluster_ids: {wrong_indices.shape[0]}",
                            "red",
                        )
                    )
                    # 처음 10개의 틀린 위치만 출력
                    num_show = min(10, wrong_indices.shape[0])
                    print(colored(f"First {num_show} mismatched positions:", "red"))
                    for i in range(num_show):
                        idx = wrong_indices[i]
                        print(
                            f"  Position {idx.tolist()}: GPU={cluster_ids_cpu[tuple(idx)].item():.0f}, RetroInfer={saved_cluster_ids[tuple(idx)].item():.0f}"
                        )
            else:
                print(colored("[PASS] cluster_ids match exactly", "green"))
        elif cluster_ids is None and saved["cluster_ids"] is None:
            print(colored("[INFO] cluster_ids not computed in both", "blue"))
        else:
            print(
                colored(
                    "[WARN] cluster_ids comparison skipped (one side missing)", "yellow"
                )
            )
        return True
    except Exception as e:
        print(colored(f"[ERROR] Failed to load or compare: {e}", "red"))
        return False


def compare_gpu_nelssa(
    hidden_states_gpu,
    lse_gpu,
    cluster_ids_gpu,
    hidden_states_nelssa,
    lse_nelssa,
    cluster_ids_nelssa,
    atol=1e-3,
    rtol=1e-3,
):
    all_passed = True
    # hidden_states 비교
    if hidden_states_gpu is not None and hidden_states_nelssa is not None:
        hidden_states_gpu_cpu = hidden_states_gpu.cpu()
        hidden_states_nelssa_cpu = hidden_states_nelssa.cpu()
        if not torch.allclose(
            hidden_states_gpu_cpu, hidden_states_nelssa_cpu, atol=atol, rtol=rtol
        ):
            diff = (hidden_states_gpu_cpu - hidden_states_nelssa_cpu).abs()
            max_diff = diff.max().item()
            diff_mask = diff > (atol + rtol * hidden_states_nelssa_cpu.abs())
            print(
                colored(
                    f"[FAIL] hidden_states mismatch! Max diff: {max_diff:.6f}", "red"
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
                    gpu_val = hidden_states_gpu_cpu[tuple(idx)].item()
                    nelssa_val = hidden_states_nelssa_cpu[tuple(idx)].item()
                    print(
                        f"  Position [{idx}]: GPU={gpu_val:.6f}, NELSSA={nelssa_val:.6f}, Diff={abs(gpu_val - nelssa_val):.6f}"
                    )
            all_passed = False
        else:
            print(colored("[PASS] hidden_states match within tolerance", "green"))
            # Sample value comparison for debugging
            gpu_flat = hidden_states_gpu_cpu.flatten()[:10]
            nelssa_flat = hidden_states_nelssa_cpu.flatten()[:10]
            for i in range(10):
                print(f"  [{i}]: GPU={gpu_flat[i].item():.6f}, "
                      f"NELSSA={nelssa_flat[i].item():.6f}")
    else:
        print(colored("[WARN] hidden_states not provided for comparison", "yellow"))
    # lse 비교
    if lse_gpu is not None and lse_nelssa is not None:
        lse_gpu_cpu = lse_gpu.cpu()
        lse_nelssa_cpu = lse_nelssa.cpu()
        # RetroInfer에서는 lse가 [B, H, S] 형태로 저장되는데, NELSSA에서는 [B, 1, H, S] 등일 수 있으므로 unsqueeze 처리 필요 시 추가 가능
        if lse_nelssa_cpu.dim() == 3 and lse_gpu_cpu.dim() == 3:
            pass  # 같은 차원
        elif lse_nelssa_cpu.dim() == 4 and lse_gpu_cpu.dim() == 3:
            lse_nelssa_cpu = lse_nelssa_cpu.squeeze(1)  # [B, 1, H, S] -> [B, H, S]
        elif lse_nelssa_cpu.dim() == 3 and lse_gpu_cpu.dim() == 4:
            lse_gpu_cpu = lse_gpu_cpu.unsqueeze(1)  # [B, H, S] -> [B, 1, H, S]
        if not torch.allclose(lse_gpu_cpu, lse_nelssa_cpu, atol=atol, rtol=rtol):
            diff = (lse_gpu_cpu - lse_nelssa_cpu).abs()
            max_diff = diff.max().item()
            diff_mask = diff > (atol + rtol * lse_nelssa_cpu.abs())
            print(colored(f"[FAIL] lse mismatch! Max diff: {max_diff:.6f}", "red"))
            # 틀린 위치 출력
            if diff_mask.any():
                wrong_indices = torch.nonzero(diff_mask, as_tuple=False)
                print(
                    colored(
                        f"Number of mismatched elements in LSE: {wrong_indices.shape[0]}",
                        "red",
                    )
                )
                # 처음 10개의 틀린 위치만 출력
                num_show = min(10, wrong_indices.shape[0])
                print(colored(f"First {num_show} mismatched positions:", "red"))
                for i in range(num_show):
                    idx = wrong_indices[i]
                    gpu_val = lse_gpu_cpu[tuple(idx)].item()
                    retro_val = lse_nelssa_cpu[tuple(idx)].item()
                    print(
                        f"  Position [{idx}]: GPU={gpu_val:.6f}, RetroInfer={retro_val:.6f}, Diff={abs(gpu_val - retro_val):.6f}"
                    )
        else:
            print(colored("[PASS] lse match within tolerance", "green"))
    elif lse_gpu is None and lse_nelssa is None:
        print(colored("[INFO] lse not computed in both", "blue"))
    else:
        print(colored("[WARN] lse comparison skipped (one side missing)", "yellow"))
    # cluster_ids 비교
    if cluster_ids_nelssa is not None and cluster_ids_gpu is not None:
        cluster_ids_cpu = cluster_ids_nelssa.cpu()
        saved_cluster_ids = cluster_ids_gpu
        # print(colored(f"cluster_ids_cpu shape: {cluster_ids_cpu.shape}", 'cyan'))
        # print(colored(f"saved_cluster_ids shape: {saved_cluster_ids.shape}", 'cyan'))
        # TODO : cluster ID 64b으로 변경해야함
        cluster_ids_cpu = cluster_ids_cpu.to(dtype=torch.int64)
        if not torch.allclose(cluster_ids_cpu, saved_cluster_ids, atol=0, rtol=0):
            diff = (cluster_ids_cpu - saved_cluster_ids).abs()
            max_diff = diff.max().item()
            diff_mask = diff > 0
            print(
                colored(f"[FAIL] cluster_ids mismatch! Max diff: {max_diff:.6f}", "red")
            )
            # 틀린 위치 출력
            if diff_mask.any():
                wrong_indices = torch.nonzero(diff_mask, as_tuple=False)
                print(
                    colored(
                        f"Number of mismatched elements in cluster_ids: {wrong_indices.shape[0]}",
                        "red",
                    )
                )
                # 처음 10개의 틀린 위치만 출력
                num_show = min(10, wrong_indices.shape[0])
                print(colored(f"First {num_show} mismatched positions:", "red"))
                for i in range(num_show):
                    idx = wrong_indices[i]
                    print(
                        f"  Position {idx.tolist()}: GPU={cluster_ids_cpu[tuple(idx)].item():.0f}, RetroInfer={saved_cluster_ids[tuple(idx)].item():.0f}"
                    )
        else:
            print(colored("[PASS] cluster_ids match exactly", "green"))
    elif cluster_ids_nelssa is None and cluster_ids_gpu is None:
        print(colored("[INFO] cluster_ids not computed in both", "blue"))
    else:
        print(
            colored(
                "[WARN] cluster_ids comparison skipped (one side missing)", "yellow"
            )
        )
    return True


def get_dtype(dtype):
    DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return DTYPE_MAP[dtype]


def main(args):
    # Set seed for reproducibility
    set_seed(2025)
    # Configuration
    model_name = args.model_name
    batch_size = args.batch_size
    input_len = args.input_len  # Use a reasonable input length
    gen_len = 2  # args.gen_len
    kv_dtype = get_dtype(args.dtype)
    device = args.device
    attn_type = args.attn_type
    num_layers = args.num_layers
    # Load test data
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    TEST_FILE = os.path.join(PROJECT_ROOT, "simple_test_data.json")
    data = json.load(open(TEST_FILE))
    prompt = []
    for dd in data:
        prompt.append(dd["input"])
    copy_round = math.ceil(batch_size / len(prompt))
    prompts = []
    for i in range(copy_round):
        prompts.extend(prompt)
    prompts = prompts[:batch_size]
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_masks = inputs.attention_mask.to(device)
    if input_len != 0:
        prefill_ids = input_ids[:, :input_len]
        attention_masks = attention_masks[:, -input_len:]  # left padding
    input_len = prefill_ids.shape[1]
    print(prefill_ids.shape)
    print(attention_masks)
    # Generate config
    attn_config = create_test_config(
        model_name, input_len, attn_type, kv_dtype, args.retrieval_budget
    )
    print(attn_config[attn_type])
    # Load model
    max_len = input_len + gen_len
    # TODO : To use bfloat16, this must be configured via the corresponding argument.
    llm = LlamaModel(
        model_name=model_name,
        max_length=max_len,
        dtype=torch.float16,
        device_map=device,
    )
    # Set attention_type for init_kv_cache
    llm.num_layers = args.num_layers
    llm.attention_type = attn_type
    llm.batch_size = args.batch_size
    llm.input_length = input_len
    llm.max_new_length = gen_len
    llm.prefill_bsz = 1
    llm.prefill_method = "full"
    valid_start = (
        attention_masks.shape[1]
        - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
    )
    print(f"valid start : {valid_start}")
    # Initialize NelssaClient for NELSSA
    nelssa_client = None
    if attn_type == "NELSSA":
        model_config = {
            "num_layers": llm.num_layers,
            "num_heads": llm.num_heads,
            "kv_heads": llm.num_key_value_heads,
            "head_dim": llm.head_dim,
            "n_clusters": attn_config["NELSSA"]["n_centroids"],
            "cache_unit_size": attn_config["NELSSA"]["pages_per_cluster"] * 8,
            "n_probe": attn_config["NELSSA"]["nprobe"],
            "kv_dtype": kv_dtype,
        }
        nelssa_client = NelssaClient(
            host=args.pnm_host,
            port=args.pnm_port,
            model_config=model_config,
        )
    # Init KV cache
    if attn_type == "NELSSA":
        llm.init_kv_cache(
            valid_start=valid_start,
            attn_config=attn_config,
            nelssa_client=nelssa_client,
        )
    else:
        llm.init_kv_cache(
            valid_start=valid_start,
            attn_config=attn_config,
        )
    cache = llm.kv_cache
    # Allocate test buffers for NELSSA
    if attn_type == "NELSSA":
        cache.allocate_test_buffer()
    # Simulate prefill for multi layer
    with torch.no_grad():
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()
        # TODO : To use bfloat16, this must be configured via the corresponding argument.
        last_hidden_states = torch.empty(
            (batch_size, 1, 4096), dtype=torch.float16, device=device
        )
        for start_bdx in range(0, batch_size, 1):
            end_bdx = min(batch_size, start_bdx + 1)
            hidden_states = llm.word_embedding(
                prefill_ids[start_bdx:end_bdx]
            )  # [1, seq_len, hidden_size]
            for ldx in range(num_layers):
                query_states = layer_prefill(
                    llm, cache, ldx, start_bdx, hidden_states, attn_type
                )
                torch.cuda.empty_cache()
            last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        last_hidden_states = llm.layernorm(
            last_hidden_states.contiguous(), llm.norm_variance_epsilon, llm.norm_weight
        )
        logits = llm.lm(last_hidden_states)
        output_ids = logits.argmax(dim=-1)
        hidden_states = (
            llm.word_embedding(output_ids).view(batch_size, 1, 32, 128).contiguous()
        )
        torch.cuda.synchronize()
        prefill_end = time.time()
        print(
            colored(
                f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n",
                "green",
            )
        )
        # Get attention output from GPU computation
        print("Start decoding ...")
        decode_start = time.time()
        if attn_type == "NELSSA" and args.only_attn:
            nel_hidden_states = hidden_states
            gpu_hidden_states = hidden_states
            for ldx in range(num_layers):
                # 같은 입력으로 비교
                test_input = hidden_states if ldx == 0 else nel_hidden_states
                nel_hidden_states, lse, cluster_ids = cache.sparse_attention_only(
                    test_input, ldx
                )
                gpu_hidden_states, gpu_lse, gpu_cluster_ids = cache.compute_using_gpu(
                    test_input, ldx
                )
                # retrieval 출력 직접 비교 (내부 디버깅용)
                if hasattr(cache, "last_nelssa_retrieval") and hasattr(
                    cache, "last_gpu_retrieval"
                ):
                    cache.last_nelssa_retrieval = (
                        cache.last_nelssa_retrieval.reshape_as(cache.last_gpu_retrieval)
                    )
                    r_diff = (
                        (cache.last_nelssa_retrieval - cache.last_gpu_retrieval)
                        .abs()
                        .max()
                    )
                    print(f"Layer {ldx} retrieval diff: {r_diff.item():.6f}")
                compare_gpu_nelssa(
                    gpu_hidden_states,
                    gpu_lse,
                    gpu_cluster_ids,
                    nel_hidden_states,
                    lse,
                    cluster_ids,
                    atol=1e-3,
                    rtol=1e-3,
                )
                cache.compare_test_buffers(verbose=True)
        elif attn_type == "NELSSA" and not args.only_attn:
            nel_hidden_states = hidden_states
            gpu_hidden_states = hidden_states
            for ldx in range(num_layers):
                nel_hidden_states = cache.compute_async(nel_hidden_states, ldx)
                gpu_hidden_states = cache.compute_gpu(gpu_hidden_states, ldx)
                compare_gpu_nelssa(
                    gpu_hidden_states,
                    lse_gpu=None,
                    cluster_ids_gpu=None,
                    hidden_states_nelssa=nel_hidden_states,
                    lse_nelssa=None,
                    cluster_ids_nelssa=None,
                    atol=1e-3,
                    rtol=1e-3,
                )
        else:
            for ldx in range(num_layers):
                input_states = hidden_states
                hidden_states, lse, cluster_ids = cache.compute_only_attn(
                    input_states, ldx
                )
            decode_end = time.time()
            # !!! Please do not compare the performance of Nelssa and GPU in this code.
            # !!! When running RetroInfer on GPU, there is cache access overhead at every layer traversal.
            print(
                colored(
                    f"Decoding latency: {round((decode_end - decode_start) * 1000 / (gen_len - 1), 2)} ms/step, "
                    f"Throughput: {round(batch_size * (gen_len - 1) / (decode_end - decode_start), 2)} tokens/s\n",
                    "green",
                )
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/sylee/dataset/Llama-3-8B-Instruct-Gradient-1048k",
        choices=[
            "/home/skchoi/LLM_Models/Llama-3.1-8B",
            "/home/sylee/dataset/Llama-3-8B-Instruct-Gradient-1048k",
        ],
        help="huggingface model name",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--input_len", type=int, default=4096, help="Slice length")
    parser.add_argument("--gen_len", type=int, default=2, help="Generation length")
    parser.add_argument("--num_layers", type=int, default=1, help="Numer of layers")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Dtype"
    )
    parser.add_argument(
        "--retrieval_budget", type=float, default=0.018, help="Retrieval budget"
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        default="NELSSA",
        choices=["Full_Flash_Attn", "RetroInfer", "NELSSA"],
        help="Attention method",
    )
    parser.add_argument(
        "--only_attn", action="store_true", help="Verify only sparse attention function"
    )
    parser.add_argument(
        "--data_path", type=str, default="", help="Input json file path"
    )
    parser.add_argument(
        "--pnm_host",
        type=str,
        default="10.0.0.2",
        help="PNM server host address",
    )
    parser.add_argument(
        "--pnm_port",
        type=int,
        default=50058,
        help="PNM server port",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    main(args)

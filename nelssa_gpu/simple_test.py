import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from config import add_config_args, generate_config  # noqa: E402
from model_hub import add_model_args, load_model, load_tokenizer  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Total Batch size")
    parser.add_argument(
        "--prefill_bsz", type=int, default=1, help="Prefilling batch size"
    )
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--context_len", type=int, default=0, help="Context length")
    parser.add_argument(
        "--do_sample", action="store_true", help="Whether to use sampling when decoding"
    )
    parser.add_argument(
        "--prefill_method",
        type=str,
        default="full",
        choices=["full", "xattn", "minfer"],
        help="Prefilling method",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="simple_test_data.json",
        help="Input json file path",
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
    parser = add_model_args(parser)
    parser = add_config_args(parser)
    args = parser.parse_args()

    return args


def load_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        else:  # jsonl
            return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    print(args)

    model_name = args.model_name
    batch_size = args.batch_size
    attn_type = args.attn_type
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = args.device

    # load input data
    TEST_FILE = os.path.join(PROJECT_ROOT, f"{args.data_path}")
    print(colored(f"Loading test data from {TEST_FILE}", "yellow"))
    data = load_data(TEST_FILE)
    if type(data) is dict:
        data = [data]
    prompt, groundtruth = [], []
    for dd in data:
        prompt.append(dd["input"])
        groundtruth.append(dd["outputs"])

    # copy to fit batch size
    copy_round = math.ceil(batch_size / len(prompt))
    prompts, groundtruths = [], []
    for i in range(copy_round):
        prompts.extend(prompt)
        groundtruths.extend(groundtruth)
    prompts = prompts[:batch_size]
    groundtruths = groundtruths[:batch_size]

    # tokenize input data
    tokenizer = load_tokenizer(model_name)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    if args.context_len != 0:
        input_ids = input_ids[:, : args.context_len]
        attention_masks = attention_masks[:, -args.context_len :]  # left padding

    input_len = input_ids.shape[1]
    gen_len = args.gen_len
    max_len = input_len + gen_len
    print(colored(f"Input length: {input_len}, Gen length: {gen_len}", "yellow"))

    attn_config = generate_config(
        model_name,
        input_len,
        attn_type,
        args,
        float(args.retrieval_budget),
        float(args.estimation_budget),
        float(args.cache_ratio),
        args.use_cuda_graph,
        args.gpu_only,
    )
    llm = load_model(model_name, max_len, dtype, device, tokenizer)

    print("Hi")
    # NelssaClient initialization
    nelssa_client = None
    if attn_type == "NELSSA":
        import os
        import sys

        print(f"DEBUG: Current sys.path: {sys.path}")

        from nelssa_comm.NelssaClient import NelssaClient

        # Construct model config for NelssaClient
        # We use a simplified config or a dedicated one
        n_clusters = max(int(input_len / 16), 1)
        nprobe = max(round(n_clusters * float(args.retrieval_budget)), 1)
        model_config = {
            "num_layers": llm.num_layers,
            "num_heads": llm.num_heads,
            "kv_heads": llm.num_key_value_heads,
            "head_dim": llm.head_dim,
            "n_clusters": n_clusters,  # matches n_centroids in attn_config
            "cache_unit_size": 32 * 8,  # pages_per_cluster * page_size
            "n_probe": nprobe,  # retrieval_budget * n_centroids
            "kv_dtype": torch.float16 if args.dtype == "fp16" else torch.bfloat16,
        }
        nelssa_client = NelssaClient(
            host=args.pnm_host,
            port=args.pnm_port,
            model_config=model_config,
        )

    out = llm.generate(
        attention_type=attn_type,
        inputs_ids=input_ids.to(llm.layers[0].device),
        attention_masks=attention_masks,
        max_new_length=gen_len,
        attn_config=attn_config,
        do_sample=args.do_sample,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        ignore_eos=False if args.do_sample else True,
        prefill_bsz=args.prefill_bsz,
        prefill_method=args.prefill_method,
        nelssa_client=nelssa_client,
    )

    result = tokenizer.batch_decode(out, skip_special_tokens=True)
    for gt, res in zip(groundtruths, result):
        print(colored(f"Answer: {gt}", "yellow"))
        print(f"{[res]}")

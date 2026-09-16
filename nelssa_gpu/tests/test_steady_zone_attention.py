"""
Steady Zone Attention 비교 테스트: opt=False (Native) vs opt=True (FlashAttn)
"""

import math

import torch
from flash_attn import flash_attn_func


def steady_zone_attention_native(
    queries: torch.Tensor,
    steady_zone_keys: torch.Tensor,
    steady_zone_values: torch.Tensor,
    static_len_tensor: torch.Tensor,
    batch_groups: int,
    group_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Native attention implementation (opt=False)

    Args:
        queries: [batch_size, 1, num_heads, head_dim]
        steady_zone_keys: [batch_size, kv_head, static_len, head_dim]
        steady_zone_values: [batch_size, kv_head, static_len, head_dim]
        static_len_tensor: scalar tensor, valid length of steady zone
        batch_groups: batch_size * kv_head
        group_size: num_heads // kv_head
        head_dim: head dimension

    Returns:
        steady_out: [batch_groups, group_size, 1, head_dim]
        steady_lse: [batch_groups, 1, group_size, 1]
    """
    static_len = static_len_tensor.item()

    # Slice steady zone up to static_len
    s_keys = steady_zone_keys[..., :static_len, :].contiguous()
    s_vals = steady_zone_values[..., :static_len, :].contiguous()

    # Reshape for GQA: [B, KV, L, D] -> [B*KV, 1, L, D]
    s_keys = s_keys.view(batch_groups, 1, -1, head_dim)
    s_vals = s_vals.view(batch_groups, 1, -1, head_dim)

    # Reshape queries: [B, 1, H, D] -> [B*KV, G, 1, D]
    q_pt = queries.view(batch_groups, group_size, 1, head_dim)

    # Native attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_pt, s_keys.transpose(-2, -1)) * scale  # [B*KV, G, 1, L]

    s_max = torch.max(scores, dim=-1, keepdim=True)[0]  # [B*KV, G, 1, 1]
    s_exp = torch.exp(scores - s_max)
    s_sum = torch.sum(s_exp, dim=-1, keepdim=True)  # [B*KV, G, 1, 1]
    s_sum = torch.clamp(s_sum, min=1e-9)

    probs = s_exp / s_sum
    s_out = torch.matmul(probs, s_vals)  # [B*KV, G, 1, D]

    steady_out = s_out.permute(0, 2, 1, 3)  # [B*KV, 1, G, D]
    steady_lse = (
        (s_max + torch.log(s_sum)).permute(0, 2, 1, 3).to(torch.float32)
    )  # [B*KV, 1, G, 1]

    return steady_out, steady_lse


def steady_zone_attention_flash(
    queries: torch.Tensor,
    steady_zone_keys: torch.Tensor,
    steady_zone_values: torch.Tensor,
    static_len_tensor: torch.Tensor,
    batch_size: int,
    kv_head: int,
    num_heads: int,
    group_size: int,
    batch_groups: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention implementation (opt=True)

    Returns:
        steady_out: [batch_groups, group_size, 1, head_dim]
        steady_lse: [batch_groups, 1, group_size, 1]
    """
    static_len = static_len_tensor.item()

    # Slice steady zone
    s_keys = steady_zone_keys[..., :static_len, :].contiguous()
    s_vals = steady_zone_values[..., :static_len, :].contiguous()

    # FlashAttn input: [batch, seq_len, num_heads, head_dim]
    # steady_zone_keys shape: [B, KV, L, D] -> FlashAttn needs [B, L, KV, D]
    q_fa = queries.view(batch_size, 1, num_heads, head_dim)  # [B, 1, H, D]
    k_fa = s_keys.permute(0, 2, 1, 3).contiguous()  # [B, L, KV, D]
    v_fa = s_vals.permute(0, 2, 1, 3).contiguous()  # [B, L, KV, D]

    # FlashAttention
    scale = 1.0 / math.sqrt(head_dim)
    steady_out_fa, steady_lse_fa, _ = flash_attn_func(
        q_fa,
        k_fa,
        v_fa,
        dropout_p=0.0,
        softmax_scale=scale,
        return_attn_probs=True,
    )
    # steady_out_fa: [B, 1, H, D]
    # steady_lse_fa: [B, H, 1]

    # Reshape output to match native format
    # Native: queries.view(batch_groups, group_size, 1, head_dim)
    #       = [B*KV, G, 1, D]
    #       여기서 batch_groups = B * KV
    #       queries[batch_idx, group_idx, :, :] 는
    #       실제 query head index = batch_idx * group_size + group_idx 가 아님!
    #
    # Native 에서 queries.reshape(B*KV, G, 1, D) 의미:
    #   batch_groups = B * KV, 즉 (b, kv) 쌍을 하나의 차원으로 flatten
    #   queries[b, kv, g, 0, :] = original queries[b, 0, kv*group_size + g, :]
    #
    # FlashAttn output: [B, 1, H, D] where H = num_heads
    # FlashAttn GQA: query head h attends KV head[h // group_size]
    #
    # Native 출력: [B*KV, 1, G, D]
    #   out[b*kv + kv_idx, 1, g, :] = attention output for query head (kv_idx*group_size + g)
    #
    # FlashAttn 출력 변환:
    #   [B, 1, H, D] -> H 를 [KV, G] 로 분해 -> [B, 1, KV, G, D]
    #   -> permute để [B, KV, 1, G, D] -> flatten để [B*KV, 1, G, D]
    steady_out = steady_out_fa.squeeze(1)  # [B, H, D]
    steady_out = steady_out.view(
        batch_size, kv_head, group_size, head_dim
    )  # [B, KV, G, D]
    steady_out = steady_out.permute(0, 1, 2, 3).reshape(
        batch_groups, 1, group_size, head_dim
    )  # [B*KV, 1, G, D]

    # LSE: FlashAttn [B, H, 1], Native [B*KV, 1, G, 1]
    steady_lse = steady_lse_fa.squeeze(-1)  # [B, H]
    steady_lse = steady_lse.view(batch_size, kv_head, group_size)  # [B, KV, G]
    steady_lse = steady_lse.reshape(batch_groups, group_size)  # [B*KV, G]
    steady_lse = steady_lse.unsqueeze(1).unsqueeze(-1)  # [B*KV, 1, G, 1]

    return steady_out, steady_lse


def test_steady_zone_attention():
    """Compare Native vs FlashAttention outputs"""
    torch.manual_seed(42)

    # Config - smaller for easier debugging
    batch_size = 4
    kv_head = 8
    num_heads = 32
    group_size = num_heads // kv_head  # 2
    batch_groups = batch_size * kv_head  # 2
    head_dim = 128
    static_len = 1024

    print(f"Config: batch={batch_size}, kv_head={kv_head}, num_heads={num_heads}")
    print(
        f"        group_size={group_size}, batch_groups={batch_groups}, head_dim={head_dim}"
    )
    print(f"        static_len={static_len}")

    # Create random inputs (fp16 for FlashAttn)
    queries = torch.randn(
        batch_size, 1, num_heads, head_dim, dtype=torch.float16, device="cuda"
    )
    steady_zone_keys = torch.randn(
        batch_size, kv_head, static_len, head_dim, dtype=torch.float16, device="cuda"
    )
    steady_zone_values = torch.randn(
        batch_size, kv_head, static_len, head_dim, dtype=torch.float16, device="cuda"
    )
    static_len_tensor = torch.tensor(static_len, dtype=torch.int32, device="cuda")

    # Print shape analysis
    print("\n--- Shape Analysis ---")
    print(f"queries.shape: {queries.shape}")
    print(f"steady_zone_keys.shape: {steady_zone_keys.shape}")
    print(f"queries[0,0,0,:5]: {queries[0, 0, 0, :5]}")

    # Native reshape
    q_native = queries.view(batch_groups, group_size, 1, head_dim)
    k_native = steady_zone_keys.view(batch_groups, 1, static_len, head_dim)
    print(f"Native q_pt.shape: {q_native.shape}")  # [B*KV, G, 1, D]
    print(f"Native s_keys.shape: {k_native.shape}")  # [B*KV, 1, L, D]
    print(f"q_native[0,0,0,:5]: {q_native[0, 0, 0, :5]}")  # batch=0, kv=0, group=0
    print(f"q_native[1,0,0,:5]: {q_native[1, 0, 0, :5]}")  # batch=0, kv=1, group=0

    # FlashAttn reshape
    q_fa = queries.view(batch_size, 1, num_heads, head_dim)
    k_fa = steady_zone_keys.view(batch_size, static_len, kv_head, head_dim)
    print(f"FlashAttn q_fa.shape: {q_fa.shape}")  # [B, 1, H, D]
    print(f"FlashAttn k_fa.shape: {k_fa.shape}")  # [B, L, KV, D]
    print(f"q_fa[0,0,0,:5]: {q_fa[0, 0, 0, :5]}")  # batch=0, head=0
    print(f"q_fa[0,0,2,:5]: {q_fa[0, 0, 2, :5]}")  # batch=0, head=2 (kv=1, group=0)

    # Run Native
    out_native, lse_native = steady_zone_attention_native(
        queries,
        steady_zone_keys,
        steady_zone_values,
        static_len_tensor,
        batch_groups,
        group_size,
        head_dim,
    )
    print("\n--- Native Output Details ---")
    print(f"out_native[0,0,0,:5]: {out_native[0, 0, 0, :5]}")  # batch_group=0, group=0
    print(f"out_native[0,0,1,:5]: {out_native[0, 0, 1, :5]}")  # batch_group=0, group=1
    print(f"out_native[1,0,0,:5]: {out_native[1, 0, 0, :5]}")  # batch_group=1, group=0
    print(f"lse_native[0,0,0]: {lse_native[0, 0, 0]}")
    print(f"lse_native[1,0,0]: {lse_native[1, 0, 0]}")

    # Run FlashAttn
    out_flash, lse_flash = steady_zone_attention_flash(
        queries,
        steady_zone_keys,
        steady_zone_values,
        static_len_tensor,
        batch_size,
        kv_head,
        num_heads,
        group_size,
        batch_groups,
        head_dim,
    )
    print("\n--- Flash Output Details ---")
    print(f"out_flash[0,0,0,:5]: {out_flash[0, 0, 0, :5]}")
    print(f"out_flash[0,0,1,:5]: {out_flash[0, 0, 1, :5]}")
    print(f"out_flash[1,0,0,:5]: {out_flash[1, 0, 0, :5]}")
    print(f"lse_flash[0,0,0]: {lse_flash[0, 0, 0]}")
    print(f"lse_flash[1,0,0]: {lse_flash[1, 0, 0]}")

    print("\n--- Output Shapes ---")
    print(f"Native out: {out_native.shape}, Flash out: {out_flash.shape}")
    print(f"Native LSE: {lse_native.shape}, Flash LSE: {lse_flash.shape}")

    print("\n--- Output Comparison ---")
    print(f"Native out  range: [{out_native.min():.6f}, {out_native.max():.6f}]")
    print(f"Flash out   range: [{out_flash.min():.6f}, {out_flash.max():.6f}]")

    diff_out = (out_native - out_flash).abs()
    print(f"Output diff       : mean={diff_out.mean():.6f}, max={diff_out.max():.6f}")

    print("\n--- LSE Comparison ---")
    print(f"Native LSE range: [{lse_native.min():.6f}, {lse_native.max():.6f}]")
    print(f"Flash LSE  range: [{lse_flash.min():.6f}, {lse_flash.max():.6f}]")

    diff_lse = (lse_native - lse_flash).abs()
    print(f"LSE diff        : mean={diff_lse.mean():.6f}, max={diff_lse.max():.6f}")

    # Check if outputs match
    out_close = torch.allclose(out_native, out_flash, rtol=1e-3, atol=1e-3)
    lse_close = torch.allclose(lse_native, lse_flash, rtol=1e-3, atol=1e-3)

    print("\n--- Results ---")
    print(f"Output close (rtol=1e-3): {out_close}")
    print(f"LSE close (rtol=1e-3): {lse_close}")

    if not out_close:
        print(f"  Output mismatch! Max diff: {diff_out.max():.6f}")
    if not lse_close:
        print(f"  LSE mismatch! Max diff: {diff_lse.max():.6f}")

    # Detailed element-wise analysis
    print("\n--- Element-wise Analysis ---")
    print(
        f"Output: {diff_out.numel()} elements, {torch.sum(diff_out > 1e-3).item()} > 1e-3, {torch.sum(diff_out > 1e-2).item()} > 1e-2"
    )
    print(
        f"LSE: {diff_lse.numel()} elements, {torch.sum(diff_lse > 1e-3).item()} > 1e-3, {torch.sum(diff_lse > 1e-2).item()} > 1e-2"
    )

    return {
        "out_native": out_native,
        "out_flash": out_flash,
        "lse_native": lse_native,
        "lse_flash": lse_flash,
        "diff_out": diff_out,
        "diff_lse": diff_lse,
    }


if __name__ == "__main__":
    test_steady_zone_attention()

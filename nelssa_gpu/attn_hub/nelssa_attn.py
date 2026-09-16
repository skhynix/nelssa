def nelssa_decode_attn(query_states, key_states, value_states, layer_idx, nelss_cache):
    """
    query_states: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
    """
    # assert query_states.size(0) == retroinfer_cache.batch_size
    # assert query_states.size(1) == 1
    # assert query_states.size(2) == retroinfer_cache.kv_head * retroinfer_cache.group_size == retroinfer_cache.num_heads
    # assert query_states.size(3) == retroinfer_cache.head_dim

    static_len = (
        nelss_cache.static_pattern_total
        if layer_idx == nelss_cache.layer_num - 1
        else nelss_cache.static_pattern_total + 1
    )
    return nelss_cache.attn_func(query_states.contiguous(), layer_idx, static_len)

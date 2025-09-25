import torch
'''
This function was taken and adapted from the Huggingface implementation for LLaMa
'''
def update_causal_mask(
    model,
    attention_mask,
    input_tensor,
    past_key_values = None,
):
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + input_tensor.shape[1], device=input_tensor.device
    )
    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if model.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = False
    
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
        elif attention_mask.dim() == 4:
            # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
            # cache. In that case, the 4D attention mask attends to the newest tokens only.
            if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                logger.warning_once(
                    "Passing a 4d mask shorter than the input length is deprecated and will be removed in "
                    "transformers v4.42.0"
                )
                offset = cache_position[0]
            else:
                offset = 0
            mask_shape = attention_mask.shape
            mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
            causal_mask[
                : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
            ] = mask_slice

    
    return causal_mask

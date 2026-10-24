#pragma once

#include "kv_cache.h"
#include "tensor.h"

#include <cstddef>

namespace tinyinfer{

// causal decoder block with:
// biased QKV projections, half-split RoPE, GQA, and SwiGLU.
struct DecoderBlockConfig{
    size_t hidden_size;
    size_t intermediate_size;
    size_t query_heads;
    size_t kv_heads;
    float rms_norm_epsilon;
    float rope_theta;
};

struct DecoderBlockWeights{
    Tensor input_norm;
    Tensor q_weight;
    Tensor q_bias;
    Tensor k_weight;
    Tensor k_bias;
    Tensor v_weight;
    Tensor v_bias;
    Tensor o_weight;
    Tensor post_attention_norm;
    Tensor gate_weight;
    Tensor up_weight;
    Tensor down_weight;
};

Tensor rope_half_split(
    const Tensor& input,
    size_t position_offset,
    float theta
);

Tensor decoder_block_prefill(
    const Tensor& hidden_states,
    const DecoderBlockConfig& config,
    const DecoderBlockWeights& weights,
    ContiguousKvCache& cache
);

Tensor decoder_block_decode(
    const Tensor& hidden_state,
    const DecoderBlockConfig& config,
    const DecoderBlockWeights& weights,
    ContiguousKvCache& cache
);

}

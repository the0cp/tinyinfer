#include "decoder_block.h"

#include "attention.h"
#include "ops.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace tinyinfer{

namespace{

void require_shape(const Tensor& tensor, const Shape& expected, const char* name){
    if(tensor.shape() != expected){
        throw std::invalid_argument(std::string(name) + " shape mismatch.");
    }
    if(tensor.dtype() != DataType::Float32 || !tensor.is_contiguous()){
        throw std::invalid_argument(
            std::string(name) + " must be contiguous float32."
        );
    }
}

// The model loader will move fixed config/weight checks to construction time.
// Until then, the free-function API keeps the complete contract here.
size_t validate_block(
    const Tensor& hidden_states,
    const DecoderBlockConfig& config,
    const DecoderBlockWeights& weights,
    const ContiguousKvCache& cache
){
    if(
        config.hidden_size == 0 || config.intermediate_size == 0 ||
        config.query_heads == 0 || config.kv_heads == 0
    ){
        throw std::invalid_argument("Decoder block dimensions must be positive.");
    }

    if(config.hidden_size % config.query_heads != 0){
        throw std::invalid_argument("Decoder hidden size must be divisible by query heads.");
    }

    if(config.query_heads % config.kv_heads != 0){
        throw std::invalid_argument("Decoder query heads must be divisible by KV heads.");
    }

    const size_t head_dim = config.hidden_size / config.query_heads;

    if(head_dim % 2 != 0){
        throw std::invalid_argument("Half-split RoPE requires an even head dimension.");
    }

    if(!std::isfinite(config.rms_norm_epsilon) || config.rms_norm_epsilon <= 0.0f){
        throw std::invalid_argument("Decoder RMSNorm epsilon must be finite and positive.");
    }

    if(!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f){
        throw std::invalid_argument("Decoder RoPE theta must be finite and positive.");
    }

    if(hidden_states.dim() != 2 || hidden_states.shape()[1] != config.hidden_size){
        throw std::invalid_argument(
            "Decoder hidden states must have shape [tokens, hidden_size]."
        );
    }

    if(hidden_states.shape()[0] == 0){
        throw std::invalid_argument("Decoder block requires at least one token.");
    }

    if(hidden_states.dtype() != DataType::Float32 || !hidden_states.is_contiguous()){
        throw std::invalid_argument("Decoder hidden states must be contiguous float32.");
    }

    const size_t kv_size = config.kv_heads * head_dim;

    require_shape(weights.input_norm, {config.hidden_size}, "input_norm");
    require_shape(
        weights.q_weight,
        {config.hidden_size, config.hidden_size},
        "q_weight"
    );
    require_shape(weights.q_bias, {config.hidden_size}, "q_bias");
    require_shape(weights.k_weight, {config.hidden_size, kv_size}, "k_weight");
    require_shape(weights.k_bias, {kv_size}, "k_bias");
    require_shape(weights.v_weight, {config.hidden_size, kv_size}, "v_weight");
    require_shape(weights.v_bias, {kv_size}, "v_bias");
    require_shape(
        weights.o_weight,
        {config.hidden_size, config.hidden_size},
        "o_weight"
    );
    require_shape(
        weights.post_attention_norm,
        {config.hidden_size},
        "post_attention_norm"
    );
    require_shape(
        weights.gate_weight,
        {config.hidden_size, config.intermediate_size},
        "gate_weight"
    );
    require_shape(
        weights.up_weight,
        {config.hidden_size, config.intermediate_size},
        "up_weight"
    );
    require_shape(
        weights.down_weight,
        {config.intermediate_size, config.hidden_size},
        "down_weight"
    );

    if(cache.num_heads() != config.kv_heads || cache.head_dim() != head_dim){
        throw std::invalid_argument("Decoder block shape does not match the KV cache.");
    }
    return head_dim;
}

Tensor finish_block(
    const Tensor& hidden_states,
    const Tensor& attention,
    const DecoderBlockConfig& config,
    const DecoderBlockWeights& weights
){
    // Concat attention heads and mix their features
    // [T, num_q_heads, head_dim] -> [T, hidden_size]
    // O = Concat(head_0, ..., head_n) * W_O

    const Tensor attention_output = linear(
        attention.reshape({hidden_states.shape()[0], config.hidden_size}),
        weights.o_weight
    );

    // H1 = X + O: Preserves an identity path for information and gradients.
    const Tensor attention_residual = add(hidden_states, attention_output);

    // Pre-MLP normalization.
    // U = RMSNorm(H1)
    const Tensor normalized = rms_norm(
        attention_residual,
        weights.post_attention_norm,
        config.rms_norm_epsilon
    );

    // SwiGLU FFN
    // F = SiLU(Gate) element-wise multiplied by Up
    // SiLU(x) = x * sigmoid(x)
    // F = SiLU(Gate) \odot Up
    const Tensor gate = linear(normalized, weights.gate_weight);
    const Tensor up = linear(normalized, weights.up_weight);

    // Down projection
    // D = F * W_down: [T, 4864] -> [T, 896]
    const Tensor feed_forward = linear(silu_mul(gate, up), weights.down_weight);
    
    // MLP residual connection
    // H_out = H1 + D: [T, hidden_size]
    return add(attention_residual, feed_forward);
}

struct Qkv{
    Tensor queries;
    Tensor keys;
    Tensor values;
};

Qkv project_qkv(
    const Tensor& hidden_states,
    const DecoderBlockConfig& config,
    const DecoderBlockWeights& weights,
    size_t head_dim,
    size_t position_offset
){
    const Tensor normalized = rms_norm(
        hidden_states,
        weights.input_norm,
        config.rms_norm_epsilon
    );

    const size_t tokens = hidden_states.shape()[0];

    Tensor queries = linear(normalized, weights.q_weight, weights.q_bias).reshape(
        {tokens, config.query_heads, head_dim}
    );

    Tensor keys = linear(normalized, weights.k_weight, weights.k_bias).reshape(
        {tokens, config.kv_heads, head_dim}
    );

    Tensor values = linear(normalized, weights.v_weight, weights.v_bias).reshape(
        {tokens, config.kv_heads, head_dim}
    );

    // Apply RoPE to Q and K before computing QK^T.
    // V remains unchanged because it carries the content
    return {
        rope_half_split(queries, position_offset, config.rope_theta),
        rope_half_split(keys, position_offset, config.rope_theta),
        std::move(values)
    };
}

}

/*
 * Rotary Position Embedding (RoPE)
 *
 * For dimension pair i at position p:
 *   omega_i = theta^(-2i / head_dim)
 *   angle   = p * omega_i
 *   q'_p    = R(angle) * q_p
 *   k'_p    = R(angle) * k_p
 *
 *   R(p*omega)^T * R(s*omega) = R((s-p)*omega)
 *
 * Therefore, the Q-K dot product depends on relative position (s-p), not only on absolute positions.
*/
Tensor rope_half_split(const Tensor& input, size_t position_offset, float theta){
    // input: [tokens, heads, head_dim]
    // RoPE rotates Q/K vectors without changing their shapes.
    // theta: RoPE frequency base

    // Each head is divided into d/2 independent 2D pairs.

    if(input.dim() != 3 || input.dtype() != DataType::Float32 || !input.is_contiguous()){
        throw std::invalid_argument(
            "rope_half_split expects contiguous float32 [tokens, heads, head_dim]."
        );
    }

    const size_t head_dim = input.shape()[2];

    if(input.shape()[0] == 0 || input.shape()[1] == 0 || head_dim == 0){
        throw std::invalid_argument("rope_half_split dimensions must be positive.");
    }

    if(head_dim % 2 != 0){
        throw std::invalid_argument("rope_half_split requires an even head dimension.");
    }

    if(!std::isfinite(theta) || theta <= 0.0f){
        throw std::invalid_argument("rope_half_split theta must be finite and positive.");
    }

    Tensor output(input.shape());
    const size_t half = head_dim / 2;
    const size_t heads = input.shape()[1];

    for(size_t i = 0; i < half; i++){
        // Frequency of the i-th dimension pair:
        // omega_i = theta^(-2i / head_dim)

        // Different pairs use different frequencies for position information at multiple scales.
        const float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        const float inverse_frequency = 1.0f / std::pow(theta, exponent);

        for(size_t token = 0; token < input.shape()[0]; token++){
            const float position = static_cast<float>(position_offset + token);
            // angle(p, i) = p * omega_i
            const float angle = position * inverse_frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);

            for(size_t head = 0; head < heads; head++){
                const size_t offset = (token * heads + head) * head_dim;

                // (x[i], x[i + head_dim/2])
                const float first = input.data()[offset + i];
                const float second = input.data()[offset + half + i];
                
                // 2D rotation:
                // x' = x*cos(angle) - y*sin(angle)
                // y' = x*sin(angle) + y*cos(angle)
                output.data()[offset + i] = first * cosine - second * sine;
                output.data()[offset + half + i] = second * cosine + first * sine;
            }
        }
    }

    return output;
}

Tensor decoder_block_prefill(
    const Tensor& hidden_states,
    const DecoderBlockConfig& config,
    const DecoderBlockWeights& weights,
    ContiguousKvCache& cache
){
    const size_t head_dim = validate_block(hidden_states, config, weights, cache);

    if(cache.size_tokens() != 0){
        throw std::logic_error("Decoder block prefill requires an empty KV cache.");
    }

    if(hidden_states.shape()[0] > cache.capacity_tokens()){
        throw std::length_error("Decoder block prefill exceeds KV cache capacity.");
    }

    const Qkv qkv = project_qkv(hidden_states, config, weights, head_dim, 0);
    const Tensor attention = attention_prefill(
        qkv.queries,
        qkv.keys,
        qkv.values,
        cache
    );

    return finish_block(hidden_states, attention, config, weights);
}

Tensor decoder_block_decode(
    const Tensor& hidden_state,
    const DecoderBlockConfig& config,
    const DecoderBlockWeights& weights,
    ContiguousKvCache& cache
){
    const size_t head_dim = validate_block(hidden_state, config, weights, cache);

    if(hidden_state.shape()[0] != 1){
        throw std::invalid_argument("Decoder block decode accepts exactly one token.");
    }

    if(cache.size_tokens() == cache.capacity_tokens()){
        throw std::length_error("Decoder block decode exceeds KV cache capacity.");
    }

    const Qkv qkv = project_qkv(
        hidden_state,
        config,
        weights,
        head_dim,
        cache.size_tokens()
    );

    const Tensor attention = attention_decode(
        qkv.queries,
        qkv.keys,
        qkv.values,
        cache
    );

    return finish_block(hidden_state, attention, config, weights);
}

}

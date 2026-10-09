#include "attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace tinyinfer{

namespace{

struct AttentionShape{
    size_t tokens;
    size_t heads;
    size_t head_dim;
};

void validate_tensor(const Tensor& tensor, const char* name){
    if(tensor.dtype() != DataType::Float32 || !tensor.is_contiguous()){
        throw std::invalid_argument(
            std::string(name) + " must be contiguous float32."
        );
    }
    if(tensor.dim() != 3){
        throw std::invalid_argument(
            std::string(name) + " must have shape [tokens, heads, head_dim]."
        );
    }
}

AttentionShape validate_qkv(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values
){
    validate_tensor(queries, "Attention queries");
    validate_tensor(keys, "Attention keys");
    validate_tensor(values, "Attention values");

    if(queries.shape() != keys.shape() || queries.shape() != values.shape()){
        throw std::invalid_argument("Attention Q, K, and V shapes must match.");
    }

    AttentionShape shape{
        queries.shape()[0],
        queries.shape()[1],
        queries.shape()[2]
    };
    if(shape.tokens == 0 || shape.heads == 0 || shape.head_dim == 0){
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    return shape;
}

float dot_product(std::span<const float> left, std::span<const float> right){
    float result = 0.0f;
    for(size_t i = 0; i < left.size(); i++){
        result += left[i] * right[i];
    }
    return result;
}

template<typename KeyAt, typename ValueAt>
void compute_head_attention(
    std::span<const float> query,
    size_t context_tokens,
    float scale,
    KeyAt&& key_at,
    ValueAt&& value_at,
    std::span<float> output
){
    float maximum = -std::numeric_limits<float>::infinity();
    for(size_t token = 0; token < context_tokens; token++){
        maximum = std::max(maximum, dot_product(query, key_at(token)) * scale);
    }

    std::fill(output.begin(), output.end(), 0.0f);
    float denominator = 0.0f;
    for(size_t token = 0; token < context_tokens; token++){
        const float weight = std::exp(
            dot_product(query, key_at(token)) * scale - maximum
        );
        const std::span<const float> value = value_at(token);
        denominator += weight;

        for(size_t element = 0; element < output.size(); element++){
            output[element] += weight * value[element];
        }
    }

    for(float& element : output){
        element /= denominator;
    }
}

void validate_cache_append(const ContiguousKvCache& cache, const AttentionShape& shape){
    if(cache.num_heads() != shape.heads || cache.head_dim() != shape.head_dim){
        throw std::invalid_argument("Attention shape does not match the KV cache configuration.");
    }
    if(shape.tokens > cache.capacity_tokens() - cache.size_tokens()){
        throw std::length_error("KV cache capacity exceeded.");
    }
}

Tensor compute_causal_attention(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    const AttentionShape& shape
){
    Tensor output(queries.shape());
    const size_t row_size = shape.head_dim;
    const size_t token_size = shape.heads * row_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));

    for(size_t token = 0; token < shape.tokens; token++){
        for(size_t head = 0; head < shape.heads; head++){
            const size_t query_offset = token * token_size + head * row_size;
            const std::span<const float> query{
                queries.data() + query_offset,
                row_size
            };
            std::span<float> output_row{
                output.data() + query_offset,
                row_size
            };

            compute_head_attention(
                query,
                token + 1,
                scale,
                [&](size_t context_token){
                    const size_t offset = context_token * token_size + head * row_size;
                    return std::span<const float>{keys.data() + offset, row_size};
                },
                [&](size_t context_token){
                    const size_t offset = context_token * token_size + head * row_size;
                    return std::span<const float>{values.data() + offset, row_size};
                },
                output_row
            );
        }
    }

    return output;
}

}

Tensor causal_attention_reference(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values
){
    const AttentionShape shape = validate_qkv(queries, keys, values);
    return compute_causal_attention(queries, keys, values, shape);
}

Tensor attention_prefill(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    ContiguousKvCache& cache
){
    const AttentionShape shape = validate_qkv(queries, keys, values);
    if(cache.size_tokens() != 0){
        throw std::logic_error("Attention prefill requires an empty KV cache.");
    }
    validate_cache_append(cache, shape);

    Tensor output = compute_causal_attention(queries, keys, values, shape);
    cache.append(keys, values);
    return output;
}

Tensor attention_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    ContiguousKvCache& cache
){
    const AttentionShape shape = validate_qkv(query, key, value);
    if(shape.tokens != 1){
        throw std::invalid_argument("Attention decode accepts exactly one token.");
    }
    validate_cache_append(cache, shape);

    Tensor output(query.shape());
    const size_t row_size = shape.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));

    cache.append(key, value);
    for(size_t head = 0; head < shape.heads; head++){
        const size_t offset = head * row_size;
        compute_head_attention(
            {query.data() + offset, row_size},
            cache.size_tokens(),
            scale,
            [&](size_t token){ return cache.key(token, head); },
            [&](size_t token){ return cache.value(token, head); },
            {output.data() + offset, row_size}
        );
    }

    return output;
}

}

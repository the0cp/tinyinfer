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
    size_t query_heads;
    size_t kv_heads;
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

    if(keys.shape() != values.shape()){
        throw std::invalid_argument("Attention K and V shapes must match.");
    }

    AttentionShape shape{
        queries.shape()[0],
        queries.shape()[1],
        keys.shape()[1],
        queries.shape()[2]
    };
    if(
        shape.tokens == 0 || shape.query_heads == 0 ||
        shape.kv_heads == 0 || shape.head_dim == 0
    ){
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    if(queries.shape()[0] != keys.shape()[0]){
        throw std::invalid_argument("Attention Q, K, and V token counts must match.");
    }
    if(queries.shape()[2] != keys.shape()[2]){
        throw std::invalid_argument("Attention Q, K, and V head dimensions must match.");
    }
    if(shape.query_heads % shape.kv_heads != 0){
        throw std::invalid_argument(
            "Attention query heads must be divisible by KV heads."
        );
    }
    return shape;
}

size_t kv_head_for_query(size_t query_head, const AttentionShape& shape) noexcept{
    const size_t queries_per_kv_head = shape.query_heads / shape.kv_heads;
    return query_head / queries_per_kv_head;
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

template<typename Cache>
void validate_cache_append(const Cache& cache, const AttentionShape& shape){
    if(cache.num_heads() != shape.kv_heads || cache.head_dim() != shape.head_dim){
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
    const size_t query_token_size = shape.query_heads * row_size;
    const size_t kv_token_size = shape.kv_heads * row_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));

    for(size_t token = 0; token < shape.tokens; token++){
        for(size_t query_head = 0; query_head < shape.query_heads; query_head++){
            const size_t kv_head = kv_head_for_query(query_head, shape);
            const size_t query_offset =
                token * query_token_size + query_head * row_size;
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
                    const size_t offset =
                        context_token * kv_token_size + kv_head * row_size;
                    return std::span<const float>{keys.data() + offset, row_size};
                },
                [&](size_t context_token){
                    const size_t offset =
                        context_token * kv_token_size + kv_head * row_size;
                    return std::span<const float>{values.data() + offset, row_size};
                },
                output_row
            );
        }
    }

    return output;
}

template<typename Cache>
Tensor prefill_impl(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    Cache& cache
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

template<typename Cache>
Tensor decode_impl(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Cache& cache
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
    for(size_t query_head = 0; query_head < shape.query_heads; query_head++){
        const size_t kv_head = kv_head_for_query(query_head, shape);
        const size_t offset = query_head * row_size;
        compute_head_attention(
            {query.data() + offset, row_size},
            cache.size_tokens(),
            scale,
            [&](size_t token){ return cache.key(token, kv_head); },
            [&](size_t token){ return cache.value(token, kv_head); },
            {output.data() + offset, row_size}
        );
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
    return prefill_impl(queries, keys, values, cache);
}

Tensor attention_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    ContiguousKvCache& cache
){
    return decode_impl(query, key, value, cache);
}

Tensor attention_prefill(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    PagedKvCache& cache
){
    return prefill_impl(queries, keys, values, cache);
}

Tensor attention_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    PagedKvCache& cache
){
    return decode_impl(query, key, value, cache);
}

}

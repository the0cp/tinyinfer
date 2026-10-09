#include "kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace tinyinfer{

namespace{

KvCacheConfig validated_config(KvCacheConfig config){
    if(config.max_tokens == 0){
        throw std::invalid_argument("KV cache max_tokens must be positive.");
    }
    if(config.num_heads == 0){
        throw std::invalid_argument("KV cache num_heads must be positive.");
    }
    if(config.head_dim == 0){
        throw std::invalid_argument("KV cache head_dim must be positive.");
    }
    return config;
}

void validate_append_tensor(
    const Tensor& tensor,
    const KvCacheConfig& config,
    const char* name
){
    if(tensor.dtype() != DataType::Float32 || !tensor.is_contiguous()){
        throw std::invalid_argument(
            std::string("KV cache ") + name + " must be contiguous float32."
        );
    }
    if(tensor.dim() != 3){
        throw std::invalid_argument(
            std::string("KV cache ") + name + " must have shape [tokens, heads, head_dim]."
        );
    }
    if(tensor.shape()[0] == 0){
        throw std::invalid_argument("KV cache cannot append zero tokens.");
    }
    if(tensor.shape()[1] != config.num_heads || tensor.shape()[2] != config.head_dim){
        throw std::invalid_argument(
            std::string("KV cache ") + name + " shape does not match its configuration."
        );
    }
}

}

ContiguousKvCache::ContiguousKvCache(KvCacheConfig config)
    : config_(validated_config(config)),
      keys_(Shape{config_.max_tokens, config_.num_heads, config_.head_dim}),
      values_(Shape{config_.max_tokens, config_.num_heads, config_.head_dim}){}

void ContiguousKvCache::append(const Tensor& keys, const Tensor& values){
    validate_append_tensor(keys, config_, "keys");
    validate_append_tensor(values, config_, "values");

    if(keys.shape() != values.shape()){
        throw std::invalid_argument("KV cache key and value shapes must match.");
    }

    const size_t incoming_tokens = keys.shape()[0];
    if(incoming_tokens > config_.max_tokens - size_tokens_){
        throw std::length_error("KV cache capacity exceeded.");
    }

    const size_t destination_offset = size_tokens_ * elements_per_token();
    std::memcpy(
        keys_.data() + destination_offset,
        keys.data(),
        keys.size_bytes()
    );
    std::memcpy(
        values_.data() + destination_offset,
        values.data(),
        values.size_bytes()
    );
    size_tokens_ += incoming_tokens;
}

void ContiguousKvCache::clear() noexcept{
    size_tokens_ = 0;
}

size_t ContiguousKvCache::size_tokens() const noexcept{
    return size_tokens_;
}

size_t ContiguousKvCache::capacity_tokens() const noexcept{
    return config_.max_tokens;
}

size_t ContiguousKvCache::num_heads() const noexcept{
    return config_.num_heads;
}

size_t ContiguousKvCache::head_dim() const noexcept{
    return config_.head_dim;
}

size_t ContiguousKvCache::used_bytes() const noexcept{
    return 2 * size_tokens_ * elements_per_token() * sizeof(float);
}

size_t ContiguousKvCache::allocated_bytes() const noexcept{
    return keys_.size_bytes() + values_.size_bytes();
}

std::span<const float> ContiguousKvCache::key(size_t token, size_t head) const{
    return head_view(keys_, token, head);
}

std::span<const float> ContiguousKvCache::value(size_t token, size_t head) const{
    return head_view(values_, token, head);
}

size_t ContiguousKvCache::elements_per_token() const noexcept{
    return config_.num_heads * config_.head_dim;
}

std::span<const float> ContiguousKvCache::head_view(
    const Tensor& storage,
    size_t token,
    size_t head
) const{
    if(token >= size_tokens_){
        throw std::out_of_range("KV cache token index is out of range.");
    }
    if(head >= config_.num_heads){
        throw std::out_of_range("KV cache head index is out of range.");
    }

    const size_t offset = token * elements_per_token() + head * config_.head_dim;
    return {storage.data() + offset, config_.head_dim};
}

}

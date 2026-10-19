#include "paged_kv_cache.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tinyinfer{

namespace{

KvBlockPoolConfig validated_config(KvBlockPoolConfig config){
    if(config.num_blocks == 0){
        throw std::invalid_argument("KV block pool num_blocks must be positive.");
    }
    if(config.block_tokens == 0){
        throw std::invalid_argument("KV block pool block_tokens must be positive.");
    }
    if(config.num_heads == 0){
        throw std::invalid_argument("KV block pool num_heads must be positive.");
    }
    if(config.head_dim == 0){
        throw std::invalid_argument("KV block pool head_dim must be positive.");
    }
    return config;
}

void validate_append_tensor(
    const Tensor& tensor,
    const KvBlockPool& pool,
    const char* name
){
    if(tensor.dtype() != DataType::Float32 || !tensor.is_contiguous()){
        throw std::invalid_argument(
            std::string("Paged KV cache ") + name + " must be contiguous float32."
        );
    }
    if(tensor.dim() != 3){
        throw std::invalid_argument(
            std::string("Paged KV cache ") + name +
            " must have shape [tokens, heads, head_dim]."
        );
    }
    if(tensor.shape()[0] == 0){
        throw std::invalid_argument("Paged KV cache cannot append zero tokens.");
    }
    if(tensor.shape()[1] != pool.num_heads() || tensor.shape()[2] != pool.head_dim()){
        throw std::invalid_argument(
            std::string("Paged KV cache ") + name +
            " shape does not match its block pool."
        );
    }
}

}

KvBlockPool::KvBlockPool(KvBlockPoolConfig config)
    : config_(validated_config(config)),
      keys_(Shape{
          config_.num_blocks,
          config_.block_tokens,
          config_.num_heads,
          config_.head_dim
      }),
      values_(Shape{
          config_.num_blocks,
          config_.block_tokens,
          config_.num_heads,
          config_.head_dim
      }),
      ref_counts_(config_.num_blocks, 0){
    free_blocks_.reserve(config_.num_blocks);
    for(size_t block = config_.num_blocks; block-- > 0;){
        free_blocks_.push_back(block);
    }
}

size_t KvBlockPool::num_blocks() const noexcept{
    return config_.num_blocks;
}

size_t KvBlockPool::free_blocks() const noexcept{
    return free_blocks_.size();
}

size_t KvBlockPool::used_blocks() const noexcept{
    return num_blocks() - free_blocks();
}

size_t KvBlockPool::block_tokens() const noexcept{
    return config_.block_tokens;
}

size_t KvBlockPool::num_heads() const noexcept{
    return config_.num_heads;
}

size_t KvBlockPool::head_dim() const noexcept{
    return config_.head_dim;
}

size_t KvBlockPool::capacity_tokens() const noexcept{
    return config_.num_blocks * config_.block_tokens;
}

size_t KvBlockPool::block_bytes() const noexcept{
    return 2 * config_.block_tokens * config_.num_heads * config_.head_dim * sizeof(float);
}

size_t KvBlockPool::used_bytes() const noexcept{
    return used_blocks() * block_bytes();
}

size_t KvBlockPool::allocated_bytes() const noexcept{
    return keys_.size_bytes() + values_.size_bytes();
}

size_t KvBlockPool::block_ref_count(KvBlockId block) const{
    if(block >= num_blocks()){
        throw std::out_of_range("KV block index is out of range.");
    }
    return ref_counts_[block];
}

KvBlockId KvBlockPool::acquire(){
    if(free_blocks_.empty()){
        throw std::length_error("KV block pool is exhausted.");
    }

    const KvBlockId block = free_blocks_.back();
    free_blocks_.pop_back();
    assert(ref_counts_[block] == 0);
    ref_counts_[block] = 1;
    return block;
}

void KvBlockPool::retain(KvBlockId block) noexcept{
    assert(block < num_blocks());
    assert(ref_counts_[block] > 0);
    ref_counts_[block]++;
}

void KvBlockPool::release(KvBlockId block) noexcept{
    assert(block < num_blocks());
    assert(ref_counts_[block] > 0);
    ref_counts_[block]--;
    if(ref_counts_[block] == 0){
        free_blocks_.push_back(block);
    }
}

std::span<float> KvBlockPool::key_token(KvBlockId block, size_t block_offset){
    const size_t begin = token_begin(block, block_offset);
    return {keys_.data() + begin, config_.num_heads * config_.head_dim};
}

std::span<float> KvBlockPool::value_token(KvBlockId block, size_t block_offset){
    const size_t begin = token_begin(block, block_offset);
    return {values_.data() + begin, config_.num_heads * config_.head_dim};
}

std::span<const float> KvBlockPool::key_head(
    KvBlockId block,
    size_t block_offset,
    size_t head
) const{
    return head_view(keys_, block, block_offset, head);
}

std::span<const float> KvBlockPool::value_head(
    KvBlockId block,
    size_t block_offset,
    size_t head
) const{
    return head_view(values_, block, block_offset, head);
}

size_t KvBlockPool::token_begin(KvBlockId block, size_t block_offset) const noexcept{
    const size_t elements_per_token = config_.num_heads * config_.head_dim;
    return (block * config_.block_tokens + block_offset) * elements_per_token;
}

std::span<const float> KvBlockPool::head_view(
    const Tensor& storage,
    KvBlockId block,
    size_t block_offset,
    size_t head
) const{
    const size_t begin = token_begin(block, block_offset) + head * config_.head_dim;
    return {storage.data() + begin, config_.head_dim};
}

PagedKvCache::PagedKvCache(KvBlockPool& pool, size_t max_tokens)
    : pool_(pool), max_tokens_(max_tokens){
    if(max_tokens_ == 0){
        throw std::invalid_argument("Paged KV cache max_tokens must be positive.");
    }
    if(max_tokens_ > pool_.capacity_tokens()){
        throw std::invalid_argument("Paged KV cache max_tokens exceeds pool capacity.");
    }
    blocks_.reserve(required_blocks(max_tokens_));
}

PagedKvCache::~PagedKvCache(){
    clear();
}

void PagedKvCache::append(const Tensor& keys, const Tensor& values){
    validate_append_tensor(keys, pool_, "keys");
    validate_append_tensor(values, pool_, "values");

    if(keys.shape() != values.shape()){
        throw std::invalid_argument("Paged KV cache key and value shapes must match.");
    }

    const size_t incoming_tokens = keys.shape()[0];
    if(incoming_tokens > max_tokens_ - size_tokens_){
        throw std::length_error("Paged KV cache capacity exceeded.");
    }

    const size_t final_tokens = size_tokens_ + incoming_tokens;
    const size_t final_blocks = required_blocks(final_tokens);
    const size_t new_blocks = final_blocks - blocks_.size();
    if(new_blocks > pool_.free_blocks()){
        throw std::length_error("KV block pool is exhausted.");
    }

    for(size_t i = 0; i < new_blocks; i++){
        blocks_.push_back(pool_.acquire());
    }

    const size_t elements_per_token = num_heads() * head_dim();
    for(size_t i = 0; i < incoming_tokens; i++){
        const size_t token = size_tokens_ + i;
        const size_t logical_block = token / pool_.block_tokens();
        const size_t block_offset = token % pool_.block_tokens();
        const KvBlockId block = blocks_[logical_block];
        const size_t source_offset = i * elements_per_token;

        std::memcpy(
            pool_.key_token(block, block_offset).data(),
            keys.data() + source_offset,
            elements_per_token * sizeof(float)
        );
        std::memcpy(
            pool_.value_token(block, block_offset).data(),
            values.data() + source_offset,
            elements_per_token * sizeof(float)
        );
    }
    size_tokens_ = final_tokens;
}

void PagedKvCache::share_prefix_from(const PagedKvCache& source, size_t tokens){
    if(this == &source){
        throw std::invalid_argument("Paged KV cache cannot share a prefix from itself.");
    }
    if(&pool_ != &source.pool_){
        throw std::invalid_argument("Paged KV caches must use the same block pool.");
    }
    if(size_tokens_ != 0){
        throw std::logic_error("Paged KV cache must be empty before sharing a prefix.");
    }
    if(tokens == 0 || tokens % pool_.block_tokens() != 0){
        throw std::invalid_argument("Shared KV prefix must contain complete blocks.");
    }
    if(tokens > source.size_tokens_){
        throw std::length_error("Shared KV prefix exceeds the source cache size.");
    }
    if(tokens > max_tokens_){
        throw std::length_error("Shared KV prefix exceeds the target cache capacity.");
    }

    const size_t block_count = tokens / pool_.block_tokens();
    for(size_t i = 0; i < block_count; i++){
        const KvBlockId block = source.blocks_[i];
        pool_.retain(block);
        blocks_.push_back(block);
    }
    size_tokens_ = tokens;
}

void PagedKvCache::clear() noexcept{
    for(KvBlockId block : blocks_){
        pool_.release(block);
    }
    blocks_.clear();
    size_tokens_ = 0;
}

size_t PagedKvCache::size_tokens() const noexcept{
    return size_tokens_;
}

size_t PagedKvCache::capacity_tokens() const noexcept{
    return max_tokens_;
}

size_t PagedKvCache::num_heads() const noexcept{
    return pool_.num_heads();
}

size_t PagedKvCache::head_dim() const noexcept{
    return pool_.head_dim();
}

size_t PagedKvCache::used_blocks() const noexcept{
    return blocks_.size();
}

size_t PagedKvCache::used_bytes() const noexcept{
    return 2 * size_tokens_ * num_heads() * head_dim() * sizeof(float);
}

size_t PagedKvCache::mapped_bytes() const noexcept{
    return blocks_.size() * pool_.block_bytes();
}

std::span<const KvBlockId> PagedKvCache::block_table() const noexcept{
    return blocks_;
}

std::span<const float> PagedKvCache::key(size_t token, size_t head) const{
    if(head >= num_heads()){
        throw std::out_of_range("Paged KV cache head index is out of range.");
    }
    const KvBlockId block = block_for_token(token);
    return pool_.key_head(block, token % pool_.block_tokens(), head);
}

std::span<const float> PagedKvCache::value(size_t token, size_t head) const{
    if(head >= num_heads()){
        throw std::out_of_range("Paged KV cache head index is out of range.");
    }
    const KvBlockId block = block_for_token(token);
    return pool_.value_head(block, token % pool_.block_tokens(), head);
}

size_t PagedKvCache::required_blocks(size_t tokens) const noexcept{
    return tokens / pool_.block_tokens() + (tokens % pool_.block_tokens() != 0 ? 1 : 0);
}

KvBlockId PagedKvCache::block_for_token(size_t token) const{
    if(token >= size_tokens_){
        throw std::out_of_range("Paged KV cache token index is out of range.");
    }
    return blocks_[token / pool_.block_tokens()];
}

}

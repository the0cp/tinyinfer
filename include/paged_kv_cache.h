#pragma once

#include "tensor.h"

#include <cstddef>
#include <span>
#include <vector>

namespace tinyinfer{

using KvBlockId = size_t;

struct KvBlockPoolConfig{
    size_t num_blocks;
    size_t block_tokens;
    size_t num_heads;
    size_t head_dim;
};

class PagedKvCache;

class KvBlockPool{
public:
    explicit KvBlockPool(KvBlockPoolConfig config);

    KvBlockPool(const KvBlockPool&) = delete;
    KvBlockPool& operator=(const KvBlockPool&) = delete;
    KvBlockPool(KvBlockPool&&) = delete;
    KvBlockPool& operator=(KvBlockPool&&) = delete;

    size_t num_blocks() const noexcept;
    size_t free_blocks() const noexcept;
    size_t used_blocks() const noexcept;
    size_t block_tokens() const noexcept;
    size_t num_heads() const noexcept;
    size_t head_dim() const noexcept;
    size_t capacity_tokens() const noexcept;
    size_t block_bytes() const noexcept;
    size_t used_bytes() const noexcept;
    size_t allocated_bytes() const noexcept;
    size_t block_ref_count(KvBlockId block) const;

private:
    friend class PagedKvCache;

    KvBlockId acquire();
    void retain(KvBlockId block) noexcept;
    void release(KvBlockId block) noexcept;

    std::span<float> key_token(KvBlockId block, size_t block_offset);
    std::span<float> value_token(KvBlockId block, size_t block_offset);
    std::span<const float> key_head(
        KvBlockId block,
        size_t block_offset,
        size_t head
    ) const;
    std::span<const float> value_head(
        KvBlockId block,
        size_t block_offset,
        size_t head
    ) const;

    size_t token_begin(KvBlockId block, size_t block_offset) const noexcept;
    std::span<const float> head_view(
        const Tensor& storage,
        KvBlockId block,
        size_t block_offset,
        size_t head
    ) const;

    KvBlockPoolConfig config_;
    Tensor keys_;
    Tensor values_;
    std::vector<KvBlockId> free_blocks_;
    std::vector<size_t> ref_counts_;
};

class PagedKvCache{
public:
    PagedKvCache(KvBlockPool& pool, size_t max_tokens);
    ~PagedKvCache();

    PagedKvCache(const PagedKvCache&) = delete;
    PagedKvCache& operator=(const PagedKvCache&) = delete;
    PagedKvCache(PagedKvCache&&) = delete;
    PagedKvCache& operator=(PagedKvCache&&) = delete;

    void append(const Tensor& keys, const Tensor& values);
    void share_prefix_from(const PagedKvCache& source, size_t tokens);
    void clear() noexcept;

    size_t size_tokens() const noexcept;
    size_t capacity_tokens() const noexcept;
    size_t num_heads() const noexcept;
    size_t head_dim() const noexcept;
    size_t used_blocks() const noexcept;
    size_t used_bytes() const noexcept;
    size_t mapped_bytes() const noexcept;
    std::span<const KvBlockId> block_table() const noexcept;

    std::span<const float> key(size_t token, size_t head) const;
    std::span<const float> value(size_t token, size_t head) const;

private:
    size_t required_blocks(size_t tokens) const noexcept;
    KvBlockId block_for_token(size_t token) const;

    KvBlockPool& pool_;
    size_t max_tokens_;
    std::vector<KvBlockId> blocks_;
    size_t size_tokens_ = 0;
};

}

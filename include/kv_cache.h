#pragma once

#include "tensor.h"

#include <cstddef>
#include <span>

namespace tinyinfer{

struct KvCacheConfig{
    size_t max_tokens;
    size_t num_heads;
    size_t head_dim;
};

class ContiguousKvCache{
public:
    explicit ContiguousKvCache(KvCacheConfig config);

    ContiguousKvCache(const ContiguousKvCache&) = delete;
    ContiguousKvCache& operator=(const ContiguousKvCache&) = delete;
    ContiguousKvCache(ContiguousKvCache&&) noexcept = default;
    ContiguousKvCache& operator=(ContiguousKvCache&&) noexcept = default;

    void append(const Tensor& keys, const Tensor& values);
    void clear() noexcept;

    size_t size_tokens() const noexcept;
    size_t capacity_tokens() const noexcept;
    size_t num_heads() const noexcept;
    size_t head_dim() const noexcept;
    size_t used_bytes() const noexcept;
    size_t allocated_bytes() const noexcept;

    std::span<const float> key(size_t token, size_t head) const;
    std::span<const float> value(size_t token, size_t head) const;

private:
    size_t elements_per_token() const noexcept;
    std::span<const float> head_view(
        const Tensor& storage,
        size_t token,
        size_t head
    ) const;

    KvCacheConfig config_;
    Tensor keys_;
    Tensor values_;
    size_t size_tokens_ = 0;
};

}

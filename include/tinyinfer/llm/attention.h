#pragma once

#include <tinyinfer/llm/kv_cache.h>
#include <tinyinfer/llm/paged_kv_cache.h>
#include <tinyinfer/core/tensor.h>

namespace tinyinfer{

Tensor causal_attention_reference(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values
);

Tensor attention_prefill(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    ContiguousKvCache& cache
);

Tensor attention_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    ContiguousKvCache& cache
);

Tensor attention_prefill(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    PagedKvCache& cache
);

Tensor attention_decode(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    PagedKvCache& cache
);

}

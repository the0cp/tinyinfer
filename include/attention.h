#pragma once

#include "kv_cache.h"
#include "tensor.h"

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

}

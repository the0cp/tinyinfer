#pragma once

#include <tinyinfer/core/tensor.h>
#include <tinyinfer/core/thread_pool.h>

#include <cstdint>
#include <span>

namespace tinyinfer{

// *_out contract: contiguous float32 activations/output, exact output shape,
// and non-overlapping byte ranges (disjoint views of one Buffer are OK).
// Embedding, Linear, RMSNorm and matmul_bt also accept immutable
// bfloat16 weights while accumulating and producing float32. Validation finishes
// before output writes.
Tensor add(const Tensor& a, const Tensor& b);
void add_out(const Tensor& a, const Tensor& b, Tensor& out);
void relu_out(const Tensor& x, Tensor& out);
Tensor matmul_bt(const Tensor& a, const Tensor& bt, ThreadPool* pool = nullptr);
Tensor embedding(const Tensor& weight, std::span<const int64_t> token_ids);
Tensor linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    ThreadPool* pool = nullptr
);
void linear_out(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out,
    ThreadPool* pool = nullptr
);
void softmax_out(const Tensor& x, Tensor& out);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float epsilon);
Tensor silu_mul(const Tensor& gate, const Tensor& up);

}

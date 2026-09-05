#pragma once

#include "tensor.h"
#include "thread_pool.h"

namespace tinyinfer{

// *_out contract: contiguous float32 inputs/output, exact output shape, and
// non-overlapping input/output byte ranges (disjoint views of one Buffer are OK).
// Inputs are read-only; every output element is overwritten. Output storage is
// never replaced. Validation finishes before the first output write.
Tensor add(const Tensor& a, const Tensor& b);
void add_out(const Tensor& a, const Tensor& b, Tensor& out);
Tensor transpose_2d(const Tensor& x);
Tensor relu(const Tensor& x);
void relu_out(const Tensor& x, Tensor& out);
Tensor naive_matmul(const Tensor& a, const Tensor& b);
Tensor fast_matmul(const Tensor& a, const Tensor& b);
Tensor blocked_matmul(const Tensor& a, const Tensor& b, size_t block_size);
Tensor matmul_transposed_b(const Tensor& a, const Tensor& bt);
Tensor parallel_matmul(const Tensor& a, const Tensor& b, size_t num_thread);
Tensor threadpool_matmul(const Tensor& a, const Tensor& b, ThreadPool& pool, size_t num_tasks);
Tensor add_bias(const Tensor& x, const Tensor& bias);
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);
void linear_out(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out);
Tensor softmax(const Tensor& x);
void softmax_out(const Tensor& x, Tensor& out);

}

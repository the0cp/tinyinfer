#pragma once

#include "tensor.h"
#include "thread_pool.h"

namespace tinyinfer{

Tensor add(const Tensor& a, const Tensor& b);
Tensor transpose_2d(const Tensor& x);
Tensor relu(const Tensor& x);
Tensor naive_matmul(const Tensor& a, const Tensor& b);
Tensor fast_matmul(const Tensor& a, const Tensor& b);
Tensor blocked_matmul(const Tensor& a, const Tensor& b, size_t block_size);
Tensor matmul_transposed_b(const Tensor& a, const Tensor& bt);
Tensor parallel_matmul(const Tensor& a, const Tensor& b, size_t num_thread);
Tensor threadpool_matmul(const Tensor& a, const Tensor& b, ThreadPool& pool, size_t num_tasks);
Tensor add_bias(const Tensor& x, const Tensor& bias);
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor &bias);
Tensor softmax(const Tensor& x);

}
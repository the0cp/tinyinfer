#include "ops.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace tinyinfer{

Tensor add(const Tensor& a, const Tensor& b){
    if(a.shape() != b.shape()){
        throw std::runtime_error("add shape mismatch.");
    }

    Tensor out(a.shape());

    for(size_t i = 0; i < a.numel(); i++){
        out.data()[i] = a.data()[i] + b.data()[i];
    }

    return out;
}

Tensor transpose_2d(const Tensor& x){
    if(x.dim() != 2){
        throw std::runtime_error("transpose_2d expects a 2D tensor.");
    }

    const size_t rows = x.shape()[0];
    const size_t cols = x.shape()[1];

    Tensor out({cols, rows});

    const float* x_data = x.data();
    float* out_data = out.data();

    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            out_data[j * rows + i] = x_data[i * cols + j];
        }
    }

    return out;
}

Tensor relu(const Tensor& x){
    Tensor out(x.shape());

    for(size_t i = 0; i < x.numel(); i++){
        out.data()[i] = std::max(0.0f, x.data()[i]);
    }

    return out;
}

Tensor naive_matmul(const Tensor& a, const Tensor& b){
    if(a.dim() != 2 || b.dim() != 2){
        throw std::runtime_error("matmul only supports 2D tensors.");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];

    if(b.shape()[0] != k){
        throw std::runtime_error("matmul shape mismatch.");
    }

    const size_t n = b.shape()[1];

    Tensor out({m, n});

    for(size_t i = 0; i < m; i++){
        for(size_t j = 0; j < n; j++){
            float sum = 0.0f;

            for(size_t p = 0; p < k; p++){
                sum += a.at({i, p}) * b.at({p, j});
            }

            out.at({i, j}) = sum;
        }
    }

    return out;
}

Tensor fast_matmul(const Tensor& a, const Tensor& b){
    if(a.dim() != 2 || b.dim() != 2){
        throw std::runtime_error("naive_matmul only supports 2D tensors.");
    }

    size_t m = a.shape()[0];
    size_t k = a.shape()[1];

    if(b.shape()[0] != k){
        throw std::runtime_error("naive_matmul shape mismatch.");
    }

    size_t n = b.shape()[1];

    Tensor out({m, n});

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.data();

    for(size_t i = 0; i < m; i++){
        const float* a_row = a_data + i * k;
        float* out_row = out_data + i * n;
        for(size_t p = 0; p < k; p++){
            const float a_val = a_row[p];
            const float* b_row = b_data + p * n;
            for(size_t j = 0; j < n; j++){
                out_row[j] += a_val * b_row[j];
            }
        }
    }

    return out;
}

Tensor blocked_matmul(const Tensor& a, const Tensor& b, size_t block_size){
    if(block_size == 0){
        throw std::runtime_error("blocked_matmul requires a non-zero block size.");
    }
    
    if(a.dim() != 2 || b.dim() != 2){
        throw std::runtime_error("matmul only supports 2D tensors.");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];

    if(b.shape()[0] != k){
        throw std::runtime_error("matmul shape mismatch.");
    }

    const size_t n = b.shape()[1];

    Tensor out({m, n});

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.data();

    for(size_t ii = 0; ii < m; ii += block_size){
        for(size_t pp = 0; pp < k; pp += block_size){
            for(size_t jj = 0; jj < n; jj += block_size){
                const size_t i_end = std::min(ii + block_size, m);
                const size_t p_end = std::min(pp + block_size, k);
                const size_t j_end = std::min(jj + block_size, n);

                for(size_t i = ii; i < i_end; i++){
                    const float* a_row = a_data + i * k;
                    float* out_row = out_data + i * n;

                    for(size_t p = pp; p < p_end; p++){
                        const float a_val = a_row[p];
                        const float* b_row = b_data + p * n;

                        for(size_t j = jj; j < j_end; j++){
                            out_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }

    return out;
}

Tensor matmul_transposed_b(const Tensor& a, const Tensor& bt){
    if(a.dim() != 2 || bt.dim() != 2){
        throw std::runtime_error("matmul_transposed_b only supports 2D tensors.");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];

    const size_t n = bt.shape()[0];

    if(bt.shape()[1] != k){
        throw std::runtime_error("matmul_transposed_b shape mismatch.");
    }

    Tensor out({m, n});

    const float* a_data = a.data();
    const float* bt_data = bt.data();
    float* out_data = out.data();

    for(size_t i = 0; i < m; i++){
        const float* a_row = a_data + i * k;
        for(size_t j = 0; j < n; j++){
            const float* bt_row = bt_data + j * k;
            float sum = 0.0f;
            for(size_t p = 0; p < k; p++){
                sum += a_row[p] * bt_row[p];
            }

            out_data[i * n + j] = sum;
        }
    }

    return out;
}

Tensor parallel_matmul(const Tensor& a, const Tensor& b, size_t num_threads){
    if(a.dim() != 2 || b.dim() != 2){
        throw std::runtime_error("parallel_matmul only supports 2D tensors.");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];

    if(b.shape()[0] != k){
        throw std::runtime_error("parallel_matmul shape mismatch.");
    }

    const size_t n = b.shape()[1];

    if(m == 0){
        return Tensor({m, n});
    }

    if(num_threads == 0){
        num_threads = 1;
    }

    if(num_threads > m){
        num_threads = m;
    }

    Tensor out({m, n});

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.data();

    auto worker = [&](size_t row_begin, size_t row_end){
        for(size_t i = row_begin; i < row_end; i++){
            const float* a_row = a_data + i * k;
            float* out_row = out_data + i * n;

            for(size_t p = 0; p < k; p++){
                const float a_val = a_row[p];
                const float* b_row = b_data + p * n;

                for(size_t j = 0; j < n; j++){
                    out_row[j] += a_val * b_row[j];
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    const size_t rows_per_thread = (m + num_threads - 1) / num_threads;

    for(size_t t = 0; t < num_threads; t++){
        const size_t row_begin = t * rows_per_thread;
        const size_t row_end = std::min(row_begin + rows_per_thread, m);

        if(row_begin >= row_end){
            break;
        }

        threads.emplace_back(worker, row_begin, row_end);
    }

    for(auto& t : threads){
        t.join();
    }

    return out;
}

Tensor threadpool_matmul(const Tensor& a, const Tensor& b, ThreadPool& pool, size_t num_tasks){
    if(a.dim() != 2 || b.dim() != 2){
        throw std::runtime_error("threadpool_matmul only supports 2D tensors.");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];

    if(b.shape()[0] != k){
        throw std::runtime_error("threadpool_matmul shape mismatch.");
    }

    const size_t n = b.shape()[1];

    if(m == 0){
        return Tensor({m, n});
    }

    if(num_tasks == 0){
        num_tasks = pool.size();
    }

    if(num_tasks > m){
        num_tasks = m;
    }

    Tensor out({m, n});

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.data();

    const size_t rows_per_task = (m + num_tasks - 1) / num_tasks;

    for(size_t task_id = 0; task_id < num_tasks; task_id++){
        const size_t row_begin = task_id * rows_per_task;
        const size_t row_end = std::min(row_begin + rows_per_task, m);

        if(row_begin >= row_end){
            break;
        }

        pool.enqueue([=](){
            for(size_t i = row_begin; i < row_end; i++){
                const float* a_row = a_data + i * k;
                float* out_row = out_data + i * n;

                for(size_t p = 0; p < k; p++){
                    const float a_val = a_row[p];
                    const float* b_row = b_data + p * n;

                    for(size_t j = 0; j < n; j++){
                        out_row[j] += a_val * b_row[j];
                    }
                }
            }
        });
    }

    pool.wait();

    return out;
}

Tensor add_bias(const Tensor& x, const Tensor& bias){
    if(x.dim() != 2){
        throw std::runtime_error("add_bias expects x to be 2D.");
    }

    if(bias.dim() != 1){
        throw std::runtime_error("add_bias expects bias to be 1D.");
    }

    const size_t batch = x.shape()[0];
    const size_t features = x.shape()[1];

    if(bias.shape()[0] != features){
        throw std::runtime_error("bias shape mismatch.");
    }

    Tensor out(x.shape());

    const float* x_data = x.data();
    const float* bias_data = bias.data();
    float* out_data = out.data();

    for(size_t i = 0; i < batch; i++){
        for(size_t j = 0; j < features; j++){
            out_data[i * features + j] = x_data[i * features + j] + bias_data[j];
        }
    }

    return out;
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias){
    Tensor y = naive_matmul(x, weight);
    return add_bias(y, bias);
}

Tensor softmax(const Tensor& x){
    if(x.dim() != 2){
        throw std::runtime_error("softmax expects a 2D tensor.");
    }

    const size_t batch = x.shape()[0];
    const size_t features = x.shape()[1];

    Tensor out(x.shape());

    if(features == 0){
        throw std::runtime_error("softmax requires a non-empty feature dimension.");
    }

    const float* x_data = x.data();
    float* out_data = out.data();

    for(size_t i = 0; i < batch; i++){
        const float* x_row = x_data + i * features;
        float* out_row = out_data + i * features;

        float max_val = x_row[0];
        float sum = 0.0f;

        for(size_t j = 1; j < features; j++){
            if(x_row[j] > max_val){
                max_val = x_row[j];
            }
        }

        for(size_t j = 0; j < features; j++){
            out_row[j] = std::exp(x_row[j] - max_val);
            sum += out_row[j];
        }

        for(size_t j = 0; j < features; j++){
            out_row[j] /= sum;
        }
    }

    return out;
}

}
#include <tinyinfer/compute/ops.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tinyinfer{

namespace{

void require_output(const Tensor& out, const Shape& shape, const char* operation){
    if(out.shape() != shape){
        throw std::runtime_error(std::string(operation) + " output shape mismatch.");
    }
    if(out.dtype() != DataType::Float32 || !out.is_contiguous()){
        throw std::runtime_error(std::string(operation) + " requires contiguous float32 output.");
    }
}

void require_disjoint_storage(
    const Tensor& input,
    const Tensor& out,
    const char* operation
){
    const size_t input_bytes = input.size_bytes();
    const size_t output_bytes = out.size_bytes();
    if(input_bytes == 0 || output_bytes == 0){
        return;
    }

    const auto input_address = reinterpret_cast<std::uintptr_t>(input.buffer()->data()) +
        input.byte_offset();
    const auto output_address = reinterpret_cast<std::uintptr_t>(out.buffer()->data()) +
        out.byte_offset();
    // Subtraction avoids overflowing an end-address calculation. Checking real
    // ranges also covers partial overlap and separately wrapped external memory.
    const bool overlaps = input_address <= output_address
        ? output_address - input_address < input_bytes
        : input_address - output_address < output_bytes;
    if(overlaps){
        throw std::invalid_argument(std::string(operation) + " input/output ranges overlap.");
    }
}

void require_input(const Tensor& input, const char* operation){
    if(input.dtype() != DataType::Float32 || !input.is_contiguous()){
        throw std::runtime_error(std::string(operation) + " requires contiguous float32 input.");
    }
}

void require_disjoint_input(const Tensor& input, const Tensor& out, const char* operation){
    require_input(input, operation);
    require_disjoint_storage(input, out, operation);
}

void require_weight(const Tensor& weight, const char* operation){
    if(
        (weight.dtype() != DataType::Float32 &&
         weight.dtype() != DataType::BFloat16) ||
        !weight.is_contiguous()
    ){
        throw std::runtime_error(
            std::string(operation) + " requires contiguous float32 or bfloat16 weights."
        );
    }
}

void require_disjoint_weight(const Tensor& weight, const Tensor& out, const char* operation){
    require_weight(weight, operation);
    require_disjoint_storage(weight, out, operation);
}

float as_float(float value) noexcept{
    return value;
}

float as_float(BFloat16 value) noexcept{
    return value.to_float();
}

struct LinearShape{
    size_t rows;
    size_t in_features;
    size_t out_features;
};

LinearShape validate_linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias
){
    if(x.dim() != 2 || weight.dim() != 2){
        throw std::runtime_error("linear expects 2D input and weight tensors.");
    }
    if(bias != nullptr && bias->dim() != 1){
        throw std::runtime_error("linear expects a 1D bias tensor.");
    }

    const LinearShape shape{x.shape()[0], x.shape()[1], weight.shape()[1]};
    if(weight.shape()[0] != shape.in_features){
        throw std::runtime_error("linear input and weight shapes do not match.");
    }
    if(bias != nullptr && bias->shape()[0] != shape.out_features){
        throw std::runtime_error("linear weight and bias shapes do not match.");
    }

    require_input(x, "linear");
    require_weight(weight, "linear");
    if(bias != nullptr){
        require_weight(*bias, "linear");
    }
    return shape;
}

LinearShape validate_linear_out(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
){
    const LinearShape shape = validate_linear(x, weight, bias);
    require_output(out, Shape{shape.rows, shape.out_features}, "linear");
    require_disjoint_storage(x, out, "linear");
    require_disjoint_storage(weight, out, "linear");
    if(bias != nullptr){
        require_disjoint_storage(*bias, out, "linear");
    }
    return shape;
}

template<typename Weight>
void linear_kernel(
    const Tensor& x,
    const Weight* weights,
    const Tensor* bias,
    Tensor& out,
    const LinearShape& shape,
    size_t row_begin,
    size_t row_end,
    size_t col_begin,
    size_t col_end
){
    const float* float_bias = bias != nullptr && bias->dtype() == DataType::Float32
        ? bias->data() : nullptr;
    const BFloat16* bfloat16_bias = bias != nullptr && float_bias == nullptr
        ? bias->bfloat16_data() : nullptr;
    const float* input = x.data();
    float* output = out.data();
    for(size_t row = row_begin; row < row_end; row++){
        float* out_row = output + row * shape.out_features;
        if(float_bias != nullptr){
            std::copy_n(
                float_bias + col_begin,
                col_end - col_begin,
                out_row + col_begin
            );
        }else if(bfloat16_bias != nullptr){
            for(size_t col = col_begin; col < col_end; col++){
                out_row[col] = bfloat16_bias[col].to_float();
            }
        }else{
            std::fill(out_row + col_begin, out_row + col_end, 0.0f);
        }
        for(size_t inner = 0; inner < shape.in_features; inner++){
            const float x_value = input[row * shape.in_features + inner];
            const Weight* weight_row = weights + inner * shape.out_features;
            for(size_t col = col_begin; col < col_end; col++){
                out_row[col] += x_value * as_float(weight_row[col]);
            }
        }
    }
}

void run_linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out,
    const LinearShape& shape,
    size_t row_begin,
    size_t row_end,
    size_t col_begin,
    size_t col_end
){
    if(weight.dtype() == DataType::Float32){
        linear_kernel(
            x, weight.data(), bias, out, shape,
            row_begin, row_end, col_begin, col_end
        );
        return;
    }
    linear_kernel(
        x, weight.bfloat16_data(), bias, out, shape,
        row_begin, row_end, col_begin, col_end
    );
}

template<typename Weight>
void matmul_bt_kernel(
    const Tensor& a,
    const Weight* weights,
    Tensor& out,
    size_t inner_size,
    size_t col_begin,
    size_t col_end
){
    const size_t rows = a.shape()[0];
    const size_t cols = out.shape()[1];
    const float* input_data = a.data();
    float* output_data = out.data();
    for(size_t row = 0; row < rows; row++){
        const float* input = input_data + row * inner_size;
        float* output = output_data + row * cols;
        for(size_t col = col_begin; col < col_end; col++){
            const Weight* weight = weights + col * inner_size;
            float sum = 0.0f;
            for(size_t inner = 0; inner < inner_size; inner++){
                sum += input[inner] * as_float(weight[inner]);
            }
            output[col] = sum;
        }
    }
}

void validate_matmul_bt(const Tensor& a, const Tensor& bt){
    if(a.dim() != 2 || bt.dim() != 2){
        throw std::runtime_error("matmul_transposed_b only supports 2D tensors.");
    }
    if(bt.shape()[1] != a.shape()[1]){
        throw std::runtime_error("matmul_transposed_b shape mismatch.");
    }
    require_input(a, "matmul_transposed_b");
    require_weight(bt, "matmul_transposed_b");
}

Tensor matmul_bt(
    const Tensor& a,
    const Tensor& bt,
    ThreadPool* pool,
    size_t num_tasks
){
    validate_matmul_bt(a, bt);
    const size_t rows = a.shape()[0];
    const size_t inner_size = a.shape()[1];
    const size_t cols = bt.shape()[0];
    Tensor out({rows, cols});
    if(out.numel() == 0 || inner_size == 0){
        return out;
    }

    const auto run = [&](const auto* weights){
        const auto kernel = [&](size_t col_begin, size_t col_end){
            matmul_bt_kernel(a, weights, out, inner_size, col_begin, col_end);
        };
        if(pool != nullptr){
            pool->parallel_for(cols, num_tasks, kernel);
        }else{
            kernel(0, cols);
        }
    };
    if(bt.dtype() == DataType::Float32){
        run(bt.data());
    }else{
        run(bt.bfloat16_data());
    }
    return out;
}

void parallel_linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out,
    ThreadPool& pool,
    size_t num_tasks,
    const LinearShape& shape
){
    if(out.numel() == 0){
        return;
    }

    if(shape.rows > 1){
        pool.parallel_for(
            shape.rows,
            num_tasks,
            [&](size_t row_begin, size_t row_end){
                run_linear(
                    x, weight, bias, out, shape,
                    row_begin, row_end, 0, shape.out_features
                );
            }
        );
        return;
    }

    pool.parallel_for(
        shape.out_features,
        num_tasks,
        [&](size_t col_begin, size_t col_end){
            run_linear(
                x, weight, bias, out, shape,
                0, 1, col_begin, col_end
            );
        }
    );
}

}

void add_out(const Tensor& a, const Tensor& b, Tensor& out){
    if(a.shape() != b.shape()){
        throw std::runtime_error("add shape mismatch.");
    }
    require_output(out, a.shape(), "add");
    require_disjoint_input(a, out, "add");
    require_disjoint_input(b, out, "add");

    const size_t count = a.numel();
    const float* left = a.data();
    const float* right = b.data();
    float* destination = out.data();
    for(size_t i = 0; i < count; i++){
        destination[i] = left[i] + right[i];
    }
}

Tensor add(const Tensor& a, const Tensor& b){
    Tensor out(a.shape());
    add_out(a, b, out);

    return out;
}

Tensor transpose_2d(const Tensor& x){
    if(x.dim() != 2){
        throw std::runtime_error("transpose_2d expects a 2D tensor.");
    }

    const size_t rows = x.shape()[0];
    const size_t cols = x.shape()[1];
    Tensor out({cols, rows});

    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            out.data()[j * rows + i] = x.data()[i * cols + j];
        }
    }

    return out;
}

void relu_out(const Tensor& x, Tensor& out){
    require_output(out, x.shape(), "relu");
    require_disjoint_input(x, out, "relu");

    const size_t count = x.numel();
    const float* source = x.data();
    float* destination = out.data();
    for(size_t i = 0; i < count; i++){
        destination[i] = std::max(0.0f, source[i]);
    }
}

Tensor relu(const Tensor& x){
    Tensor out(x.shape());
    relu_out(x, out);

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
        throw std::runtime_error("fast_matmul only supports 2D tensors.");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];

    if(b.shape()[0] != k){
        throw std::runtime_error("fast_matmul shape mismatch.");
    }

    const size_t n = b.shape()[1];
    Tensor out({m, n});

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* out_data = out.data();

    for(size_t i = 0; i < m; i++){
        const float* a_row = a_data + i * k;
        float* out_row = out_data + i * n;

        for(size_t p = 0; p < k; p++){
            const float a_value = a_row[p];
            const float* b_row = b_data + p * n;

            for(size_t j = 0; j < n; j++){
                out_row[j] += a_value * b_row[j];
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

    for(size_t ii = 0; ii < m; ii += block_size){
        for(size_t pp = 0; pp < k; pp += block_size){
            for(size_t jj = 0; jj < n; jj += block_size){
                const size_t i_end = std::min(ii + block_size, m);
                const size_t p_end = std::min(pp + block_size, k);
                const size_t j_end = std::min(jj + block_size, n);

                for(size_t i = ii; i < i_end; i++){
                    const float* a_row = a.data() + i * k;
                    float* out_row = out.data() + i * n;

                    for(size_t p = pp; p < p_end; p++){
                        const float a_value = a_row[p];
                        const float* b_row = b.data() + p * n;

                        for(size_t j = jj; j < j_end; j++){
                            out_row[j] += a_value * b_row[j];
                        }
                    }
                }
            }
        }
    }

    return out;
}

Tensor matmul_transposed_b(const Tensor& a, const Tensor& bt){
    return matmul_bt(a, bt, nullptr, 1);
}

Tensor threadpool_matmul_transposed_b(
    const Tensor& a,
    const Tensor& bt,
    ThreadPool& pool,
    size_t num_tasks
){
    return matmul_bt(a, bt, &pool, num_tasks);
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

    num_threads = std::max<size_t>(1, std::min(num_threads, m));
    Tensor out({m, n});
    const size_t rows_per_thread = (m + num_threads - 1) / num_threads;

    auto worker = [&](size_t row_begin, size_t row_end){
        for(size_t i = row_begin; i < row_end; i++){
            const float* a_row = a.data() + i * k;
            float* out_row = out.data() + i * n;

            for(size_t p = 0; p < k; p++){
                const float a_value = a_row[p];
                const float* b_row = b.data() + p * n;

                for(size_t j = 0; j < n; j++){
                    out_row[j] += a_value * b_row[j];
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(size_t t = 0; t < num_threads; t++){
        const size_t row_begin = t * rows_per_thread;
        const size_t row_end = std::min(row_begin + rows_per_thread, m);

        if(row_begin < row_end){
            threads.emplace_back(worker, row_begin, row_end);
        }
    }

    for(auto& thread : threads){
        thread.join();
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

    Tensor out({m, n});
    pool.parallel_for(m, num_tasks, [&](size_t row_begin, size_t row_end){
        for(size_t i = row_begin; i < row_end; i++){
            const float* a_row = a.data() + i * k;
            float* out_row = out.data() + i * n;

            for(size_t p = 0; p < k; p++){
                const float a_value = a_row[p];
                const float* b_row = b.data() + p * n;

                for(size_t j = 0; j < n; j++){
                    out_row[j] += a_value * b_row[j];
                }
            }
        }
    });
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

    for(size_t i = 0; i < batch; i++){
        for(size_t j = 0; j < features; j++){
            out.data()[i * features + j] = x.data()[i * features + j] + bias.data()[j];
        }
    }

    return out;
}

void embedding_out(
    const Tensor& weight,
    std::span<const int64_t> token_ids,
    Tensor& out
){
    if(weight.dim() != 2){
        throw std::runtime_error("embedding expects a 2D weight tensor.");
    }
    
    const size_t vocabulary_size = weight.shape()[0];
    const size_t hidden_size = weight.shape()[1];

    if(vocabulary_size == 0 || hidden_size == 0){
        throw std::runtime_error("embedding dimensions must be positive.");
    }

    for(int64_t token_id : token_ids){
        if(token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_size){
            throw std::out_of_range("embedding token id is out of range.");
        }
    }

    require_output(out, {token_ids.size(), hidden_size}, "embedding");
    require_disjoint_weight(weight, out, "embedding");

    if(weight.dtype() == DataType::Float32){
        for(size_t token = 0; token < token_ids.size(); token++){
            const size_t row = static_cast<size_t>(token_ids[token]);
            std::copy_n(
                weight.data() + row * hidden_size,
                hidden_size,
                out.data() + token * hidden_size
            );
        }
    }else{
        const BFloat16* source = weight.bfloat16_data();
        for(size_t token = 0; token < token_ids.size(); token++){
            const size_t row = static_cast<size_t>(token_ids[token]);
            float* destination = out.data() + token * hidden_size;
            for(size_t feature = 0; feature < hidden_size; feature++){
                destination[feature] = source[row * hidden_size + feature].to_float();
            }
        }
    }
}

Tensor embedding(const Tensor& weight, std::span<const int64_t> token_ids){
    if(weight.dim() != 2){
        throw std::runtime_error("embedding expects a 2D weight tensor.");
    }

    Tensor out({token_ids.size(), weight.shape()[1]});
    embedding_out(weight, token_ids, out);
    return out;
}

void linear_out(const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out){
    const LinearShape shape = validate_linear_out(x, weight, &bias, out);
    if(out.numel() == 0){
        return;
    }
    run_linear(
        x, weight, &bias, out, shape,
        0, shape.rows, 0, shape.out_features
    );
}

void linear_out(const Tensor& x, const Tensor& weight, Tensor& out){
    const LinearShape shape = validate_linear_out(x, weight, nullptr, out);
    if(out.numel() == 0){
        return;
    }
    run_linear(
        x, weight, nullptr, out, shape,
        0, shape.rows, 0, shape.out_features
    );
}

void threadpool_linear_out(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& out,
    ThreadPool& pool,
    size_t num_tasks
){
    const LinearShape shape = validate_linear_out(x, weight, &bias, out);
    parallel_linear(x, weight, &bias, out, pool, num_tasks, shape);
}

void threadpool_linear_out(
    const Tensor& x,
    const Tensor& weight,
    Tensor& out,
    ThreadPool& pool,
    size_t num_tasks
){
    const LinearShape shape = validate_linear_out(x, weight, nullptr, out);
    parallel_linear(x, weight, nullptr, out, pool, num_tasks, shape);
}

Tensor linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    ThreadPool* pool
){
    const LinearShape shape = validate_linear(x, weight, bias);
    Tensor out({shape.rows, shape.out_features});
    if(pool != nullptr){
        parallel_linear(x, weight, bias, out, *pool, pool->size(), shape);
    }else if(out.numel() != 0){
        run_linear(
            x, weight, bias, out, shape,
            0, shape.rows, 0, shape.out_features
        );
    }
    return out;
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias){
    return linear(x, weight, &bias, nullptr);
}

Tensor linear(const Tensor& x, const Tensor& weight){
    return linear(x, weight, nullptr, nullptr);
}

void softmax_out(const Tensor& x, Tensor& out){
    if(x.dim() != 2){
        throw std::runtime_error("softmax expects a 2D tensor.");
    }

    const size_t batch = x.shape()[0];
    const size_t features = x.shape()[1];

    if(features == 0){
        throw std::runtime_error("softmax requires a non-empty feature dimension.");
    }
    require_output(out, x.shape(), "softmax");
    require_disjoint_input(x, out, "softmax");

    const float* source = x.data();
    float* destination = out.data();

    for(size_t i = 0; i < batch; i++){
        const float* x_row = source + i * features;
        float* out_row = destination + i * features;
        float max_value = x_row[0];

        for(size_t j = 1; j < features; j++){
            max_value = std::max(max_value, x_row[j]);
        }

        float sum = 0.0f;

        for(size_t j = 0; j < features; j++){
            out_row[j] = std::exp(x_row[j] - max_value);
            sum += out_row[j];
        }

        for(size_t j = 0; j < features; j++){
            out_row[j] /= sum;
        }
    }
}

Tensor softmax(const Tensor& x){
    Tensor out(x.shape());
    softmax_out(x, out);

    return out;
}

void rms_norm_out(
    const Tensor& x,
    const Tensor& weight,
    float epsilon,
    Tensor& out
){
    if(x.dim() == 0 || weight.dim() != 1){
        throw std::runtime_error("rms_norm expects a tensor and a 1D weight.");
    }

    const size_t features = x.shape().back();

    if(features == 0 || weight.shape()[0] != features){
        throw std::runtime_error("rms_norm weight shape mismatch.");
    }

    if(!std::isfinite(epsilon) || epsilon <= 0.0f){
        throw std::runtime_error("rms_norm epsilon must be finite and positive.");
    }

    require_output(out, x.shape(), "rms_norm");
    require_disjoint_input(x, out, "rms_norm");
    require_disjoint_weight(weight, out, "rms_norm");

    const size_t rows = x.numel() / features;

    for(size_t row = 0; row < rows; row++){
        const float* source = x.data() + row * features;
        float* destination = out.data() + row * features;
        float square_sum = 0.0f;

        for(size_t feature = 0; feature < features; feature++){
            square_sum += source[feature] * source[feature];
        }

        const float scale = 1.0f / std::sqrt(
            square_sum / static_cast<float>(features) + epsilon
        );

        if(weight.dtype() == DataType::Float32){
            for(size_t feature = 0; feature < features; feature++){
                destination[feature] = source[feature] * scale * weight.data()[feature];
            }
        }else{
            const BFloat16* norm_weight = weight.bfloat16_data();
            for(size_t feature = 0; feature < features; feature++){
                destination[feature] =
                    source[feature] * scale * norm_weight[feature].to_float();
            }
        }
    }
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float epsilon){
    Tensor out(x.shape());
    rms_norm_out(x, weight, epsilon, out);
    return out;
}

void silu_mul_out(const Tensor& gate, const Tensor& up, Tensor& out){
    if(gate.shape() != up.shape()){
        throw std::runtime_error("silu_mul input shape mismatch.");
    }

    require_output(out, gate.shape(), "silu_mul");
    require_disjoint_input(gate, out, "silu_mul");
    require_disjoint_input(up, out, "silu_mul");

    for(size_t i = 0; i < gate.numel(); i++){
        const float value = gate.data()[i];
        out.data()[i] = value / (1.0f + std::exp(-value)) * up.data()[i];
    }
}

Tensor silu_mul(const Tensor& gate, const Tensor& up){
    Tensor out(gate.shape());
    silu_mul_out(gate, up, out);
    return out;
}

}

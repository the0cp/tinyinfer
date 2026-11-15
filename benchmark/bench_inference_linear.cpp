#include <tinyinfer/compute/ops.h>
#include <tinyinfer/core/buffer.h>
#include <tinyinfer/core/data_type.h>
#include <tinyinfer/core/thread_pool.h>
#include <tinyinfer/runtime/threading_policy.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

using namespace tinyinfer;

static Tensor bf16_tensor(Shape shape, float value){
    auto buffer = make_cpu_buffer(tensor_bytes(shape, DataType::BFloat16));
    auto* data = static_cast<BFloat16*>(buffer->data());
    std::fill_n(data, tensor_numel(shape), BFloat16::from_float(value));
    return Tensor(buffer, 0, std::move(shape), DataType::BFloat16);
}

template<typename Function>
static double median_ms(Function&& function){
    function();
    std::vector<double> samples;
    for(size_t i = 0; i < 5; i++){
        const auto start = std::chrono::steady_clock::now();
        function();
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start
        ).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void bench_linear(
    std::string_view name,
    const Tensor& input,
    const Tensor& weight,
    ThreadPool& pool
){
    Tensor serial_output({input.shape()[0], weight.shape()[1]});
    Tensor threaded_output({input.shape()[0], weight.shape()[1]});
    const double serial_ms = median_ms([&](){
        linear_out(input, weight, nullptr, serial_output);
    });
    const double threaded_ms = median_ms([&](){
        linear_out(input, weight, nullptr, threaded_output, &pool);
    });
    const bool threaded = prefer_threaded_linear(
        input.shape()[0],
        input.shape()[1],
        weight.shape()[1],
        pool.size()
    );
    std::cout << name
              << " serial_ms=" << serial_ms
              << " threaded_ms=" << threaded_ms
              << " speedup=" << serial_ms / threaded_ms
              << " selected=" << (threaded ? "threaded" : "serial")
              << " check=" << threaded_output.data()[0] << "\n";
}

int main(){
    const size_t cpu_threads = std::max<size_t>(
        1,
        std::thread::hardware_concurrency()
    );
    ThreadPool pool(cpu_threads);
    Tensor kv_weight = bf16_tensor({896, 128}, 0.001f);
    Tensor hidden_weight = bf16_tensor({896, 896}, 0.001f);
    Tensor mlp_weight = bf16_tensor({896, 4864}, 0.001f);
    Tensor prefill_input({32, 896});
    Tensor decode_input({1, 896});
    prefill_input.fill(0.5f);
    decode_input.fill(0.5f);

    std::cout << std::fixed << std::setprecision(3)
              << "cpu_threads=" << cpu_threads << "\n";
    bench_linear("prefill_32x896_to_4864", prefill_input, mlp_weight, pool);
    bench_linear("decode_1x896_to_128", decode_input, kv_weight, pool);
    bench_linear("decode_1x896_to_896", decode_input, hidden_weight, pool);
    bench_linear("decode_1x896_to_4864", decode_input, mlp_weight, pool);
    return 0;
}

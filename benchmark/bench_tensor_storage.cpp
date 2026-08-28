#include "tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace{

using namespace tinyinfer;
using Clock = std::chrono::steady_clock;

template<typename Function>
double ns_per_iteration(size_t iterations, Function&& function){
    const auto start = Clock::now();

    for(size_t i = 0; i < iterations; i++){
        function(i);
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    return static_cast<double>(elapsed.count()) / static_cast<double>(iterations);
}

struct LatencySummary{
    double minimum_ns = 0.0;
    double median_ns = 0.0;
    double p95_ns = 0.0;
    double maximum_ns = 0.0;
};

template<typename Function>
LatencySummary measure_batches(size_t batches, size_t iterations, Function&& function){
    std::vector<double> samples;
    samples.reserve(batches);

    for(size_t batch = 0; batch < batches; batch++){
        samples.push_back(ns_per_iteration(iterations, function));
    }

    std::sort(samples.begin(), samples.end());
    const size_t median_index = samples.size() / 2;
    const size_t p95_index = std::min(
        samples.size() - 1,
        static_cast<size_t>(std::ceil(static_cast<double>(samples.size()) * 0.95)) - 1
    );

    return LatencySummary{
        samples.front(),
        samples[median_index],
        samples[p95_index],
        samples.back()
    };
}

}

int main(){
    using namespace tinyinfer;

    constexpr size_t warmup_iterations = 1000;
    constexpr size_t measured_batches = 25;
    constexpr size_t owning_iterations_per_batch = 200;
    constexpr size_t view_iterations_per_batch = 20000;
    size_t checksum = 0;

    Tensor base({64, 64});

    for(size_t i = 0; i < warmup_iterations; i++){
        Tensor view = base.reshape({32, 128});
        checksum += view.shape()[0];
    }

    const LatencySummary owning = measure_batches(
        measured_batches,
        owning_iterations_per_batch,
        [&](size_t i){
        Tensor tensor({64, 64});
        tensor.data()[0] = static_cast<float>(i);
        checksum += tensor.shape()[0];
    });

    const Buffer* base_storage = base.buffer().get();
    const LatencySummary reshape = measure_batches(
        measured_batches,
        view_iterations_per_batch,
        [&](size_t){
        Tensor view = base.reshape({32, 128});

        if(view.buffer().get() != base_storage){
            throw std::runtime_error("reshape allocated a different data buffer");
        }

        checksum += view.shape()[0];
    });

    std::cout << "build="
#ifdef NDEBUG
              << "release"
#else
              << "debug"
#endif
              << ", shape=[64,64], dtype=float32"
              << "\nwarmup_iterations=" << warmup_iterations
              << ", measured_batches=" << measured_batches
              << "\nowning_iterations_per_batch=" << owning_iterations_per_batch
              << ", reshape_iterations_per_batch=" << view_iterations_per_batch
              << std::fixed << std::setprecision(2)
              << "\nowning_min_ns=" << owning.minimum_ns
              << ", owning_median_ns=" << owning.median_ns
              << ", owning_p95_ns=" << owning.p95_ns
              << ", owning_max_ns=" << owning.maximum_ns
              << "\nreshape_min_ns=" << reshape.minimum_ns
              << ", reshape_median_ns=" << reshape.median_ns
              << ", reshape_p95_ns=" << reshape.p95_ns
              << ", reshape_max_ns=" << reshape.maximum_ns
              << "\nreshape_reuses_data_buffer=yes"
              << "\nchecksum=" << checksum << "\n";

    return checksum == 0 ? 1 : 0;
}

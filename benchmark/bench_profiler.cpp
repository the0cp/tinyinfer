#include "inference_session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace{

using namespace tinyinfer;
using Clock = std::chrono::steady_clock;

Graph make_graph(){
    Graph graph;
    graph.add_node("left", OpType::ReLU, {"input"}, "left_out");
    graph.add_node("right", OpType::ReLU, {"input"}, "right_out");
    graph.add_node("sum", OpType::Add, {"left_out", "right_out"}, "output");
    return graph;
}

struct LatencySummary{
    size_t count = 0;
    double minimum_us = 0.0;
    double median_us = 0.0;
    double p95_us = 0.0;
    double maximum_us = 0.0;
};

LatencySummary summarize(std::vector<uint64_t> samples_ns){
    if(samples_ns.empty()){
        throw std::invalid_argument("Latency summary requires samples.");
    }

    std::sort(samples_ns.begin(), samples_ns.end());
    const size_t median_index = samples_ns.size() / 2;
    const size_t p95_index = std::min(
        samples_ns.size() - 1,
        static_cast<size_t>(std::ceil(static_cast<double>(samples_ns.size()) * 0.95)) - 1
    );

    return LatencySummary{
        samples_ns.size(),
        static_cast<double>(samples_ns.front()) / 1000.0,
        static_cast<double>(samples_ns[median_index]) / 1000.0,
        static_cast<double>(samples_ns[p95_index]) / 1000.0,
        static_cast<double>(samples_ns.back()) / 1000.0
    };
}

template<typename Run>
LatencySummary measure(size_t iterations, Run&& run){
    std::vector<uint64_t> samples;
    samples.reserve(iterations);

    for(size_t i = 0; i < iterations; i++){
        const auto start = Clock::now();
        run();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start
        );
        samples.push_back(static_cast<uint64_t>(elapsed.count()));
    }

    return summarize(std::move(samples));
}

}

int main(){
    using namespace tinyinfer;

    InferenceSession session;
    session.load(make_graph(), "input", {1, 1024}, "output");
    session.initialize();

    Tensor input({1, 1024});
    input.fill(1.0f);

    constexpr size_t warmup_iterations = 100;
    constexpr size_t measured_iterations = 1000;
    float checksum = 0.0f;

    for(size_t i = 0; i < warmup_iterations; i++){
        checksum += session.run(input).data()[0];
    }

    const LatencySummary normal = measure(measured_iterations, [&](){
        checksum += session.run(input).data()[0];
    });

    RunProfile last_profile;
    const LatencySummary profiled = measure(measured_iterations, [&](){
        checksum += session.run(input, last_profile).data()[0];
    });

    const double overhead_percent = normal.median_us == 0.0
        ? 0.0
        : ((profiled.median_us / normal.median_us) - 1.0) * 100.0;

    std::cout << "build="
#ifdef NDEBUG
              << "release"
#else
              << "debug"
#endif
              << ", mode=sequential, threads=1, graph_nodes=3, input_shape=[1,1024]"
              << "\nwarmup_runs=" << warmup_iterations
              << ", measured_runs_per_case=" << measured_iterations
              << std::fixed << std::setprecision(3)
              << "\nnormal_min_us=" << normal.minimum_us
              << ", normal_median_us=" << normal.median_us
              << ", normal_p95_us=" << normal.p95_us
              << ", normal_max_us=" << normal.maximum_us
              << "\nprofiled_min_us=" << profiled.minimum_us
              << ", profiled_median_us=" << profiled.median_us
              << ", profiled_p95_us=" << profiled.p95_us
              << ", profiled_max_us=" << profiled.maximum_us
              << "\nmedian_overhead_percent=" << overhead_percent
              << "\nprofile_peak_live_bytes=" << last_profile.peak_live_bytes
              << ", profile_allocations=" << last_profile.owned_value_allocations
              << "\nchecksum=" << checksum << "\n";

    const float expected = static_cast<float>((warmup_iterations + measured_iterations * 2) * 2);
    return std::fabs(checksum - expected) < 1e-5f ? 0 : 1;
}

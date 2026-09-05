#include "inference_session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace{

using namespace tinyinfer;
using Clock = std::chrono::steady_clock;

Graph make_relu_chain(size_t node_count){
    Graph graph;
    std::string input = "input";

    for(size_t i = 0; i < node_count; i++){
        const std::string output = "v" + std::to_string(i);
        graph.add_node("relu_" + std::to_string(i), OpType::ReLU, {input}, output);
        input = output;
    }

    return graph;
}

std::unique_ptr<InferenceSession> make_session(size_t nodes, size_t elements, bool planned){
    SessionOptions options;
    options.enable_memory_planning = planned;

    auto session = std::make_unique<InferenceSession>(options);
    session->load(
        make_relu_chain(nodes),
        "input",
        {1, elements},
        "v" + std::to_string(nodes - 1)
    );
    session->initialize();
    return session;
}

// Distribution of batch averages, not individual-request tail latency.
struct Latency{
    double median_us;
    double p95_us;
};

Latency measure(InferenceSession& session, const Tensor& input, size_t batches, size_t iterations){
    std::vector<double> samples;
    samples.reserve(batches);
    float checksum = 0.0f;

    for(size_t batch = 0; batch < batches; batch++){
        const auto start = Clock::now();
        for(size_t i = 0; i < iterations; i++){
            Tensor output = session.run(input);
            checksum += output.data()[i % output.numel()];
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
        samples.push_back(
            static_cast<double>(elapsed.count()) / static_cast<double>(iterations) / 1000.0
        );
    }

    if(!std::isfinite(checksum)){
        throw std::runtime_error("benchmark checksum is not finite");
    }

    std::sort(samples.begin(), samples.end());
    const size_t p95 = std::min(
        samples.size() - 1,
        static_cast<size_t>(std::ceil(static_cast<double>(samples.size()) * 0.95)) - 1
    );
    return Latency{samples[samples.size() / 2], samples[p95]};
}

}

int main(){
    constexpr size_t nodes = 12;
    constexpr size_t elements = 4096;
    constexpr size_t warmups = 20;
    constexpr size_t batches = 21;
    constexpr size_t iterations = 50;

    std::unique_ptr<InferenceSession> unplanned = make_session(nodes, elements, false);
    std::unique_ptr<InferenceSession> planned = make_session(nodes, elements, true);
    Tensor input({1, elements});
    input.fill(1.0f);

    for(size_t i = 0; i < warmups; i++){
        (void)unplanned->run(input);
        (void)planned->run(input);
    }

    RunProfile unplanned_profile;
    RunProfile planned_profile;
    (void)unplanned->run(input, unplanned_profile);
    (void)planned->run(input, planned_profile);

    const Latency unplanned_latency = measure(*unplanned, input, batches, iterations);
    const Latency planned_latency = measure(*planned, input, batches, iterations);
    const MemoryPlan& memory = planned->session_state().memory_plan();

    std::cout << "build="
#ifdef NDEBUG
              << "release"
#else
              << "debug"
#endif
              << ", nodes=" << nodes << ", tensor_elements=" << elements
              << "\nmeasurement=batches:" << batches << ", iterations_per_batch:" << iterations
              << "\nunplanned_managed_buffers=" << unplanned_profile.managed_buffer_allocations
              << ", unplanned_allocated_bytes=" << unplanned_profile.managed_allocated_bytes
              << ", unplanned_logical_peak_bytes=" << unplanned_profile.peak_live_bytes
              << "\nplanned_managed_buffers=" << planned_profile.managed_buffer_allocations
              << ", planned_arena_bytes=" << memory.arena_bytes()
              << ", planned_value_reuses=" << memory.reuse_count()
              << std::fixed << std::setprecision(3)
              << "\nunplanned_batch_mean_median_us=" << unplanned_latency.median_us
              << ", unplanned_batch_mean_p95_us=" << unplanned_latency.p95_us
              << "\nplanned_batch_mean_median_us=" << planned_latency.median_us
              << ", planned_batch_mean_p95_us=" << planned_latency.p95_us
              << "\n";

    return 0;
}

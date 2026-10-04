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

struct LinearShape{
    const char* name;
    size_t batch;
    size_t input_features;
    size_t output_features;
};

Graph make_graph(size_t input_features, size_t output_features){
    std::vector<float> weights(input_features * output_features);
    std::vector<float> bias(output_features);
    for(size_t index = 0; index < weights.size(); index++){
        weights[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.005f;
    }

    Graph graph;
    graph.set_tensor("weight", Tensor({input_features, output_features}, weights));
    graph.set_tensor("bias", Tensor({output_features}, bias));
    graph.add_node("linear", OpType::Linear, {"input", "weight", "bias"}, "output");
    return graph;
}

std::unique_ptr<InferenceSession> make_session(const LinearShape& shape, size_t threads){
    SessionOptions options;
    options.intra_op_threads = threads;
    auto session = std::make_unique<InferenceSession>(options);
    session->load(
        make_graph(shape.input_features, shape.output_features),
        "input",
        {shape.batch, shape.input_features},
        "output"
    );
    session->initialize();
    return session;
}

struct Latency{
    double median_us;
    double p95_us;
    float checksum;
};

Latency measure(InferenceSession& session, const Tensor& input){
    constexpr size_t warmups = 3;
    constexpr size_t batches = 9;
    constexpr size_t iterations = 3;
    for(size_t index = 0; index < warmups; index++){
        (void)session.run(input);
    }

    std::vector<double> samples;
    float checksum = 0.0f;
    for(size_t batch = 0; batch < batches; batch++){
        const auto start = Clock::now();
        for(size_t iteration = 0; iteration < iterations; iteration++){
            Tensor output = session.run(input);
            checksum += output.data()[(batch + iteration) % output.numel()];
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start
        );
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
    return Latency{samples[samples.size() / 2], samples[p95], checksum};
}

}

int main(){
    const std::vector<LinearShape> shapes{
        {"tiny", 1, 32, 32},
        {"small_batch", 4, 32, 32},
        {"square", 64, 128, 128},
        {"tall_batch", 256, 64, 64},
        {"wide", 8, 512, 512}
    };

    std::cout << "build="
#ifdef NDEBUG
              << "release\n";
#else
              << "debug\n";
#endif
    std::cout << std::fixed << std::setprecision(3);

    for(const LinearShape& shape : shapes){
        Tensor input({shape.batch, shape.input_features});
        for(size_t index = 0; index < input.numel(); index++){
            input.data()[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.02f;
        }

        for(size_t threads : {size_t{1}, size_t{4}}){
            std::unique_ptr<InferenceSession> session = make_session(shape, threads);
            const KernelSelectionRecord& selection =
                session->plan().nodes().front().kernel_selection;
            const Latency latency = measure(*session, input);
            std::cout << "shape=" << shape.name
                      << " [" << shape.batch << "," << shape.input_features << "]x["
                      << shape.input_features << "," << shape.output_features << "]"
                      << ", threads=" << threads
                      << ", selected=" << selection.kernel_name
                      << ", cost=" << selection.estimated_cost
                      << ", median_us=" << latency.median_us
                      << ", p95_us=" << latency.p95_us
                      << ", checksum=" << latency.checksum << "\n";
        }
    }

    return 0;
}

#include "inference_session.h"

#include <chrono>
#include <iostream>

using namespace tinyinfer;

static Graph make_branch_graph(size_t branches){
    Graph graph;

    for(size_t i = 0; i < branches; i++){
        graph.add_node(
            "relu_" + std::to_string(i),
            OpType::ReLU,
            {"input"},
            "branch_" + std::to_string(i)
        );
    }

    std::string accumulated = "branch_0";

    for(size_t i = 1; i < branches; i++){
        const std::string output = "sum_" + std::to_string(i);
        graph.add_node(
            "add_" + std::to_string(i),
            OpType::Add,
            {accumulated, "branch_" + std::to_string(i)},
            output
        );
        accumulated = output;
    }

    graph.add_node("output_relu", OpType::ReLU, {accumulated}, "output");
    return graph;
}

static long long run_benchmark(ExecutionMode mode, size_t iterations){
    SessionOptions options;
    options.execution_mode = mode;
    options.inter_op_threads = 4;

    InferenceSession session(options);
    session.load(make_branch_graph(8), "input", {64, 64}, "output");
    session.initialize();
    Tensor input({64, 64});
    input.fill(1.0f);

    const auto start = std::chrono::steady_clock::now();
    float checksum = 0.0f;

    for(size_t i = 0; i < iterations; i++){
        checksum += session.run(input).data()[0];
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    if(checksum != static_cast<float>(iterations * 8)){
        throw std::runtime_error("benchmark correctness check failed");
    }

    return elapsed;
}

int main(){
    constexpr size_t iterations = 1000;
    std::cout << "sequential_us="
              << run_benchmark(ExecutionMode::Sequential, iterations) << "\n";
    std::cout << "parallel_us="
              << run_benchmark(ExecutionMode::Parallel, iterations) << "\n";
    return 0;
}

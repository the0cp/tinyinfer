#include "inference_session.h"

#include <chrono>
#include <iostream>

int main(){
    using namespace tinyinfer;

    Graph graph;
    graph.add_node("left", OpType::ReLU, {"input"}, "left_out");
    graph.add_node("right", OpType::ReLU, {"input"}, "right_out");
    graph.add_node("sum", OpType::Add, {"left_out", "right_out"}, "output");

    InferenceSession session;
    session.load(std::move(graph), "input", {1, 1024}, "output");
    session.initialize();

    Tensor input({1, 1024});
    input.fill(1.0f);

    constexpr size_t iterations = 10000;
    const auto start = std::chrono::steady_clock::now();

    float checksum = 0.0f;
    for(size_t i = 0; i < iterations; i++){
        checksum += session.run(input).data()[0];
    }

    const auto end = std::chrono::steady_clock::now();
    const auto microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "iterations=" << iterations
              << ", elapsed_us=" << microseconds
              << ", checksum=" << checksum << "\n";
    return checksum == static_cast<float>(iterations * 2) ? 0 : 1;
}

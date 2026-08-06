#include "inference_session.h"

#include <iostream>

int main(){
    using namespace tinyinfer;

    Graph graph;
    graph.add_node("left", OpType::ReLU, {"input"}, "left_out");
    graph.add_node("right", OpType::ReLU, {"input"}, "right_out");
    graph.add_node("residual_add", OpType::Add, {"left_out", "right_out"}, "output");

    SessionOptions options;
    options.execution_mode = ExecutionMode::Parallel;
    options.inter_op_threads = 2;

    InferenceSession session(options);
    session.load(std::move(graph), "input", {1, 3}, "output");
    session.initialize();

    Tensor output = session.run(Tensor({1, 3}, {-1.0f, 2.0f, 3.0f}));
    std::cout << "residual output = ["
              << output.data()[0] << ", "
              << output.data()[1] << ", "
              << output.data()[2] << "]\n";
    return 0;
}

#include "inference_session.h"

#include <iostream>

int main(){
    using namespace tinyinfer;

    Graph graph;
    graph.set_tensor("weight", Tensor({2, 2}, {1.0f, -1.0f, 1.0f, 1.0f}));
    graph.set_tensor("bias", Tensor({2}, {0.0f, 0.0f}));
    graph.add_node("linear", OpType::Linear, {"input", "weight", "bias"}, "hidden");
    graph.add_node("relu", OpType::ReLU, {"hidden"}, "output");

    InferenceSession session;
    session.load(std::move(graph), "input", {1, 2}, "output");
    session.initialize();

    RunProfile profile;
    Tensor output = session.run(Tensor({1, 2}, {1.0f, 2.0f}), profile);
    std::cout << "output = [" << output.data()[0] << ", " << output.data()[1] << "]\n\n";
    std::cout << session.plan().dump();
    std::cout << "\n" << profile.summary();
    return 0;
}

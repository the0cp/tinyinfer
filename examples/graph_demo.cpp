#include "graph.h"

#include <iostream>
#include <utility>

using namespace tinyinfer;

int main(){
    Tensor x({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor w1({3, 4}, {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    });

    Tensor b1({4}, {
        0.1f, 0.1f, 0.1f, 0.1f
    });

    Tensor w2({4, 2}, {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f,
        0.7f, 0.8f
    });

    Tensor b2({2}, {
        0.5f, 0.5f
    });

    Graph graph;

    graph.set_tensor("w1", std::move(w1));
    graph.set_tensor("b1", std::move(b1));
    graph.set_tensor("w2", std::move(w2));
    graph.set_tensor("b2", std::move(b2));

    graph.add_node("linear2", OpType::Linear, {"a1", "w2", "b2"}, "logits");
    graph.add_node("softmax", OpType::Softmax, {"logits"}, "prob");
    graph.add_node("relu1", OpType::ReLU, {"h1"}, "a1");
    graph.add_node("linear1", OpType::Linear, {"input", "w1", "b1"}, "h1");

    std::cout << graph.dump() << "\n";

    ExecutionPlan plan = graph.compile("input", x.shape(), "prob");

    std::cout << graph.dump_plan(plan) << "\n";
    std::cout << graph.dump_memory_plan(plan) << "\n";

    Tensor prob = graph.run(plan, x);

    std::cout << graph.dump_tensors() << "\n";

    std::cout << "Output:\n";
    std::cout << prob.at({0, 0}) << " "
              << prob.at({0, 1}) << "\n";
    std::cout << prob.at({1, 0}) << " "
              << prob.at({1, 1}) << "\n";

    return 0;
}

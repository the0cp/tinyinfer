#include "graph.h"

#include <iostream>

using namespace tinyinfer;

int main() {
    Tensor x({2, 3}, {
        1, -2, 3,
        4, -5, 6
    });

    Tensor w({3, 3}, {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    });

    Tensor b({3}, {
        0, 0, 0
    });

    Graph graph;

    graph.set_tensor("w", std::move(w));
    graph.set_tensor("b", std::move(b));

    graph.add_node("relu1", OpType::ReLU, {"sum"}, "output");
    graph.add_node("add1", OpType::Add, {"input", "h"}, "sum");
    graph.add_node("linear1", OpType::Linear, {"input", "w", "b"}, "h");

    std::cout << graph.dump() << "\n";

    Tensor y = graph.forward("input", x, "output");

    std::cout << graph.dump_execution_order() << "\n";
    std::cout << graph.dump_tensors() << "\n";

    std::cout << "Output:\n";
    std::cout << y.at({0, 0}) << " "
              << y.at({0, 1}) << " "
              << y.at({0, 2}) << "\n";

    std::cout << y.at({1, 0}) << " "
              << y.at({1, 1}) << " "
              << y.at({1, 2}) << "\n";

    return 0;
}
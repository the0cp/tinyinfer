#include "ops.h"
#include "module.h"
#include "graph.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace tinyinfer;

static void assert_close(float actual, float expected, float eps = 1e-4f){
    if(std::fabs(actual - expected) > eps){
        std::cerr << "assert_close failed: actual = "
                  << actual << ", expected = " << expected << "\n";
        throw std::runtime_error("test failed");
    }
}

static void test_transpose_2d(){
    Tensor x({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor y = transpose_2d(x);

    assert_close(y.at({0, 0}), 1);
    assert_close(y.at({0, 1}), 4);

    assert_close(y.at({1, 0}), 2);
    assert_close(y.at({1, 1}), 5);

    assert_close(y.at({2, 0}), 3);
    assert_close(y.at({2, 1}), 6);
}

static void test_naive_matmul(){
    Tensor a({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor b({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });

    Tensor c = naive_matmul(a, b);

    assert_close(c.at({0, 0}), 22);
    assert_close(c.at({0, 1}), 28);
    assert_close(c.at({1, 0}), 49);
    assert_close(c.at({1, 1}), 64);
}

static void test_matmul_transposed_b(){
    Tensor a({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor b({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });

    Tensor bt = transpose_2d(b);

    Tensor c = matmul_transposed_b(a, bt);

    assert_close(c.at({0, 0}), 22);
    assert_close(c.at({0, 1}), 28);
    assert_close(c.at({1, 0}), 49);
    assert_close(c.at({1, 1}), 64);
}

static void test_add_bias(){
    Tensor x({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor b({3}, {
        10, 20, 30
    });

    Tensor y = add_bias(x, b);

    assert_close(y.at({0, 0}), 11);
    assert_close(y.at({0, 1}), 22);
    assert_close(y.at({0, 2}), 33);
    assert_close(y.at({1, 0}), 14);
    assert_close(y.at({1, 1}), 25);
    assert_close(y.at({1, 2}), 36);
}

static void test_mlp_forward(){
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

    Tensor h = linear(x, w1, b1);
    Tensor a = relu(h);
    Tensor y = linear(a, w2, b2);

    assert_close(y.at({0, 0}), 8.78f);
    assert_close(y.at({0, 1}), 10.7f);
    assert_close(y.at({1, 0}), 19.04f);
    assert_close(y.at({1, 1}), 23.3f);
}

static void test_softmax() {
    Tensor x({1, 3}, {
        1.0f, 2.0f, 3.0f
    });

    Tensor y = softmax(x);

    float sum = y.at({0, 0}) + y.at({0, 1}) + y.at({0, 2});

    assert_close(sum, 1.0f);
    assert_close(y.at({0, 0}), 0.0900306f);
    assert_close(y.at({0, 1}), 0.244728f);
    assert_close(y.at({0, 2}), 0.665241f);
}

static void test_sequential_forward() {
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

    Sequential model;
    model.add(std::make_unique<Linear>(std::move(w1), std::move(b1)));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(std::move(w2), std::move(b2)));

    Tensor y = model.forward(x);

    assert_close(y.at({0, 0}), 8.78f);
    assert_close(y.at({0, 1}), 10.7f);
    assert_close(y.at({1, 0}), 19.04f);
    assert_close(y.at({1, 1}), 23.3f);
}

static void test_graph_forward() {
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

    graph.add_node("linear1", OpType::Linear, {"input", "w1", "b1"}, "h1");
    graph.add_node("relu1", OpType::ReLU, {"h1"}, "a1");
    graph.add_node("linear2", OpType::Linear, {"a1", "w2", "b2"}, "logits");

    Tensor y = graph.forward("input", x, "logits");

    assert_close(y.at({0, 0}), 8.78f);
    assert_close(y.at({0, 1}), 10.7f);
    assert_close(y.at({1, 0}), 19.04f);
    assert_close(y.at({1, 1}), 23.3f);
}

static void test_graph_topological_forward() {
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
    graph.add_node("relu1", OpType::ReLU, {"h1"}, "a1");
    graph.add_node("linear1", OpType::Linear, {"input", "w1", "b1"}, "h1");

    Tensor y = graph.forward("input", x, "logits");

    assert_close(y.at({0, 0}), 8.78f);
    assert_close(y.at({0, 1}), 10.7f);
    assert_close(y.at({1, 0}), 19.04f);
    assert_close(y.at({1, 1}), 23.3f);
}

static void test_graph_repeated_forward() {
    Tensor x1({1, 3}, {
        1, 2, 3
    });

    Tensor x2({1, 3}, {
        4, 5, 6
    });

    Tensor w1({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });

    Tensor b1({2}, {
        0, 0
    });

    Graph graph;

    graph.set_tensor("w1", std::move(w1));
    graph.set_tensor("b1", std::move(b1));

    graph.add_node("linear1", OpType::Linear, {"input", "w1", "b1"}, "logits");

    Tensor y1 = graph.forward("input", x1, "logits");
    Tensor y2 = graph.forward("input", x2, "logits");

    assert_close(y1.at({0, 0}), 22);
    assert_close(y1.at({0, 1}), 28);

    assert_close(y2.at({0, 0}), 49);
    assert_close(y2.at({0, 1}), 64);
}

static void test_graph_execution_order() {
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
    graph.add_node("relu1", OpType::ReLU, {"h1"}, "a1");
    graph.add_node("linear1", OpType::Linear, {"input", "w1", "b1"}, "h1");

    Tensor y = graph.forward("input", x, "logits");

    assert_close(y.at({0, 0}), 8.78f);
    assert_close(y.at({0, 1}), 10.7f);
    assert_close(y.at({1, 0}), 19.04f);
    assert_close(y.at({1, 1}), 23.3f);

    const auto& order = graph.last_execution_order();

    if (order.size() != 3) {
        throw std::runtime_error("execution order size mismatch");
    }

    if (order[0] != "linear1" || order[1] != "relu1" || order[2] != "linear2") {
        throw std::runtime_error("unexpected execution order");
    }
}

static void test_graph_missing_dependency() {
    Graph graph;

    graph.add_node("relu1", OpType::ReLU, {"missing_tensor"}, "out");

    Tensor x({1, 1}, {1.0f});

    bool caught = false;

    try {
        graph.forward("input", x, "out");
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("missing dependency test failed");
    }
}

static void test_graph_residual_add() {
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

    Tensor y = graph.forward("input", x, "output");

    assert_close(y.at({0, 0}), 2.0f);
    assert_close(y.at({0, 1}), 0.0f);
    assert_close(y.at({0, 2}), 6.0f);

    assert_close(y.at({1, 0}), 8.0f);
    assert_close(y.at({1, 1}), 0.0f);
    assert_close(y.at({1, 2}), 12.0f);

    const auto& order = graph.last_execution_order();

    if (order.size() != 3) {
        throw std::runtime_error("residual execution order size mismatch");
    }

    if (order[0] != "linear1" || order[1] != "add1" || order[2] != "relu1") {
        throw std::runtime_error("unexpected residual execution order");
    }
}

int main(){
    test_transpose_2d();
    test_naive_matmul();
    test_matmul_transposed_b();
    test_add_bias();
    test_mlp_forward();
    test_softmax();
    test_sequential_forward();
    test_graph_forward();
    test_graph_topological_forward();
    test_graph_repeated_forward();
    test_graph_execution_order();
    test_graph_missing_dependency();
    test_graph_residual_add();

    std::cout << "All tests passed.\n";
    return 0;
}
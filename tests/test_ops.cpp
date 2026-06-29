#include "ops.h"
#include "module.h"
#include "graph.h"
#include "operator_registry.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <atomic>
#include <memory>
#include <string>
#include <utility>

using namespace tinyinfer;

static void assert_close(float actual, float expected, float eps = 1e-4f){
    if(std::fabs(actual - expected) > eps){
        std::cerr << "assert_close failed: actual = "
                  << actual << ", expected = " << expected << "\n";
        throw std::runtime_error("test failed");
    }
}

static void assert_shape(
    const Shape& actual,
    const Shape& expected
){
    if(actual != expected){
        throw std::runtime_error("shape mismatch");
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

static void test_graph_forward(){
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

static void test_graph_compile_topological_order(){
    Tensor w1({3, 4});
    Tensor b1({4});
    Tensor w2({4, 2});
    Tensor b2({2});

    Graph graph;

    graph.set_tensor("w1", std::move(w1));
    graph.set_tensor("b1", std::move(b1));
    graph.set_tensor("w2", std::move(w2));
    graph.set_tensor("b2", std::move(b2));

    graph.add_node("linear2", OpType::Linear, {"a1", "w2", "b2"}, "logits");
    graph.add_node("relu1", OpType::ReLU, {"h1"}, "a1");
    graph.add_node("linear1", OpType::Linear, {"input", "w1", "b1"}, "h1");

    ExecutionPlan plan = graph.compile("input", {2, 3}, "logits");

    const auto& order = plan.execution_order();

    if(order.size() != 3){
        throw std::runtime_error("execution order size mismatch");
    }

    if(order[0] != "linear1" ||
       order[1] != "relu1" ||
       order[2] != "linear2"){
        throw std::runtime_error("unexpected execution order");
    }

    if(plan.node_count() != 3){
        throw std::runtime_error("ExecutionPlan node count mismatch");
    }
}

static void test_execution_plan_reuse(){
    Graph graph;

    Tensor weight({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });

    Tensor bias({2}, {0, 0});

    graph.set_tensor("weight", std::move(weight));
    graph.set_tensor("bias", std::move(bias));

    graph.add_node(
        "linear1",
        OpType::Linear,
        {"input", "weight", "bias"},
        "output"
    );

    ExecutionPlan plan = graph.compile(
        "input",
        {1, 3},
        "output"
    );

    Tensor x1({1, 3}, {1, 2, 3});
    Tensor x2({1, 3}, {4, 5, 6});

    Tensor y1 = graph.run(plan, x1);
    Tensor y2 = graph.run(plan, x2);

    assert_close(y1.at({0, 0}), 22);
    assert_close(y1.at({0, 1}), 28);
    assert_close(y2.at({0, 0}), 49);
    assert_close(y2.at({0, 1}), 64);
}

static void test_execution_plan_shapes(){
    Graph graph;

    graph.set_tensor("w1", Tensor({3, 4}));
    graph.set_tensor("b1", Tensor({4}));
    graph.set_tensor("w2", Tensor({4, 2}));
    graph.set_tensor("b2", Tensor({2}));

    graph.add_node("linear2", OpType::Linear, {"a1", "w2", "b2"}, "logits");
    graph.add_node("relu1", OpType::ReLU, {"h1"}, "a1");
    graph.add_node("linear1", OpType::Linear, {"input", "w1", "b1"}, "h1");

    ExecutionPlan plan = graph.compile("input", {2, 3}, "logits");

    assert_shape(plan.shape("input"), {2, 3});
    assert_shape(plan.shape("h1"), {2, 4});
    assert_shape(plan.shape("a1"), {2, 4});
    assert_shape(plan.shape("logits"), {2, 2});

    if(plan.input_name() != "input" ||
       plan.output_name() != "logits"){
        throw std::runtime_error("ExecutionPlan endpoint mismatch");
    }
}

static void test_execution_plan_invalidation(){
    Graph graph;

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"input"},
        "output"
    );

    ExecutionPlan plan = graph.compile(
        "input",
        {1, 1},
        "output"
    );

    graph.set_tensor(
        "new_constant",
        Tensor({1}, {1.0f})
    );

    bool caught = false;

    try{
        graph.run(plan, Tensor({1, 1}, {1.0f}));
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "stale ExecutionPlan was accepted"
        );
    }
}

static void test_execution_plan_wrong_graph(){
    Graph graph_a;
    Graph graph_b;

    graph_a.add_node("relu", OpType::ReLU, {"input"}, "output");
    graph_b.add_node("relu", OpType::ReLU, {"input"}, "output");

    ExecutionPlan plan = graph_a.compile(
        "input",
        {1, 1},
        "output"
    );

    bool caught = false;

    try{
        graph_b.run(plan, Tensor({1, 1}, {1.0f}));
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "ExecutionPlan was accepted by another Graph"
        );
    }
}

static void test_execution_plan_input_shape_mismatch(){
    Graph graph;

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"input"},
        "output"
    );

    ExecutionPlan plan = graph.compile(
        "input",
        {2, 3},
        "output"
    );

    bool caught = false;

    try{
        graph.run(plan, Tensor({1, 3}));
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "ExecutionPlan accepted an incorrect input shape"
        );
    }
}

static void test_graph_missing_dependency(){
    Graph graph;

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"missing_tensor"},
        "out"
    );

    bool caught = false;

    try{
        graph.compile("input", {1, 1}, "out");
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "missing dependency test failed"
        );
    }
}

static void test_graph_residual_add(){
    Tensor x({2, 3}, {
        1, -2, 3,
        4, -5, 6
    });

    Tensor w({3, 3}, {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    });

    Tensor b({3}, {0, 0, 0});

    Graph graph;

    graph.set_tensor("w", std::move(w));
    graph.set_tensor("b", std::move(b));

    graph.add_node("relu1", OpType::ReLU, {"sum"}, "output");
    graph.add_node("add1", OpType::Add, {"input", "h"}, "sum");
    graph.add_node("linear1", OpType::Linear, {"input", "w", "b"}, "h");

    ExecutionPlan plan = graph.compile(
        "input",
        x.shape(),
        "output"
    );

    Tensor y = graph.run(plan, x);

    assert_close(y.at({0, 0}), 2.0f);
    assert_close(y.at({0, 1}), 0.0f);
    assert_close(y.at({0, 2}), 6.0f);
    assert_close(y.at({1, 0}), 8.0f);
    assert_close(y.at({1, 1}), 0.0f);
    assert_close(y.at({1, 2}), 12.0f);

    const auto& order = plan.execution_order();

    if(order.size() != 3 ||
       order[0] != "linear1" ||
       order[1] != "add1" ||
       order[2] != "relu1"){
        throw std::runtime_error(
            "unexpected residual execution order"
        );
    }
}

static void test_graph_duplicate_node_name(){
    Graph graph;

    graph.add_node("same", OpType::ReLU, {"input"}, "a");
    graph.add_node("same", OpType::ReLU, {"a"}, "b");

    bool caught = false;

    try{
        graph.compile("input", {1, 1}, "b");
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "duplicate node name test failed"
        );
    }
}

static void test_graph_duplicate_output(){
    Graph graph;

    graph.add_node("relu1", OpType::ReLU, {"input"}, "same_output");
    graph.add_node("relu2", OpType::ReLU, {"input"}, "same_output");

    bool caught = false;

    try{
        graph.compile("input", {1, 1}, "same_output");
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "duplicate output test failed"
        );
    }
}

static void test_graph_wrong_input_count(){
    Graph graph;

    graph.add_node("add1", OpType::Add, {"input"}, "out");

    bool caught = false;

    try{
        graph.compile("input", {1, 1}, "out");
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "wrong input count test failed"
        );
    }
}

static void test_graph_missing_requested_output(){
    Graph graph;

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"input"},
        "real_output"
    );

    bool caught = false;

    try{
        graph.compile(
            "input",
            {1, 1},
            "not_exist_output"
        );
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "missing requested output test failed"
        );
    }
}

static void test_graph_linear_shape_mismatch(){
    Graph graph;

    graph.set_tensor("weight", Tensor({4, 2}));
    graph.set_tensor("bias", Tensor({2}));

    graph.add_node(
        "linear1",
        OpType::Linear,
        {"input", "weight", "bias"},
        "output"
    );

    bool caught = false;

    try{
        graph.compile("input", {2, 3}, "output");
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "linear shape mismatch test failed"
        );
    }
}

static void test_graph_add_shape_mismatch(){
    Graph graph;

    graph.set_tensor("skip", Tensor({2, 4}));

    graph.add_node(
        "add1",
        OpType::Add,
        {"input", "skip"},
        "output"
    );

    bool caught = false;

    try{
        graph.compile("input", {2, 3}, "output");
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "add shape mismatch test failed"
        );
    }
}

static void test_graph_constant_name_collision(){
    Graph graph;

    graph.set_tensor("weight", Tensor({1, 1}, {1.0f}));
    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"input"},
        "weight"
    );

    bool caught = false;

    try{
        graph.compile("input", {1, 1}, "weight");
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "node output was allowed to overwrite a constant"
        );
    }
}

static void test_graph_dump_plan(){
    Graph graph;

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"input"},
        "output"
    );

    ExecutionPlan plan = graph.compile(
        "input",
        {1, 2},
        "output"
    );

    const std::string dump = graph.dump_plan(plan);

    if(dump.find("relu1") == std::string::npos ||
       dump.find("[1, 2]") == std::string::npos){
        throw std::runtime_error(
            "ExecutionPlan dump is missing expected information"
        );
    }
}

static void test_operator_registry() {
    OperatorRegistry registry;

    const OperatorDefinition& linear =
        registry.get(OpType::Linear);

    if (linear.name != "Linear") {
        throw std::runtime_error(
            "Linear registry name mismatch"
        );
    }

    if (linear.input_count != 3) {
        throw std::runtime_error(
            "Linear registry input count mismatch"
        );
    }

    if (linear.execute == nullptr) {
        throw std::runtime_error(
            "Linear execution kernel is missing"
        );
    }

    if (linear.infer_shape == nullptr) {
        throw std::runtime_error(
            "Linear shape kernel is missing"
        );
    }

    const OperatorDefinition& add =
        registry.get(OpType::Add);

    if (add.name != "Add" ||
        add.input_count != 2) {
        throw std::runtime_error(
            "Add registry metadata mismatch"
        );
    }
}

static void test_thread_pool_tasks(){
    ThreadPool pool(4);

    std::atomic<size_t> counter{0};

    for(size_t i = 0; i < 1000; i++){
        pool.enqueue([&counter](){
            counter.fetch_add(
                1,
                std::memory_order_relaxed
            );
        });
    }

    pool.wait();

    if(counter.load() != 1000){
        throw std::runtime_error(
            "ThreadPool task count mismatch"
        );
    }
}

static void test_thread_pool_exception(){
    ThreadPool pool(2);

    pool.enqueue([](){
        throw std::runtime_error("task failed");
    });

    bool caught = false;

    try{
        pool.wait();
    }catch(const std::runtime_error& error){
        caught =
            std::string(error.what()) == "task failed";
    }

    if(!caught){
        throw std::runtime_error(
            "ThreadPool did not propagate task failure"
        );
    }

    std::atomic<bool> ran{false};

    pool.enqueue([&ran](){
        ran.store(true);
    });

    pool.wait();

    if(!ran.load()){
        throw std::runtime_error(
            "ThreadPool is unusable after task failure"
        );
    }
}

static void test_thread_pool_self_wait(){
    ThreadPool pool(1);

    pool.enqueue([&pool](){
        pool.wait();
    });

    bool caught = false;

    try{
        pool.wait();
    }catch(const std::logic_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "ThreadPool self-wait was not rejected"
        );
    }
}

static void test_threadpool_matmul_zero_rows(){
    Tensor a({0, 3});
    Tensor b({3, 2});

    ThreadPool pool(2);

    Tensor out =
        threadpool_matmul(a, b, pool, 2);

    assert_shape(out.shape(), {0, 2});
}

static void test_blocked_matmul_zero_block(){
    Tensor a({1, 1}, {1.0f});
    Tensor b({1, 1}, {1.0f});

    bool caught = false;

    try{
        blocked_matmul(a, b, 0);
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "blocked_matmul accepted block size 0"
        );
    }
}

static void test_softmax_empty_features(){
    Tensor x({2, 0});

    bool caught = false;

    try{
        softmax(x);
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "softmax accepted empty feature dimension"
        );
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
    test_graph_compile_topological_order();
    test_execution_plan_reuse();
    test_execution_plan_shapes();
    test_execution_plan_invalidation();
    test_execution_plan_wrong_graph();
    test_execution_plan_input_shape_mismatch();
    test_graph_missing_dependency();
    test_graph_residual_add();
    test_graph_duplicate_node_name();
    test_graph_duplicate_output();
    test_graph_wrong_input_count();
    test_graph_missing_requested_output();
    test_graph_linear_shape_mismatch();
    test_graph_add_shape_mismatch();
    test_graph_constant_name_collision();
    test_graph_dump_plan();
    test_operator_registry();
    test_thread_pool_tasks();
    test_thread_pool_exception();
    test_thread_pool_self_wait();
    test_threadpool_matmul_zero_rows();
    test_blocked_matmul_zero_block();
    test_softmax_empty_features();

    std::cout << "All tests passed.\n";
    return 0;
}

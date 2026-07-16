#include "ops.h"
#include "module.h"
#include "graph.h"
#include "operator_registry.h"
#include "model_loader.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <atomic>
#include <exception>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace tinyinfer;

static_assert(!std::is_copy_constructible_v<Graph>);
static_assert(!std::is_copy_assignable_v<Graph>);
static_assert(!std::is_move_constructible_v<Graph>);
static_assert(!std::is_move_assignable_v<Graph>);
static_assert(!std::is_copy_constructible_v<ExecutionContext>);
static_assert(!std::is_copy_assignable_v<ExecutionContext>);
static_assert(!std::is_move_constructible_v<ExecutionContext>);
static_assert(!std::is_move_assignable_v<ExecutionContext>);

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

static void assert_lifetime(
    const TensorMemoryInfo& info,
    size_t produced_at,
    size_t first_use,
    size_t last_use,
    const std::string& name
){
    if(info.produced_at != produced_at || info.first_use != first_use || info.last_use != last_use){
        throw std::runtime_error("lifetime mismatch for tensor: " + name);
    }
}


static void assert_consumers(
    const NodeScheduleInfo& info,
    const std::vector<size_t>& expected,
    const std::string& name
){
    if(info.consumers != expected){
        throw std::runtime_error("consumer list mismatch for node: " + name);
    }
}

static const TensorMemoryInfo& find_memory_info(const ExecutionPlan& plan, const std::string& name){
    for(const TensorMemoryInfo& info : plan.memory_infos()){
        if(info.name == name){
            return info;
        }
    }

    throw std::runtime_error("memory info not found: " + name);
}

static std::filesystem::path reset_temp_dir(const std::string& name){
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

static void write_text_file(const std::filesystem::path& path, const std::string& text){
    std::ofstream file(path);

    if(!file){
        throw std::runtime_error("failed to open text file for writing: " + path.string());
    }

    file << text;

    if(!file){
        throw std::runtime_error("failed to write text file: " + path.string());
    }
}

static void write_float_weights(const std::filesystem::path& path, const std::vector<float>& values){
    std::ofstream file(path, std::ios::binary);

    if(!file){
        throw std::runtime_error("failed to open weights file for writing: " + path.string());
    }

    if(!values.empty()){
        file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    if(!file){
        throw std::runtime_error("failed to write weights file: " + path.string());
    }
}

static void expect_model_load_failure(const std::filesystem::path& manifest_path, const std::string& test_name){
    bool caught = false;

    try{
        std::unique_ptr<LoadedModel> model = ModelLoader::load(manifest_path);
        (void)model;
    }catch(const std::exception&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(test_name + " did not fail");
    }
}

static void expect_model_write_failure(
    const ModelPackage& package,
    const std::filesystem::path& manifest_path,
    const std::string& test_name
){
    bool caught = false;

    try{
        ModelWriter::save(package, manifest_path);
    }catch(const std::exception&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(test_name + " did not fail");
    }
}

static Tensor run_graph(const Graph& graph, const ExecutionPlan& plan, const Tensor& input){
    ExecutionContext context;
    return graph.run(plan, context, input);
}

static Tensor registry_dummy_execute(const TensorInputs&){
    return Tensor({1}, {0.0f});
}

static Shape registry_dummy_infer_shape(const ShapeInputs&, const std::string&){
    return {1};
}


static void test_tensor_shape_overflow(){
    bool caught = false;

    try{
        Tensor too_large({
            std::numeric_limits<size_t>::max(),
            2
        });
        (void)too_large;
    }catch(const std::runtime_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error("Tensor accepted an overflowing shape");
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

static void test_graph_run_releases_unused_intermediates(){
    Tensor x({1, 3}, {
        -1.0f, 2.0f, 3.0f
    });

    Graph graph;

    graph.add_node("relu1", OpType::ReLU, {"input"}, "hidden");
    graph.add_node("relu2", OpType::ReLU, {"hidden"}, "output");

    ExecutionPlan plan = graph.compile("input", x.shape(), "output");
    ExecutionContext context;
    Tensor y = graph.run(plan, context, x);

    assert_close(y.at({0, 0}), 0.0f);
    assert_close(y.at({0, 1}), 2.0f);
    assert_close(y.at({0, 2}), 3.0f);

    if(context.has_tensor("hidden")){
        throw std::runtime_error("unused intermediate tensor was not released");
    }

    if(!context.has_tensor("input") || !context.has_tensor("output")){
        throw std::runtime_error("runtime cleanup removed input or output tensor");
    }
}

static void test_graph_run_keeps_intermediate_until_last_use(){
    Tensor x({1, 3}, {
        -1.0f, 2.0f, 3.0f
    });

    Graph graph;

    graph.add_node("relu1", OpType::ReLU, {"input"}, "hidden");
    graph.add_node("relu2", OpType::ReLU, {"hidden"}, "hidden2");
    graph.add_node("add1", OpType::Add, {"hidden", "hidden2"}, "output");

    ExecutionPlan plan = graph.compile("input", x.shape(), "output");
    ExecutionContext context;
    Tensor y = graph.run(plan, context, x);

    assert_close(y.at({0, 0}), 0.0f);
    assert_close(y.at({0, 1}), 4.0f);
    assert_close(y.at({0, 2}), 6.0f);

    if(context.has_tensor("hidden") || context.has_tensor("hidden2")){
        throw std::runtime_error("runtime cleanup kept a dead intermediate tensor");
    }

    if(!context.has_tensor("input") || !context.has_tensor("output")){
        throw std::runtime_error("runtime cleanup removed input or output tensor");
    }
}

static void test_graph_run_ready_queue_branch_graph(){
    Tensor x({1, 3}, {
        -1.0f, 2.0f, 3.0f
    });

    Graph graph;

    graph.add_node("left_relu", OpType::ReLU, {"input"}, "left");
    graph.add_node("right_relu", OpType::ReLU, {"input"}, "right");
    graph.add_node("join_add", OpType::Add, {"left", "right"}, "joined");
    graph.add_node("out_relu", OpType::ReLU, {"joined"}, "output");

    ExecutionPlan plan = graph.compile("input", x.shape(), "output");
    ExecutionContext context;
    Tensor y = graph.run(plan, context, x);

    assert_close(y.at({0, 0}), 0.0f);
    assert_close(y.at({0, 1}), 4.0f);
    assert_close(y.at({0, 2}), 6.0f);

    if(context.has_tensor("left") ||
       context.has_tensor("right") ||
       context.has_tensor("joined")){
        throw std::runtime_error("ready-queue executor kept a dead branch intermediate tensor");
    }

    if(!context.has_tensor("input") || !context.has_tensor("output")){
        throw std::runtime_error("ready-queue executor removed input or output tensor");
    }
}


static void test_graph_run_parallel_branch_graph(){
    Tensor x({1, 3}, {
        -1.0f, 2.0f, 3.0f
    });

    Graph graph;

    graph.add_node("left_relu", OpType::ReLU, {"input"}, "left");
    graph.add_node("right_relu", OpType::ReLU, {"input"}, "right");
    graph.add_node("join_add", OpType::Add, {"left", "right"}, "joined");
    graph.add_node("out_relu", OpType::ReLU, {"joined"}, "output");

    ExecutionPlan plan = graph.compile("input", x.shape(), "output");
    ExecutionContext context;
    ThreadPool pool(2);

    Tensor y = graph.run_parallel(plan, context, x, pool);

    assert_close(y.at({0, 0}), 0.0f);
    assert_close(y.at({0, 1}), 4.0f);
    assert_close(y.at({0, 2}), 6.0f);

    if(context.has_tensor("left") ||
       context.has_tensor("right") ||
       context.has_tensor("joined")){
        throw std::runtime_error("parallel executor kept a dead branch intermediate tensor");
    }

    if(!context.has_tensor("input") || !context.has_tensor("output")){
        throw std::runtime_error("parallel executor removed input or output tensor");
    }
}

static void test_graph_run_parallel_rejects_same_pool_worker(){
    Graph graph;

    graph.add_node("relu1", OpType::ReLU, {"input"}, "output");

    ExecutionPlan plan = graph.compile("input", {1, 2}, "output");
    Tensor x({1, 2}, {-1.0f, 2.0f});
    ThreadPool pool(2);

    pool.enqueue([&](){
        ExecutionContext context;
        (void)graph.run_parallel(plan, context, x, pool);
    });

    bool caught = false;

    try{
        pool.wait();
    }catch(const std::logic_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "parallel executor did not reject being called from its own worker pool"
        );
    }
}

static void test_graph_run_parallel_stress(){
    Graph graph;

    graph.add_node("left_relu", OpType::ReLU, {"input"}, "left");
    graph.add_node("right_relu", OpType::ReLU, {"input"}, "right");
    graph.add_node("join_add", OpType::Add, {"left", "right"}, "joined");
    graph.add_node("out_relu", OpType::ReLU, {"joined"}, "output");

    ExecutionPlan plan = graph.compile("input", {1, 4}, "output");
    ThreadPool pool(4);

    for(size_t i = 0; i < 64; i++){
        const float base = static_cast<float>(i % 7) - 3.0f;
        Tensor x({1, 4}, {base, base + 1.0f, base + 2.0f, base + 3.0f});

        ExecutionContext context;
        Tensor y = graph.run_parallel(plan, context, x, pool);

        for(size_t j = 0; j < 4; j++){
            const float relu_value = std::max(0.0f, x.at({0, j}));
            assert_close(y.at({0, j}), relu_value * 2.0f);
        }

        if(context.has_tensor("left") ||
           context.has_tensor("right") ||
           context.has_tensor("joined")){
            throw std::runtime_error("parallel stress run kept a dead intermediate tensor");
        }

        if(!context.has_tensor("input") || !context.has_tensor("output")){
            throw std::runtime_error("parallel stress run removed input or output tensor");
        }
    }
}


static void test_graph_run_parallel_does_not_wait_for_unrelated_pool_tasks(){
    Graph graph;

    graph.add_node("left_relu", OpType::ReLU, {"input"}, "left");
    graph.add_node("right_relu", OpType::ReLU, {"input"}, "right");
    graph.add_node("join_add", OpType::Add, {"left", "right"}, "output");

    ExecutionPlan plan = graph.compile("input", {1, 3}, "output");
    Tensor input({1, 3}, {-1.0f, 2.0f, 3.0f});
    ThreadPool pool(3);

    std::promise<void> blocker_started_promise;
    std::future<void> blocker_started = blocker_started_promise.get_future();

    std::promise<void> release_blocker_promise;
    std::shared_future<void> release_blocker =
        release_blocker_promise.get_future().share();

    pool.enqueue([&](){
        blocker_started_promise.set_value();
        release_blocker.wait();
    });

    blocker_started.wait();

    std::future<Tensor> graph_future = std::async(std::launch::async, [&](){
        ExecutionContext context;
        return graph.run_parallel(plan, context, input, pool);
    });

    const std::future_status status =
        graph_future.wait_for(std::chrono::seconds(2));

    if(status != std::future_status::ready){
        release_blocker_promise.set_value();
        graph_future.wait();
        pool.wait();

        throw std::runtime_error(
            "parallel executor waited for unrelated ThreadPool work"
        );
    }

    Tensor output = graph_future.get();

    assert_close(output.at({0, 0}), 0.0f);
    assert_close(output.at({0, 1}), 4.0f);
    assert_close(output.at({0, 2}), 6.0f);

    release_blocker_promise.set_value();
    pool.wait();
}

static void test_graph_run_parallel_failure_cancels_descendants(){
    Graph graph;

    graph.add_node("relu1", OpType::ReLU, {"input"}, "hidden");
    graph.add_node("relu2", OpType::ReLU, {"hidden"}, "output");

    ExecutionPlan plan = graph.compile("input", {1, 2}, "output");

    // White-box fault injection: the plan itself is non-const, so this
    // deliberately corrupts scheduler metadata to exercise a worker failure.
    auto& schedule =
        const_cast<std::vector<NodeScheduleInfo>&>(plan.schedule_infos());
    schedule[0].consumers.push_back(999);

    Tensor input({1, 2}, {-1.0f, 2.0f});
    ExecutionContext context;
    ThreadPool pool(2);

    bool caught = false;

    try{
        (void)graph.run_parallel(plan, context, input, pool);
    }catch(const std::logic_error&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error(
            "parallel executor did not propagate worker failure"
        );
    }

    if(context.has_tensor("input") ||
       context.has_tensor("hidden") ||
       context.has_tensor("output")){
        throw std::runtime_error(
            "failed parallel run left partial tensors in ExecutionContext"
        );
    }

    std::atomic<bool> pool_reused{false};

    pool.enqueue([&](){
        pool_reused.store(true);
    });

    pool.wait();

    if(!pool_reused.load()){
        throw std::runtime_error(
            "ThreadPool was not reusable after parallel graph failure"
        );
    }
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

    Tensor y1 = run_graph(graph, plan, x1);
    Tensor y2 = run_graph(graph, plan, x2);

    assert_close(y1.at({0, 0}), 22);
    assert_close(y1.at({0, 1}), 28);
    assert_close(y2.at({0, 0}), 49);
    assert_close(y2.at({0, 1}), 64);
}

static void test_execution_context_separate_workspaces(){
    Graph graph;

    graph.add_node("relu1", OpType::ReLU, {"input"}, "output");

    ExecutionPlan plan = graph.compile("input", {1, 2}, "output");

    ExecutionContext context1;
    ExecutionContext context2;

    Tensor x1({1, 2}, {-1.0f, 2.0f});
    Tensor x2({1, 2}, {3.0f, -4.0f});

    Tensor y1 = graph.run(plan, context1, x1);
    Tensor y2 = graph.run(plan, context2, x2);

    assert_close(y1.at({0, 0}), 0.0f);
    assert_close(y1.at({0, 1}), 2.0f);
    assert_close(y2.at({0, 0}), 3.0f);
    assert_close(y2.at({0, 1}), 0.0f);

    const Tensor& saved1 = context1.tensor("output");
    const Tensor& saved2 = context2.tensor("output");

    assert_close(saved1.at({0, 0}), 0.0f);
    assert_close(saved1.at({0, 1}), 2.0f);
    assert_close(saved2.at({0, 0}), 3.0f);
    assert_close(saved2.at({0, 1}), 0.0f);
}

static void test_execution_context_parallel_runs(){
    Graph graph;

    Tensor weight({3, 2}, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    });

    Tensor bias({2}, {1.0f, -1.0f});

    graph.set_tensor("weight", std::move(weight));
    graph.set_tensor("bias", std::move(bias));

    graph.add_node(
        "linear1",
        OpType::Linear,
        {"input", "weight", "bias"},
        "hidden"
    );

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"hidden"},
        "output"
    );

    ExecutionPlan plan = graph.compile("input", {1, 3}, "output");

    constexpr size_t thread_count = 8;
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(thread_count);
    std::vector<float> actual0(thread_count, 0.0f);
    std::vector<float> actual1(thread_count, 0.0f);

    threads.reserve(thread_count);

    for(size_t i = 0; i < thread_count; i++){
        threads.emplace_back([&, i]{
            try{
                const float base = static_cast<float>(i + 1);
                Tensor input({1, 3}, {base, base + 1.0f, base + 2.0f});

                ExecutionContext context;
                Tensor output = graph.run(plan, context, input);

                actual0[i] = output.at({0, 0});
                actual1[i] = output.at({0, 1});

                if(context.has_tensor("hidden")){
                    throw std::runtime_error("parallel run kept a dead intermediate tensor");
                }

                if(!context.has_tensor("input") || !context.has_tensor("output")){
                    throw std::runtime_error("parallel run lost input or output tensor");
                }
            }catch(...){
                errors[i] = std::current_exception();
            }
        });
    }

    for(std::thread& thread : threads){
        thread.join();
    }

    for(const std::exception_ptr& error : errors){
        if(error){
            std::rethrow_exception(error);
        }
    }

    for(size_t i = 0; i < thread_count; i++){
        const float base = static_cast<float>(i + 1);
        const float expected0 = base + (base + 1.0f) * 3.0f + (base + 2.0f) * 5.0f + 1.0f;
        const float expected1 = base * 2.0f + (base + 1.0f) * 4.0f + (base + 2.0f) * 6.0f - 1.0f;

        assert_close(actual0[i], expected0);
        assert_close(actual1[i], expected1);
    }
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

static void test_execution_plan_memory_infos(){
    Graph graph;

    graph.set_tensor("weight", Tensor({3, 2}));
    graph.set_tensor("bias", Tensor({2}));

    graph.add_node(
        "linear1",
        OpType::Linear,
        {"input", "weight", "bias"},
        "h1"
    );

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"h1"},
        "output"
    );

    ExecutionPlan plan = graph.compile("input", {4, 3}, "output");

    if(plan.memory_infos().size() != 5){
        throw std::runtime_error("ExecutionPlan memory info count mismatch");
    }

    const TensorMemoryInfo& input = find_memory_info(plan, "input");
    assert_shape(input.shape, {4, 3});

    if(input.numel != 12 || input.byte_size != 12 * sizeof(float) ||
       !input.is_input || input.is_output || input.is_constant || input.is_intermediate){
        throw std::runtime_error("input memory info mismatch");
    }

    assert_lifetime(input, tensor_lifetime_npos, 0, 0, "input");

    const TensorMemoryInfo& weight = find_memory_info(plan, "weight");

    if(weight.numel != 6 || weight.byte_size != 6 * sizeof(float) ||
       weight.is_input || weight.is_output || !weight.is_constant || weight.is_intermediate){
        throw std::runtime_error("constant memory info mismatch");
    }

    assert_lifetime(weight, tensor_lifetime_npos, 0, 0, "weight");

    const TensorMemoryInfo& h1 = find_memory_info(plan, "h1");

    if(h1.numel != 8 || h1.byte_size != 8 * sizeof(float) ||
       h1.is_input || h1.is_output || h1.is_constant || !h1.is_intermediate){
        throw std::runtime_error("intermediate memory info mismatch");
    }

    assert_lifetime(h1, 0, 1, 1, "h1");

    const TensorMemoryInfo& output = find_memory_info(plan, "output");

    if(output.numel != 8 || output.byte_size != 8 * sizeof(float) ||
       output.is_input || !output.is_output || output.is_constant || output.is_intermediate){
        throw std::runtime_error("output memory info mismatch");
    }

    assert_lifetime(output, 1, tensor_lifetime_npos, tensor_lifetime_npos, "output");
}

static void test_execution_plan_lifetime_multi_use(){
    Graph graph;

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"input"},
        "hidden"
    );

    graph.add_node(
        "add1",
        OpType::Add,
        {"input", "hidden"},
        "output"
    );

    ExecutionPlan plan = graph.compile("input", {2, 3}, "output");

    const TensorMemoryInfo& input = find_memory_info(plan, "input");
    const TensorMemoryInfo& hidden = find_memory_info(plan, "hidden");
    const TensorMemoryInfo& output = find_memory_info(plan, "output");

    assert_lifetime(input, tensor_lifetime_npos, 0, 1, "input");
    assert_lifetime(hidden, 0, 1, 1, "hidden");
    assert_lifetime(output, 1, tensor_lifetime_npos, tensor_lifetime_npos, "output");
}


static void test_execution_plan_scheduler_infos(){
    Graph graph;

    graph.add_node(
        "left_relu",
        OpType::ReLU,
        {"input"},
        "left"
    );

    graph.add_node(
        "right_relu",
        OpType::ReLU,
        {"input"},
        "right"
    );

    graph.add_node(
        "join_add",
        OpType::Add,
        {"left", "right"},
        "joined"
    );

    graph.add_node(
        "out_relu",
        OpType::ReLU,
        {"joined"},
        "output"
    );

    ExecutionPlan plan = graph.compile("input", {2, 3}, "output");

    const auto& schedule = plan.schedule_infos();

    if(schedule.size() != 4){
        throw std::runtime_error("scheduler info count mismatch");
    }

    if(schedule[0].dependency_count != 0 ||
       schedule[1].dependency_count != 0 ||
       schedule[2].dependency_count != 2 ||
       schedule[3].dependency_count != 1){
        throw std::runtime_error("scheduler dependency count mismatch");
    }

    assert_consumers(schedule[0], {2}, "left_relu");
    assert_consumers(schedule[1], {2}, "right_relu");
    assert_consumers(schedule[2], {3}, "join_add");
    assert_consumers(schedule[3], {}, "out_relu");
}

static void test_graph_dump_scheduler_plan(){
    Graph graph;

    graph.add_node("left_relu", OpType::ReLU, {"input"}, "left");
    graph.add_node("right_relu", OpType::ReLU, {"input"}, "right");
    graph.add_node("join_add", OpType::Add, {"left", "right"}, "joined");
    graph.add_node("out_relu", OpType::ReLU, {"joined"}, "output");

    ExecutionPlan plan = graph.compile("input", {2, 3}, "output");
    const std::string dump = graph.dump_scheduler_plan(plan);

    if(dump.find("Scheduler plan:") == std::string::npos ||
       dump.find("[0] left_relu: dependencies=0, consumers=[2] join_add") == std::string::npos ||
       dump.find("[1] right_relu: dependencies=0, consumers=[2] join_add") == std::string::npos ||
       dump.find("[2] join_add: dependencies=2, consumers=[3] out_relu") == std::string::npos ||
       dump.find("[3] out_relu: dependencies=1, consumers=<none>") == std::string::npos){
        throw std::runtime_error("SchedulerPlan dump is missing expected information");
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
        run_graph(graph, plan, Tensor({1, 1}, {1.0f}));
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
        run_graph(graph_b, plan, Tensor({1, 1}, {1.0f}));
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
        run_graph(graph, plan, Tensor({1, 3}));
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

    Tensor y = run_graph(graph, plan, x);

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

static void test_graph_dump_memory_plan(){
    Graph graph;

    graph.set_tensor("weight", Tensor({3, 2}));
    graph.set_tensor("bias", Tensor({2}));

    graph.add_node(
        "linear1",
        OpType::Linear,
        {"input", "weight", "bias"},
        "hidden"
    );

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {"hidden"},
        "output"
    );

    ExecutionPlan plan = graph.compile(
        "input",
        {4, 3},
        "output"
    );

    graph.dump_memory_plan(plan);
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

    bool caught_duplicate_name = false;

    try{
        registry.register_operator(
            static_cast<OpType>(1000),
            OperatorDefinition{
                "ReLU",
                1,
                registry_dummy_execute,
                registry_dummy_infer_shape
            }
        );
    }catch(const std::runtime_error&){
        caught_duplicate_name = true;
    }

    if(!caught_duplicate_name){
        throw std::runtime_error(
            "OperatorRegistry accepted a duplicate operator name"
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

static void test_model_loader(){
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "tinyinfer_model_loader_test";

    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const std::filesystem::path weights_path = directory / "weights.bin";

    {
        std::ofstream weights(
            weights_path,
            std::ios::binary
        );

        const float data[] = {
            2.0f, 3.0f,
            1.0f
        };

        weights.write(
            reinterpret_cast<const char*>(data),
            sizeof(data)
        );
    }

    const std::filesystem::path manifest_path = directory / "model.ti";

    {
        std::ofstream manifest(manifest_path);

        manifest <<
            "TINYINFER_MODEL 1\n"
            "weights weights.bin\n"
            "input input 2 1 2\n"
            "output output\n"
            "tensor weight f32 2 2 1 0 8\n"
            "tensor bias f32 1 1 8 4\n"
            "node linear1 Linear 3 "
            "input weight bias output\n"
            "end\n";
    }

    std::unique_ptr<LoadedModel> model = ModelLoader::load(manifest_path);

    Tensor input({1, 2}, {
        4.0f,
        5.0f
    });

    Tensor output = model->run(input);

    assert_shape(output.shape(), {1, 1});
    assert_close(output.at({0, 0}), 24.0f);

    const auto& order =
        model->plan().execution_order();

    if(order.size() != 1 || order[0] != "linear1"){
        throw std::runtime_error(
            "Loaded model execution order mismatch"
        );
    }

    std::filesystem::remove_all(directory);
}

static void test_model_writer_round_trip(){
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "tinyinfer_model_writer_round_trip";

    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const std::filesystem::path manifest_path = directory / "model.ti";

    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1, 2};
    package.output_name = "output";

    package.tensors.push_back(NamedTensor{
        "weight",
        Tensor({2, 1}, {2.0f, 3.0f})
    });

    package.tensors.push_back(NamedTensor{
        "bias",
        Tensor({1}, {1.0f})
    });

    package.nodes.push_back(NodeMetadata{
        "linear1",
        "Linear",
        {"input", "weight", "bias"},
        "output"
    });

    ModelWriter::save(package, manifest_path);

    std::unique_ptr<LoadedModel> model = ModelLoader::load(manifest_path);

    Tensor input({1, 2}, {4.0f, 5.0f});
    Tensor output = model->run(input);

    assert_shape(output.shape(), {1, 1});
    assert_close(output.at({0, 0}), 24.0f);

    const auto& order = model->plan().execution_order();

    if(order.size() != 1 || order[0] != "linear1"){
        throw std::runtime_error("round-trip execution order mismatch");
    }

    std::filesystem::remove_all(directory);
}

static void test_model_writer_creates_files(){
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "tinyinfer_model_writer_files";

    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const std::filesystem::path manifest_path = directory / "model.ti";
    const std::filesystem::path weights_path = directory / "weights.bin";

    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1, 1};
    package.output_name = "output";

    package.tensors.push_back(NamedTensor{
        "weight",
        Tensor({1, 1}, {2.0f})
    });

    package.tensors.push_back(NamedTensor{
        "bias",
        Tensor({1}, {1.0f})
    });

    package.nodes.push_back(NodeMetadata{
        "linear1",
        "Linear",
        {"input", "weight", "bias"},
        "output"
    });

    ModelWriter::save(package, manifest_path);

    if(!std::filesystem::exists(manifest_path)){
        throw std::runtime_error("model manifest was not created");
    }

    if(!std::filesystem::exists(weights_path)){
        throw std::runtime_error("weights file was not created");
    }

    if(std::filesystem::file_size(weights_path) != 8){
        throw std::runtime_error("weights file size mismatch");
    }

    std::filesystem::remove_all(directory);
}

static void test_model_loader_unknown_operator(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_unknown_operator");
    const std::filesystem::path manifest_path = dir / "model.ti";
    const std::filesystem::path weights_path = dir / "weights.bin";

    write_float_weights(weights_path, {});

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights weights.bin\n"
        "input input 2 1 1\n"
        "output output\n"
        "node bad MysteryOp 1 input output\n"
        "end\n"
    );

    expect_model_load_failure(manifest_path, "unknown operator model");
    std::filesystem::remove_all(dir);
}

static void test_model_loader_short_weights_file(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_short_weights");
    const std::filesystem::path manifest_path = dir / "model.ti";
    const std::filesystem::path weights_path = dir / "weights.bin";

    write_float_weights(weights_path, {2.0f});

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights weights.bin\n"
        "input input 2 1 2\n"
        "output output\n"
        "tensor weight f32 2 2 1 0 8\n"
        "tensor bias f32 1 1 8 4\n"
        "node linear1 Linear 3 input weight bias output\n"
        "end\n"
    );

    expect_model_load_failure(manifest_path, "short weights file model");
    std::filesystem::remove_all(dir);
}

static void test_model_loader_tensor_byte_size_mismatch(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_tensor_byte_size");
    const std::filesystem::path manifest_path = dir / "model.ti";
    const std::filesystem::path weights_path = dir / "weights.bin";

    write_float_weights(weights_path, {2.0f, 3.0f, 1.0f});

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights weights.bin\n"
        "input input 2 1 2\n"
        "output output\n"
        "tensor weight f32 2 2 1 0 4\n"
        "tensor bias f32 1 1 8 4\n"
        "node linear1 Linear 3 input weight bias output\n"
        "end\n"
    );

    expect_model_load_failure(manifest_path, "tensor byte size mismatch model");
    std::filesystem::remove_all(dir);
}

static void test_model_loader_missing_end(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_missing_end");
    const std::filesystem::path manifest_path = dir / "model.ti";
    const std::filesystem::path weights_path = dir / "weights.bin";

    write_float_weights(weights_path, {2.0f, 3.0f, 1.0f});

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights weights.bin\n"
        "input input 2 1 2\n"
        "output output\n"
        "tensor weight f32 2 2 1 0 8\n"
        "tensor bias f32 1 1 8 4\n"
        "node linear1 Linear 3 input weight bias output\n"
    );

    expect_model_load_failure(manifest_path, "missing end model");
    std::filesystem::remove_all(dir);
}

static void test_model_loader_content_after_end(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_content_after_end");
    const std::filesystem::path manifest_path = dir / "model.ti";
    const std::filesystem::path weights_path = dir / "weights.bin";

    write_float_weights(weights_path, {});

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights weights.bin\n"
        "input input 2 1 1\n"
        "output output\n"
        "node relu1 ReLU 1 input output\n"
        "end\n"
        "node extra ReLU 1 input other\n"
    );

    expect_model_load_failure(manifest_path, "content after end model");
    std::filesystem::remove_all(dir);
}

static void test_model_loader_node_input_count_mismatch(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_node_input_count");
    const std::filesystem::path manifest_path = dir / "model.ti";
    const std::filesystem::path weights_path = dir / "weights.bin";

    write_float_weights(weights_path, {2.0f, 3.0f, 1.0f});

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights weights.bin\n"
        "input input 2 1 2\n"
        "output output\n"
        "tensor weight f32 2 2 1 0 8\n"
        "tensor bias f32 1 1 8 4\n"
        "node linear1 Linear 2 input weight bias output\n"
        "end\n"
    );

    expect_model_load_failure(manifest_path, "node input count mismatch model");
    std::filesystem::remove_all(dir);
}

static void test_model_loader_shape_mismatch(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_shape_mismatch");
    const std::filesystem::path manifest_path = dir / "model.ti";
    const std::filesystem::path weights_path = dir / "weights.bin";

    write_float_weights(weights_path, {1.0f, 2.0f, 3.0f, 0.0f});

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights weights.bin\n"
        "input input 2 1 2\n"
        "output output\n"
        "tensor weight f32 2 3 1 0 12\n"
        "tensor bias f32 1 1 12 4\n"
        "node linear1 Linear 3 input weight bias output\n"
        "end\n"
    );

    expect_model_load_failure(manifest_path, "shape mismatch model");
    std::filesystem::remove_all(dir);
}

static void test_model_writer_duplicate_tensor(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_writer_duplicate_tensor");
    const std::filesystem::path manifest_path = dir / "model.ti";

    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1, 1};
    package.output_name = "output";

    package.tensors.push_back(NamedTensor{"weight", Tensor({1, 1}, {1.0f})});
    package.tensors.push_back(NamedTensor{"weight", Tensor({1, 1}, {2.0f})});

    package.nodes.push_back(NodeMetadata{"relu1", "ReLU", {"input"}, "output"});

    bool caught = false;

    try{
        ModelWriter::save(package, manifest_path);
    }catch(const std::exception&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error("ModelWriter accepted duplicate tensor names");
    }

    std::filesystem::remove_all(dir);
}

static ModelPackage valid_writer_package(){
    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1, 1};
    package.output_name = "output";

    package.tensors.push_back(NamedTensor{
        "weight",
        Tensor({1, 1}, {1.0f})
    });

    package.nodes.push_back(NodeMetadata{
        "relu1",
        "ReLU",
        {"input"},
        "output"
    });

    return package;
}

static void test_model_writer_invalid_names(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_writer_invalid_names");
    const std::filesystem::path manifest_path = dir / "model.ti";

    {
        ModelPackage package = valid_writer_package();
        package.input_name.clear();
        expect_model_write_failure(package, manifest_path, "empty input name");
    }

    {
        ModelPackage package = valid_writer_package();
        package.output_name.clear();
        expect_model_write_failure(package, manifest_path, "empty output name");
    }

    {
        ModelPackage package = valid_writer_package();
        package.tensors[0].name.clear();
        expect_model_write_failure(package, manifest_path, "empty tensor name");
    }

    {
        ModelPackage package = valid_writer_package();
        package.nodes[0].name.clear();
        expect_model_write_failure(package, manifest_path, "empty node name");
    }

    {
        ModelPackage package = valid_writer_package();
        package.nodes[0].op_name.clear();
        expect_model_write_failure(package, manifest_path, "empty node op name");
    }

    {
        ModelPackage package = valid_writer_package();
        package.nodes[0].inputs[0].clear();
        expect_model_write_failure(package, manifest_path, "empty node input name");
    }

    {
        ModelPackage package = valid_writer_package();
        package.nodes[0].output.clear();
        expect_model_write_failure(package, manifest_path, "empty node output name");
    }

    std::filesystem::remove_all(dir);
}

static void test_model_writer_empty_weights_filename(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_writer_empty_weights_filename");
    const std::filesystem::path manifest_path = dir / "model.ti";

    bool caught = false;

    try{
        ModelWriter::save(valid_writer_package(), manifest_path, "");
    }catch(const std::exception&){
        caught = true;
    }

    if(!caught){
        throw std::runtime_error("ModelWriter accepted an empty weights filename");
    }

    std::filesystem::remove_all(dir);
}

static void test_model_loader_missing_weights_file(){
    const std::filesystem::path dir = reset_temp_dir("tinyinfer_bad_missing_weights");
    const std::filesystem::path manifest_path = dir / "model.ti";

    write_text_file(manifest_path,
        "TINYINFER_MODEL 1\n"
        "weights missing.bin\n"
        "input input 2 1 1\n"
        "output output\n"
        "node relu1 ReLU 1 input output\n"
        "end\n"
    );

    expect_model_load_failure(manifest_path, "missing weights file model");
    std::filesystem::remove_all(dir);
}

int main(){
    test_tensor_shape_overflow();
    test_transpose_2d();
    test_naive_matmul();
    test_matmul_transposed_b();
    test_add_bias();
    test_mlp_forward();
    test_softmax();
    test_sequential_forward();
    test_graph_forward();
    test_graph_run_releases_unused_intermediates();
    test_graph_run_keeps_intermediate_until_last_use();
    test_graph_run_ready_queue_branch_graph();
    test_graph_run_parallel_branch_graph();
    test_graph_run_parallel_rejects_same_pool_worker();
    test_graph_run_parallel_stress();
    test_graph_run_parallel_does_not_wait_for_unrelated_pool_tasks();
    test_graph_run_parallel_failure_cancels_descendants();
    test_graph_compile_topological_order();
    test_execution_plan_reuse();
    test_execution_context_separate_workspaces();
    test_execution_context_parallel_runs();
    test_execution_plan_shapes();
    test_execution_plan_memory_infos();
    test_execution_plan_lifetime_multi_use();
    test_execution_plan_scheduler_infos();
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
    test_graph_dump_memory_plan();
    test_graph_dump_scheduler_plan();
    test_operator_registry();
    test_thread_pool_tasks();
    test_thread_pool_exception();
    test_thread_pool_self_wait();
    test_threadpool_matmul_zero_rows();
    test_blocked_matmul_zero_block();
    test_softmax_empty_features();
    test_model_loader();
    test_model_writer_round_trip();
    test_model_writer_creates_files();
    test_model_loader_unknown_operator();
    test_model_loader_short_weights_file();
    test_model_loader_tensor_byte_size_mismatch();
    test_model_loader_missing_end();
    test_model_loader_content_after_end();
    test_model_loader_node_input_count_mismatch();
    test_model_loader_shape_mismatch();
    test_model_writer_duplicate_tensor();
    test_model_writer_invalid_names();
    test_model_writer_empty_weights_filename();
    test_model_loader_missing_weights_file();

    std::cout << "All tests passed.\n";
    return 0;
}

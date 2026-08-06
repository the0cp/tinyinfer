#include "executor.h"
#include "kernel_registry.h"
#include "model_loader.h"
#include "module.h"
#include "op_kernel.h"
#include "operator_registry.h"
#include "ops.h"
#include "session_state.h"
#include "thread_pool.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace{

using namespace tinyinfer;

static_assert(!std::is_copy_constructible_v<ExecutionFrame>);
static_assert(!std::is_copy_assignable_v<ExecutionFrame>);
static_assert(!std::is_move_constructible_v<ExecutionFrame>);
static_assert(!std::is_move_assignable_v<ExecutionFrame>);

void expect(bool condition, const std::string& message){
    if(!condition){
        throw std::runtime_error("EXPECT failed: " + message);
    }
}

void expect_close(float actual, float expected, float tolerance = 1e-4f){
    if(std::fabs(actual - expected) > tolerance){
        throw std::runtime_error(
            "EXPECT_CLOSE failed: actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected)
        );
    }
}

void expect_tensor(const Tensor& tensor, const Shape& shape, const std::vector<float>& values){
    expect(tensor.shape() == shape, "tensor shape mismatch");
    expect(tensor.numel() == values.size(), "tensor element count mismatch");

    for(size_t i = 0; i < values.size(); i++){
        expect_close(tensor.data()[i], values[i]);
    }
}

template<typename Function>
void expect_throw(Function&& function, const std::string& needle){
    try{
        function();
    }catch(const std::exception& error){
        expect(
            std::string(error.what()).find(needle) != std::string::npos,
            "exception did not contain '" + needle + "': " + error.what()
        );
        return;
    }

    throw std::runtime_error("Expected exception containing: " + needle);
}

std::filesystem::path reset_temp_dir(const std::string& name){
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

void write_text(const std::filesystem::path& path, const std::string& content){
    std::ofstream file(path);
    file << content;
    expect(static_cast<bool>(file), "write text file");
}

void write_floats(const std::filesystem::path& path, const std::vector<float>& values){
    std::ofstream file(path, std::ios::binary);

    if(!values.empty()){
        file.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float))
        );
    }

    expect(static_cast<bool>(file), "write float file");
}

Graph make_relu_graph(){
    Graph graph;
    graph.add_node("relu", OpType::ReLU, {"input"}, "output");
    return graph;
}

SessionState build_state(Graph graph, const KernelRegistry& kernels){
    OperatorRegistry operators;
    graph.resolve("input", {1, 2}, "output", operators);
    return SessionState::build(std::move(graph), operators, kernels);
}

void test_tensor_basics(){
    Tensor tensor({2, 3});
    expect(tensor.dim() == 2, "tensor rank");
    expect(tensor.numel() == 6, "tensor numel");
    expect(tensor.strides() == std::vector<size_t>({3, 1}), "row-major strides");

    tensor.at({1, 2}) = 7.0f;
    expect_close(tensor.data()[5], 7.0f);
    tensor.fill(2.0f);
    expect_close(tensor.at({0, 1}), 2.0f);

    expect_throw([&](){ (void)tensor.at({2, 0}); }, "out of range");
    expect_throw([&](){ (void)tensor.at({0}); }, "dim mismatch");
    expect_throw([](){ Tensor({2, 2}, {1.0f}); }, "does not match");
    expect_throw([](){ Tensor({std::numeric_limits<size_t>::max(), 2}); }, "overflows");
}

void test_add_relu_and_transpose(){
    Tensor a({2, 2}, {-1.0f, 2.0f, 3.0f, -4.0f});
    Tensor b({2, 2}, {2.0f, 3.0f, 4.0f, 5.0f});
    expect_tensor(add(a, b), {2, 2}, {1.0f, 5.0f, 7.0f, 1.0f});
    expect_tensor(relu(a), {2, 2}, {0.0f, 2.0f, 3.0f, 0.0f});
    expect_tensor(transpose_2d(a), {2, 2}, {-1.0f, 3.0f, 2.0f, -4.0f});
    expect_throw([&](){ (void)add(a, Tensor({1}, {1.0f})); }, "shape mismatch");
    expect_throw([](){ (void)transpose_2d(Tensor({1})); }, "2D");
}

void verify_matmul_result(const Tensor& result){
    expect_tensor(result, {2, 2}, {22.0f, 28.0f, 49.0f, 64.0f});
}

void test_matmul_implementations(){
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {1, 2, 3, 4, 5, 6});

    verify_matmul_result(naive_matmul(a, b));
    verify_matmul_result(fast_matmul(a, b));
    verify_matmul_result(blocked_matmul(a, b, 2));
    verify_matmul_result(matmul_transposed_b(a, transpose_2d(b)));
    verify_matmul_result(parallel_matmul(a, b, 8));

    ThreadPool pool(2);
    verify_matmul_result(threadpool_matmul(a, b, pool, 8));

    expect_throw([&](){ (void)blocked_matmul(a, b, 0); }, "non-zero");
    expect_throw(
        [&](){ (void)fast_matmul(a, Tensor({4, 1})); },
        "shape mismatch"
    );

    Tensor zero_rows({0, 3});
    expect(threadpool_matmul(zero_rows, b, pool, 2).shape() == Shape({0, 2}), "zero rows");
}

void test_linear_softmax_and_sequential(){
    Tensor input({1, 2}, {1.0f, 2.0f});
    Tensor weight({2, 3}, {1, 0, -1, 0, 1, 1});
    Tensor bias({3}, {0, 0, 0});

    Tensor logits = linear(input, weight, bias);
    expect_tensor(logits, {1, 3}, {1.0f, 2.0f, 1.0f});

    Tensor probabilities = softmax(logits);
    expect_close(
        probabilities.data()[0] + probabilities.data()[1] + probabilities.data()[2],
        1.0f
    );
    expect(probabilities.data()[1] > probabilities.data()[0], "softmax ordering");
    expect_throw([](){ (void)softmax(Tensor({2, 0})); }, "non-empty");

    Sequential model;
    model.add(std::make_unique<Linear>(weight, bias));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Softmax>());
    expect(model.size() == 3, "Sequential size");
    Tensor result = model.forward(input);
    expect_close(result.data()[0] + result.data()[1] + result.data()[2], 1.0f);
}

void test_operator_registry_schema_only(){
    OperatorRegistry registry;
    const OperatorSchema& linear_schema = registry.get(OpType::Linear);
    expect(linear_schema.name == "Linear", "Linear name");
    expect(linear_schema.input_count == 3, "Linear arity");
    expect(linear_schema.infer_shape != nullptr, "shape inference exists");
    expect(registry.type_from_name("Add") == OpType::Add, "name lookup");
    expect_throw([&](){ (void)registry.type_from_name("Unknown"); }, "Unknown operator");

    expect_throw(
        [&](){
            registry.register_schema(
                static_cast<OpType>(1000),
                OperatorSchema{"ReLU", 1, 1, linear_schema.infer_shape}
            );
        },
        "already registered"
    );
}

void test_kernel_registry(){
    KernelRegistry builtins;
    std::unique_ptr<OpKernel> kernel =
        builtins.create_kernel(Node{"relu", OpType::ReLU, {"input"}, "output"});
    expect(kernel != nullptr, "built-in kernel creation");

    KernelRegistry empty(false);
    expect_throw(
        [&](){
            (void)empty.create_kernel(Node{"relu", OpType::ReLU, {"input"}, "output"});
        },
        "No CPU kernel"
    );
}

void test_thread_pool(){
    ThreadPool pool(4);
    std::atomic<size_t> counter = 0;

    for(size_t i = 0; i < 1000; i++){
        pool.enqueue([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
    }

    pool.wait();
    expect(counter == 1000, "all tasks complete");

    pool.enqueue([](){ throw std::runtime_error("task failed"); });
    expect_throw([&](){ pool.wait(); }, "task failed");

    std::atomic<bool> recovered = false;
    pool.enqueue([&](){ recovered = true; });
    pool.wait();
    expect(recovered, "pool remains reusable after failure");
    expect_throw([&](){ pool.enqueue({}); }, "empty task");
}

void test_thread_pool_self_wait(){
    ThreadPool pool(1);
    pool.enqueue([&](){ pool.wait(); });
    expect_throw([&](){ pool.wait(); }, "cannot wait");
}

void test_parallel_executor_rejects_same_pool_worker(){
    KernelRegistry kernels;
    SessionState state = build_state(make_relu_graph(), kernels);
    ThreadPool pool(1);
    Tensor input({1, 2}, {-1.0f, 2.0f});

    pool.enqueue([&](){
        ExecutionFrame frame(state);
        frame.bind_input(state.execution_plan().input_index(), input);
        ParallelExecutor executor(pool);
        executor.execute(state, frame);
    });

    expect_throw([&](){ pool.wait(); }, "cannot be entered");
}

void test_parallel_executor_does_not_wait_for_unrelated_tasks(){
    KernelRegistry kernels;
    SessionState state = build_state(make_relu_graph(), kernels);
    ThreadPool pool(2);
    std::promise<void> release;
    std::shared_future<void> blocker = release.get_future().share();
    std::atomic<bool> blocker_started = false;

    pool.enqueue([&](){
        blocker_started = true;
        blocker.wait();
    });

    while(!blocker_started.load()){
        std::this_thread::yield();
    }

    Tensor input({1, 2}, {-1.0f, 2.0f});
    ExecutionFrame frame(state);
    frame.bind_input(state.execution_plan().input_index(), input);
    ParallelExecutor executor(pool);

    const auto start = std::chrono::steady_clock::now();
    executor.execute(state, frame);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    expect(elapsed < std::chrono::seconds(1), "executor waited for unrelated pool work");
    expect_tensor(frame.fetch(state.execution_plan().output_index()), {1, 2}, {0.0f, 2.0f});
    release.set_value();
    pool.wait();
}

class ThrowKernel final : public OpKernel{
public:
    explicit ThrowKernel(std::atomic<size_t>& calls) : calls_(calls){}

    void compute(OpKernelContext&) const override{
        calls_.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("intentional kernel failure");
    }

private:
    std::atomic<size_t>& calls_;
};

class CountKernel final : public OpKernel{
public:
    explicit CountKernel(std::atomic<size_t>& calls) : calls_(calls){}

    void compute(OpKernelContext& context) const override{
        calls_.fetch_add(1, std::memory_order_relaxed);
        context.set_output(relu(context.input(0)));
    }

private:
    std::atomic<size_t>& calls_;
};

void test_parallel_failure_cancels_descendants(){
    Graph graph;
    graph.add_node("fail", OpType::ReLU, {"input"}, "hidden");
    graph.add_node("descendant", OpType::ReLU, {"hidden"}, "output");

    OperatorRegistry operators;
    graph.resolve("input", {1, 2}, "output", operators);

    std::atomic<size_t> fail_calls = 0;
    std::atomic<size_t> descendant_calls = 0;
    KernelRegistry kernels(false);
    kernels.register_kernel(OpType::ReLU, [&](const Node& node){
        if(node.name == "fail"){
            return std::unique_ptr<OpKernel>(std::make_unique<ThrowKernel>(fail_calls));
        }
        return std::unique_ptr<OpKernel>(std::make_unique<CountKernel>(descendant_calls));
    });

    SessionState state = SessionState::build(std::move(graph), operators, kernels);
    Tensor input({1, 2}, {-1.0f, 2.0f});
    ExecutionFrame frame(state);
    frame.bind_input(state.execution_plan().input_index(), input);
    ThreadPool pool(2);
    ParallelExecutor executor(pool);

    expect_throw([&](){ executor.execute(state, frame); }, "intentional kernel failure");
    expect(fail_calls == 1, "failing node count");
    expect(descendant_calls == 0, "descendant must not run after producer failure");
}

void test_model_parse_failures(){
    struct Case{
        std::string name;
        std::string manifest;
        std::vector<float> weights;
        std::string expected;
        bool create_weights = true;
    };

    const std::vector<Case> cases{
        {
            "unknown_operator",
            "TINYINFER_MODEL 1\nweights weights.bin\ninput input 2 1 1\noutput output\n"
            "node n Unknown 1 input output\nend\n",
            {},
            "Unknown operator"
        },
        {
            "short_weights",
            "TINYINFER_MODEL 1\nweights weights.bin\ninput input 2 1 2\noutput output\n"
            "tensor w f32 2 2 1 0 8\ntensor b f32 1 1 8 4\n"
            "node n Linear 3 input w b output\nend\n",
            {1.0f},
            "out of bounds"
        },
        {
            "byte_size",
            "TINYINFER_MODEL 1\nweights weights.bin\ninput input 2 1 2\noutput output\n"
            "tensor w f32 2 2 1 0 4\nend\n",
            {1.0f},
            "does not match shape"
        },
        {
            "missing_end",
            "TINYINFER_MODEL 1\nweights weights.bin\ninput input 2 1 1\noutput input\n",
            {},
            "no end"
        },
        {
            "after_end",
            "TINYINFER_MODEL 1\nweights weights.bin\ninput input 2 1 1\noutput input\nend\n"
            "node extra ReLU 1 input out\n",
            {},
            "content found after end"
        },
        {
            "input_count",
            "TINYINFER_MODEL 1\nweights weights.bin\ninput input 2 1 2\noutput output\n"
            "node n Linear 2 input w b output\nend\n",
            {},
            "does not match declaration"
        },
        {
            "missing_weights",
            "TINYINFER_MODEL 1\nweights missing.bin\ninput input 2 1 1\noutput input\nend\n",
            {},
            "Cannot get weights file size",
            false
        }
    };

    for(const Case& test : cases){
        const std::filesystem::path directory = reset_temp_dir("tinyinfer_" + test.name);
        const std::filesystem::path manifest = directory / "model.ti";
        write_text(manifest, test.manifest);

        if(test.create_weights){
            write_floats(directory / "weights.bin", test.weights);
        }

        expect_throw([&](){ (void)ModelLoader::load_graph(manifest); }, test.expected);
        std::filesystem::remove_all(directory);
    }
}

ModelPackage valid_package(){
    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1, 1};
    package.output_name = "output";
    package.tensors.push_back(NamedTensor{"weight", Tensor({1, 1}, {2.0f})});
    package.tensors.push_back(NamedTensor{"bias", Tensor({1}, {1.0f})});
    package.nodes.push_back(NodeMetadata{
        "linear", "Linear", {"input", "weight", "bias"}, "output"
    });
    return package;
}

void test_model_writer_validation(){
    const std::filesystem::path directory = reset_temp_dir("tinyinfer_writer_validation");
    const std::filesystem::path manifest = directory / "model.ti";

    ModelPackage duplicate = valid_package();
    duplicate.tensors.push_back(NamedTensor{"weight", Tensor({1}, {1.0f})});
    expect_throw([&](){ ModelWriter::save(duplicate, manifest); }, "duplicate tensor");

    ModelPackage bad_name = valid_package();
    bad_name.nodes[0].name.clear();
    expect_throw([&](){ ModelWriter::save(bad_name, manifest); }, "cannot be empty");

    ModelPackage package = valid_package();
    expect_throw([&](){ ModelWriter::save(package, manifest, ""); }, "cannot be empty");

    ModelWriter::save(package, manifest);
    expect(std::filesystem::exists(manifest), "manifest created");
    expect(std::filesystem::file_size(directory / "weights.bin") == 8, "weight bytes");
    std::filesystem::remove_all(directory);
}

struct TestCase{
    const char* name;
    void (*run)();
};

}

int main(){
    const std::vector<TestCase> tests{
        {"tensor_basics", test_tensor_basics},
        {"add_relu_and_transpose", test_add_relu_and_transpose},
        {"matmul_implementations", test_matmul_implementations},
        {"linear_softmax_and_sequential", test_linear_softmax_and_sequential},
        {"operator_registry_schema_only", test_operator_registry_schema_only},
        {"kernel_registry", test_kernel_registry},
        {"thread_pool", test_thread_pool},
        {"thread_pool_self_wait", test_thread_pool_self_wait},
        {"parallel_executor_rejects_same_pool_worker", test_parallel_executor_rejects_same_pool_worker},
        {"parallel_executor_does_not_wait_for_unrelated_tasks", test_parallel_executor_does_not_wait_for_unrelated_tasks},
        {"parallel_failure_cancels_descendants", test_parallel_failure_cancels_descendants},
        {"model_parse_failures", test_model_parse_failures},
        {"model_writer_validation", test_model_writer_validation}
    };

    size_t passed = 0;

    for(const TestCase& test : tests){
        try{
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
            passed++;
        }catch(const std::exception& error){
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << "\n";
            return 1;
        }
    }

    std::cout << "Passed " << passed << " foundation tests.\n";
    return 0;
}

#include "execution_frame.h"
#include "executor.h"
#include "graph_optimizer.h"
#include "inference_session.h"
#include "kernel_registry.h"
#include "model_loader.h"
#include "operator_registry.h"
#include "session_state.h"
#include "value_index.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace{

using namespace tinyinfer;

void expect(bool condition, const std::string& message){
    if(!condition){
        throw std::runtime_error("EXPECT failed: " + message);
    }
}

void expect_close(float actual, float expected, float tolerance = 1e-5f){
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
        if(std::string(error.what()).find(needle) == std::string::npos){
            throw std::runtime_error(
                "Exception did not contain '" + needle + "': " + error.what()
            );
        }
        return;
    }

    throw std::runtime_error("Expected exception containing: " + needle);
}

Graph make_linear_relu_graph(){
    Graph graph;
    graph.set_tensor("weight", Tensor({2, 2}, {1.0f, -1.0f, 1.0f, 1.0f}));
    graph.set_tensor("bias", Tensor({2}, {0.0f, 0.0f}));

    // Deliberately insert the consumer first. resolve() must derive topology from data dependencies.
    graph.add_node("relu", OpType::ReLU, {"hidden"}, "output");
    graph.add_node("linear", OpType::Linear, {"input", "weight", "bias"}, "hidden");
    return graph;
}

std::unique_ptr<InferenceSession> make_session(SessionOptions options = {}){
    auto session = std::make_unique<InferenceSession>(options);
    session->load(make_linear_relu_graph(), "input", {1, 2}, "output");
    session->initialize();
    return session;
}

void test_value_name_index_map(){
    ValueNameIndexMap map;
    const ValueIndex a = map.add("a");
    const ValueIndex b = map.add("b");

    expect(a == 0 && b == 1, "value indices must be dense and deterministic");
    expect(map.get("b") == b, "name to index lookup");
    expect(map.name(a) == "a", "index to name lookup");
    expect_throw([&](){ map.add("a"); }, "already indexed");
}

void test_graph_resolve_and_topology(){
    OperatorRegistry registry;
    Graph graph = make_linear_relu_graph();
    graph.resolve("input", {1, 2}, "output", registry);

    expect(graph.is_resolved(), "graph must be resolved");
    expect(graph.topological_order().size() == 2, "topological order size");
    expect(graph.nodes().at(graph.topological_order()[0]).name == "linear", "producer first");
    expect(graph.nodes().at(graph.topological_order()[1]).name == "relu", "consumer second");
    expect(graph.shape("output") == Shape({1, 2}), "shape inference output");

    graph.add_node("extra", OpType::ReLU, {"output"}, "extra_output");
    expect(!graph.is_resolved(), "mutation must invalidate resolved state");
}

void test_graph_validation(){
    OperatorRegistry registry;

    Graph missing;
    missing.add_node("relu", OpType::ReLU, {"missing"}, "output");
    expect_throw(
        [&](){ missing.resolve("input", {1, 2}, "output", registry); },
        "missing value"
    );

    Graph cycle;
    cycle.add_node("a", OpType::ReLU, {"b_out"}, "a_out");
    cycle.add_node("b", OpType::ReLU, {"a_out"}, "b_out");
    expect_throw(
        [&](){ cycle.resolve("input", {1, 2}, "a_out", registry); },
        "cycle detected"
    );

    Graph duplicate;
    duplicate.add_node("a", OpType::ReLU, {"input"}, "same");
    duplicate.add_node("b", OpType::ReLU, {"input"}, "same");
    expect_throw(
        [&](){ duplicate.resolve("input", {1, 2}, "same", registry); },
        "more than one producer"
    );
}

void test_sequential_session(){
    std::unique_ptr<InferenceSession> session = make_session();
    Tensor output = session->run(Tensor({1, 2}, {1.0f, 2.0f}));
    expect_tensor(output, {1, 2}, {3.0f, 1.0f});

    const ExecutionPlan& plan = session->plan();
    expect(plan.node_count() == 2, "compiled node count");
    expect(plan.value_index("hidden") != invalid_value_index, "compiled ValueIndex");
    expect(plan.nodes()[0].name == "linear", "plan uses resolved topology");
    expect(plan.dump().find("linear") != std::string::npos, "plan dump");
}

void test_session_owns_compiled_state(){
    std::unique_ptr<InferenceSession> session;

    {
        Graph graph = make_linear_relu_graph();
        session = std::make_unique<InferenceSession>();
        session->load(std::move(graph), "input", {1, 2}, "output");
        session->initialize();
    }

    Tensor output = session->run(Tensor({1, 2}, {2.0f, 1.0f}));
    expect_tensor(output, {1, 2}, {3.0f, 0.0f});
}

void test_execution_frame_release(){
    OperatorRegistry operators;
    KernelRegistry kernels;
    Graph graph = make_linear_relu_graph();
    graph.resolve("input", {1, 2}, "output", operators);
    SessionState state = SessionState::build(std::move(graph), operators, kernels);
    ExecutionFrame frame(state);
    Tensor input({1, 2}, {1.0f, 2.0f});
    frame.bind_input(state.execution_plan().input_index(), input);

    const ValueIndex weight = state.execution_plan().value_index("weight");
    expect(
        frame.value(weight).data() == state.graph().constant("weight").data(),
        "initializers must be borrowed from immutable SessionState"
    );
    expect(
        frame.value(state.execution_plan().input_index()).data() == input.data(),
        "the per-run input must be borrowed instead of copied"
    );

    SequentialExecutor executor;
    executor.execute(state, frame);

    const ValueIndex hidden = state.execution_plan().value_index("hidden");
    expect(!frame.has_value(hidden), "dead intermediate must be released");
    expect(frame.has_value(state.execution_plan().output_index()), "graph output must remain live");
}

void test_parallel_session(){
    Graph graph;
    graph.add_node("left", OpType::ReLU, {"input"}, "left_out");
    graph.add_node("right", OpType::ReLU, {"input"}, "right_out");
    graph.add_node("sum", OpType::Add, {"left_out", "right_out"}, "output");

    SessionOptions options;
    options.execution_mode = ExecutionMode::Parallel;
    options.inter_op_threads = 4;

    InferenceSession session(options);
    session.load(std::move(graph), "input", {1, 3}, "output");
    session.initialize();

    Tensor output = session.run(Tensor({1, 3}, {-1.0f, 2.0f, 3.0f}));
    expect_tensor(output, {1, 3}, {0.0f, 4.0f, 6.0f});
    expect(session.plan().nodes()[2].dependency_count == 2, "join node dependency count");
}

void test_concurrent_run(){
    std::unique_ptr<InferenceSession> session = make_session();
    std::atomic<size_t> successes = 0;
    std::vector<std::thread> threads;

    for(size_t i = 0; i < 8; i++){
        threads.emplace_back([&, i](){
            const float x = static_cast<float>(i);
            Tensor output = session->run(Tensor({1, 2}, {x, 2.0f}));
            const float expected0 = x + 2.0f;
            const float expected1 = std::max(0.0f, -x + 2.0f);

            if(std::fabs(output.data()[0] - expected0) < 1e-5f &&
               std::fabs(output.data()[1] - expected1) < 1e-5f){
                successes++;
            }
        });
    }

    for(auto& thread : threads){
        thread.join();
    }

    expect(successes == threads.size(), "same session must support concurrent Run calls");
}

void test_failure_does_not_poison_session(){
    std::unique_ptr<InferenceSession> session = make_session();

    expect_throw(
        [&](){ (void)session->run(Tensor({2, 1}, {1.0f, 2.0f})); },
        "Input shape mismatch"
    );

    Tensor output = session->run(Tensor({1, 2}, {1.0f, 2.0f}));
    expect_tensor(output, {1, 2}, {3.0f, 1.0f});
}

void test_dead_node_elimination(){
    Graph graph = make_linear_relu_graph();
    graph.add_node("dead", OpType::ReLU, {"input"}, "unused");

    InferenceSession session;
    session.pass_manager().emplace_pass<DeadNodeEliminationPass>();
    session.load(std::move(graph), "input", {1, 2}, "output");
    session.initialize();

    expect(session.graph().num_nodes() == 2, "dead node must be removed before planning");
    expect_tensor(session.run(Tensor({1, 2}, {1.0f, 2.0f})), {1, 2}, {3.0f, 1.0f});
}

class InvalidPass final : public GraphPass{
public:
    std::string_view name() const noexcept override{
        return "InvalidPass";
    }

    void run(GraphRewriteContext& graph, const GraphOptimizationContext&) const override{
        std::vector<Node> nodes = graph.nodes();
        nodes.push_back(Node{"bad", OpType::ReLU, {"does_not_exist"}, "bad_output"});
        graph.replace_nodes(std::move(nodes));
    }
};

void test_optimizer_transaction(){
    OperatorRegistry registry;
    Graph graph = make_linear_relu_graph();
    PassManager manager;
    manager.emplace_pass<InvalidPass>();

    expect_throw(
        [&](){
            (void)manager.run(
                graph,
                GraphOptimizationContext{"input", {1, 2}, "output"},
                registry
            );
        },
        "missing value"
    );

    expect(graph.num_nodes() == 2, "failed pass must not mutate the original Graph");
}

void test_empty_graph_output_is_input(){
    InferenceSession session;
    session.load(Graph{}, "input", {1, 2}, "input");
    session.initialize();

    Tensor output = session.run(Tensor({1, 2}, {4.0f, 5.0f}));
    expect_tensor(output, {1, 2}, {4.0f, 5.0f});
}

void test_empty_graph_output_is_initializer(){
    Graph graph;
    graph.set_tensor("answer", Tensor({1}, {42.0f}));

    InferenceSession session;
    session.load(std::move(graph), "input", {1}, "answer");
    session.initialize();

    Tensor output = session.run(Tensor({1}, {0.0f}));
    expect_tensor(output, {1}, {42.0f});
}

void test_model_round_trip(){
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "tinyinfer_architecture_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1, 2};
    package.output_name = "output";
    package.tensors.push_back(NamedTensor{
        "weight",
        Tensor({2, 2}, {1.0f, -1.0f, 1.0f, 1.0f})
    });
    package.tensors.push_back(NamedTensor{"bias", Tensor({2}, {0.0f, 0.0f})});
    package.nodes.push_back(NodeMetadata{
        "linear", "Linear", {"input", "weight", "bias"}, "hidden"
    });
    package.nodes.push_back(NodeMetadata{"relu", "ReLU", {"hidden"}, "output"});

    const std::filesystem::path manifest = directory / "model.ti";
    ModelWriter::save(package, manifest);
    std::unique_ptr<InferenceSession> session = ModelLoader::load(manifest);

    expect(session->metadata().version == 1, "model metadata version");
    expect_tensor(session->run(Tensor({1, 2}, {1.0f, 2.0f})), {1, 2}, {3.0f, 1.0f});
    std::filesystem::remove_all(directory);
}

void test_malformed_model(){
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "tinyinfer_bad_model_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const std::filesystem::path manifest = directory / "model.ti";
    const std::filesystem::path weights = directory / "weights.bin";
    std::ofstream(weights, std::ios::binary).close();

    {
        std::ofstream file(manifest);
        file << "TINYINFER_MODEL 1\n";
        file << "weights weights.bin\n";
        file << "input input 2 1 2\n";
        file << "output output\n";
        file << "node bad UnknownOp 1 input output\n";
        file << "end\n";
    }

    expect_throw([&](){ (void)ModelLoader::load_graph(manifest); }, "Unknown operator");
    std::filesystem::remove_all(directory);
}

struct TestCase{
    const char* name;
    void (*run)();
};

}

int main(){
    const std::vector<TestCase> tests{
        {"value_name_index_map", test_value_name_index_map},
        {"graph_resolve_and_topology", test_graph_resolve_and_topology},
        {"graph_validation", test_graph_validation},
        {"sequential_session", test_sequential_session},
        {"session_owns_compiled_state", test_session_owns_compiled_state},
        {"execution_frame_release", test_execution_frame_release},
        {"parallel_session", test_parallel_session},
        {"concurrent_run", test_concurrent_run},
        {"failure_does_not_poison_session", test_failure_does_not_poison_session},
        {"dead_node_elimination", test_dead_node_elimination},
        {"optimizer_transaction", test_optimizer_transaction},
        {"empty_graph_output_is_input", test_empty_graph_output_is_input},
        {"empty_graph_output_is_initializer", test_empty_graph_output_is_initializer},
        {"model_round_trip", test_model_round_trip},
        {"malformed_model", test_malformed_model}
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

    std::cout << "Passed " << passed << " tests.\n";
    return 0;
}

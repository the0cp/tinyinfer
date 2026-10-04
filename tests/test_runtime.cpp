#include "execution_frame.h"
#include "executor.h"
#include "graph_optimizer.h"
#include "inference_session.h"
#include "kernel_registry.h"
#include "model_loader.h"
#include "operator_registry.h"
#include "profiler.h"
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

void test_sequential_run_profile(){
    std::unique_ptr<InferenceSession> session = make_session();
    RunProfile profile;
    Tensor output = session->run(Tensor({1, 2}, {1.0f, 2.0f}), profile);

    expect_tensor(output, {1, 2}, {3.0f, 1.0f});
    expect(profile.success, "profiled sequential run must succeed");
    expect(profile.error.empty(), "successful profile must not contain an error");
    expect(profile.nodes.size() == 2, "profile must contain one event per node");
    expect(profile.nodes[0].node_name == "linear", "first profile node name");
    expect(profile.nodes[1].node_name == "relu", "second profile node name");
    expect(
        profile.nodes[0].status == NodeProfileStatus::Completed &&
        profile.nodes[1].status == NodeProfileStatus::Completed,
        "every sequential node must complete"
    );
    expect(profile.owned_value_allocations == 2, "one owned frame value per node output");
    expect(profile.cumulative_allocated_bytes == 16, "two float32 tensors with two values each");
    expect(profile.peak_live_bytes == 16, "hidden and output overlap before hidden release");
    expect(profile.live_bytes_at_finish == 8, "only graph output remains live at finish");
    expect(profile.value_events.size() == 3, "two allocations and one release must be recorded");
    expect(
        profile.value_events.back().action == ValueProfileAction::Released,
        "the final value event releases the intermediate"
    );
    expect(profile.summary().find("owned_value_allocations=2") != std::string::npos, "profile summary");
}

void test_profile_failure_before_execution(){
    std::unique_ptr<InferenceSession> session = make_session();
    RunProfile profile;

    expect_throw(
        [&](){ (void)session->run(Tensor({2, 1}, {1.0f, 2.0f}), profile); },
        "Input shape mismatch"
    );

    expect(!profile.success, "failed run profile must report failure");
    expect(profile.error.find("Input shape mismatch") != std::string::npos, "profile failure text");
    expect(profile.owned_value_allocations == 0, "input validation failure allocates no frame output");

    for(const NodeProfile& node : profile.nodes){
        expect(node.status == NodeProfileStatus::NotStarted, "no node starts after input rejection");
    }
}

void test_profile_failure_before_initialization(){
    InferenceSession session;
    RunProfile profile;
    profile.success = true;

    expect_throw(
        [&](){ (void)session.run(Tensor({1}, {1.0f}), profile); },
        "not initialized"
    );

    expect(!profile.success, "pre-initialization failure resets stale profile state");
    expect(profile.error.find("not initialized") != std::string::npos, "precondition failure text");
    expect(profile.nodes.empty(), "there is no compiled plan before initialization");
}

class ProfileThrowKernel final : public OpKernel{
public:
    void compute(OpKernelContext&) const override{
        throw std::runtime_error("profiled kernel failure");
    }
};

void test_profile_kernel_failure(){
    OperatorRegistry operators;
    KernelRegistry kernels(false);
    kernels.register_kernel(
        OpType::ReLU,
        KernelCandidate{
            .name = "test.profile_throw",
            .threading = KernelThreading::Serial,
            .priority = 0,
            .match = [](const KernelSelectionContext&){ return KernelMatch::accept(0); },
            .factory = [](const Node&, const KernelSelectionContext&){
                return std::make_unique<ProfileThrowKernel>();
            }
        }
    );

    Graph graph;
    graph.add_node("fail", OpType::ReLU, {"input"}, "output");
    graph.resolve("input", {1}, "output", operators);
    SessionState state = SessionState::build(std::move(graph), operators, kernels);
    RunProfiler profiler(state.execution_plan());
    ExecutionFrame frame(state, &profiler);
    Tensor input({1}, {1.0f});
    frame.bind_input(state.execution_plan().input_index(), input);
    SequentialExecutor executor;

    expect_throw(
        [&](){ executor.execute(state, frame, &profiler); },
        "profiled kernel failure"
    );

    profiler.finish(false, "profiled kernel failure");
    const RunProfile profile = profiler.snapshot();
    expect(profile.nodes.size() == 1, "failed kernel profile node count");
    expect(profile.nodes[0].status == NodeProfileStatus::Failed, "kernel event must be failed");
    expect(
        profile.nodes[0].error.find("profiled kernel failure") != std::string::npos,
        "kernel event must preserve the failure"
    );
}

void test_profile_trace_export(){
    std::unique_ptr<InferenceSession> session = make_session();
    RunProfile profile;
    (void)session->run(Tensor({1, 2}, {1.0f, 2.0f}), profile);

    const std::string json = profile.to_chrome_trace_json();
    expect(json.find("\"traceEvents\"") != std::string::npos, "trace event array");
    expect(json.find("\"name\":\"linear\"") != std::string::npos, "linear trace event");
    expect(json.find("\"name\":\"relu\"") != std::string::npos, "relu trace event");
    expect(json.find("\"cat\":\"memory\"") != std::string::npos, "memory trace event");
    expect(json.find("released: hidden") != std::string::npos, "released value name");

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "tinyinfer_profile_trace.json";
    profile.save_chrome_trace(path.string());
    expect(std::filesystem::exists(path), "trace file must be created");
    expect(std::filesystem::file_size(path) > 0, "trace file must not be empty");
    std::filesystem::remove(path);
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

void test_parallel_run_profile(){
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

    RunProfile profile;
    Tensor output = session.run(Tensor({1, 3}, {-1.0f, 2.0f, 3.0f}), profile);
    expect_tensor(output, {1, 3}, {0.0f, 4.0f, 6.0f});
    expect(profile.success, "profiled parallel run must succeed");
    expect(profile.nodes.size() == 3, "parallel profile node count");
    expect(profile.owned_value_allocations == 3, "parallel owned value allocations");
    expect(profile.cumulative_allocated_bytes == 36, "three float32 outputs with three values");
    expect(profile.peak_live_bytes == 36, "branch inputs and sum output overlap before release");
    expect(profile.live_bytes_at_finish == 12, "only parallel graph output remains live");

    for(const NodeProfile& node : profile.nodes){
        expect(node.status == NodeProfileStatus::Completed, "every parallel node must complete");
    }
}

void test_concurrent_profiled_runs(){
    std::unique_ptr<InferenceSession> session = make_session();
    std::atomic<size_t> successes = 0;
    std::vector<std::thread> threads;

    for(size_t i = 0; i < 8; i++){
        threads.emplace_back([&, i](){
            RunProfile profile;
            const float x = static_cast<float>(i);
            Tensor output = session->run(Tensor({1, 2}, {x, 2.0f}), profile);

            if(profile.success &&
               profile.nodes.size() == 2 &&
               profile.owned_value_allocations == 2 &&
               std::fabs(output.data()[0] - (x + 2.0f)) < 1e-5f){
                successes++;
            }
        });
    }

    for(auto& thread : threads){
        thread.join();
    }

    expect(successes == threads.size(), "concurrent run profiles must remain isolated");
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
        {"sequential_run_profile", test_sequential_run_profile},
        {"profile_failure_before_execution", test_profile_failure_before_execution},
        {"profile_failure_before_initialization", test_profile_failure_before_initialization},
        {"profile_kernel_failure", test_profile_kernel_failure},
        {"profile_trace_export", test_profile_trace_export},
        {"session_owns_compiled_state", test_session_owns_compiled_state},
        {"execution_frame_release", test_execution_frame_release},
        {"parallel_session", test_parallel_session},
        {"parallel_run_profile", test_parallel_run_profile},
        {"concurrent_run", test_concurrent_run},
        {"concurrent_profiled_runs", test_concurrent_profiled_runs},
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

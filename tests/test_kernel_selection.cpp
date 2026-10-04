#include "execution_frame.h"
#include "executor.h"
#include "inference_session.h"
#include "kernel_registry.h"
#include "operator_registry.h"
#include "ops.h"
#include "session_state.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace{

using namespace tinyinfer;

void expect(bool condition, const std::string& message){
    if(!condition){
        throw std::runtime_error("EXPECT failed: " + message);
    }
}

template<typename Function>
void expect_throw(Function&& function, const std::string& needle){
    try{
        function();
    }catch(const std::exception& error){
        expect(std::string(error.what()).find(needle) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("Expected exception containing: " + needle);
}

class ReluTestKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        relu_out(context.input(0), context.output());
    }
};

KernelCandidate candidate(
    std::string name,
    uint64_t cost,
    KernelThreading threading = KernelThreading::Serial,
    int priority = 0
){
    return KernelCandidate{
        .name = std::move(name),
        .threading = threading,
        .priority = priority,
        .match = [cost](const KernelSelectionContext&){ return KernelMatch::accept(cost); },
        .factory = [](const Node&, const KernelSelectionContext&){
            return std::make_unique<ReluTestKernel>();
        }
    };
}

KernelSelectionContext relu_context(size_t intra_threads = 1){
    return KernelSelectionContext{
        {Shape{2, 4}},
        Shape{2, 4},
        intra_threads
    };
}

Node relu_node(){
    return Node{"relu", OpType::ReLU, {"input"}, "output"};
}

void test_selector_filters_then_ranks_deterministically(){
    KernelRegistry registry(false);
    registry.register_kernel(OpType::ReLU, candidate("serial", 100));
    registry.register_kernel(
        OpType::ReLU,
        candidate("threaded", 10, KernelThreading::IntraOp)
    );

    expect(
        registry.select_kernel(relu_node(), relu_context()).record.kernel_name == "serial",
        "threaded candidate requires an intra-op budget"
    );
    expect(
        registry.select_kernel(relu_node(), relu_context(4)).record.kernel_name ==
            "threaded",
        "lowest compatible cost wins"
    );

    KernelRegistry tie(false);
    tie.register_kernel(OpType::ReLU, candidate("z_kernel", 10, KernelThreading::Serial, 2));
    tie.register_kernel(OpType::ReLU, candidate("a_kernel", 10));
    expect(
        tie.select_kernel(relu_node(), relu_context()).record.kernel_name == "z_kernel",
        "explicit priority resolves equal-cost ties"
    );

    KernelRegistry rejected(false);
    KernelCandidate unsupported = candidate("unsupported", 1);
    unsupported.match = [](const KernelSelectionContext&){
        return KernelMatch::reject("shape is unsupported");
    };
    rejected.register_kernel(OpType::ReLU, std::move(unsupported));
    expect_throw(
        [&](){ (void)rejected.select_kernel(relu_node(), relu_context()); },
        "shape is unsupported"
    );
}

std::vector<float> values(size_t count, float scale){
    std::vector<float> result(count);
    for(size_t index = 0; index < count; index++){
        result[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * scale;
    }
    return result;
}

Graph linear_graph(size_t input_features, size_t output_features){
    Graph graph;
    graph.set_tensor(
        "weight",
        Tensor(
            {input_features, output_features},
            values(input_features * output_features, 0.01f)
        )
    );
    graph.set_tensor("bias", Tensor({output_features}, values(output_features, 0.02f)));
    graph.add_node("linear", OpType::Linear, {"input", "weight", "bias"}, "output");
    return graph;
}

std::unique_ptr<InferenceSession> linear_session(
    size_t batch,
    size_t input_features,
    size_t output_features,
    SessionOptions options
){
    auto session = std::make_unique<InferenceSession>(options);
    session->load(
        linear_graph(input_features, output_features),
        "input",
        {batch, input_features},
        "output"
    );
    session->initialize();
    return session;
}

void test_builtin_selection_is_cached_observable_and_correct(){
    SessionOptions options;
    options.intra_op_threads = 4;

    auto small = linear_session(2, 4, 4, options);
    expect(
        small->plan().nodes().front().kernel_selection.kernel_name == "cpu.linear.serial",
        "small Linear stays serial"
    );
    expect(!small->session_state().requires_intra_op_thread_pool(),
           "serial plan does not create unused workers");

    auto large = linear_session(64, 128, 128, options);
    expect(
        large->plan().nodes().front().kernel_selection.kernel_name ==
            "cpu.linear.threaded",
        "large Linear selects threaded kernel"
    );
    expect(large->session_state().requires_intra_op_thread_pool(),
           "threaded plan requests workers");
    expect(large->plan().dump_kernel_plan().find("lowest estimated cost") !=
               std::string::npos,
           "kernel decision is observable");

    Tensor input({64, 128}, values(64 * 128, 0.03f));
    const Tensor expected = linear(
        input,
        large->graph().constant("weight"),
        large->graph().constant("bias")
    );
    const Tensor actual = large->run(input);
    for(size_t index = 0; index < actual.numel(); index++){
        expect(std::fabs(actual.data()[index] - expected.data()[index]) < 1e-5f,
               "selected kernel numerical result");
    }

    SessionOptions parallel_options;
    parallel_options.execution_mode = ExecutionMode::Parallel;
    parallel_options.inter_op_threads = 4;
    parallel_options.intra_op_threads = 4;
    auto parallel = linear_session(64, 128, 128, parallel_options);
    expect(
        parallel->plan().nodes().front().kernel_selection.kernel_name ==
            "cpu.linear.serial",
        "nested intra-op suppression affects selection"
    );
}

void test_selection_happens_once_during_build(){
    std::atomic<size_t> match_calls{0};
    std::atomic<size_t> factory_calls{0};
    KernelRegistry kernels(false);
    KernelCandidate registration = candidate("counted", 0);
    registration.match = [&](const KernelSelectionContext&){
        match_calls.fetch_add(1, std::memory_order_relaxed);
        return KernelMatch::accept(1);
    };
    registration.factory = [&](const Node&, const KernelSelectionContext& context){
        factory_calls.fetch_add(1, std::memory_order_relaxed);
        expect(context.input_shapes == std::vector<Shape>{{1, 4}},
               "factory receives compiled input shapes");
        return std::make_unique<ReluTestKernel>();
    };
    kernels.register_kernel(OpType::ReLU, std::move(registration));

    Graph graph;
    graph.add_node("relu", OpType::ReLU, {"input"}, "output");
    OperatorRegistry operators;
    graph.resolve("input", {1, 4}, "output", operators);
    SessionState state = SessionState::build(std::move(graph), operators, kernels);

    Tensor input({1, 4}, {-1.0f, 2.0f, -3.0f, 4.0f});
    for(size_t run = 0; run < 2; run++){
        ExecutionFrame frame(state);
        frame.bind_input(state.execution_plan().input_index(), input);
        SequentialExecutor{}.execute(state, frame);
    }
    expect(match_calls.load() == 1, "candidate matched once");
    expect(factory_calls.load() == 1, "kernel constructed once");
}

void test_session_exposes_preinitialization_kernel_registry(){
    Graph graph;
    graph.add_node("relu", OpType::ReLU, {"input"}, "output");

    bool serial_kernel_saw_pool = false;
    KernelCandidate custom = candidate("test.session_relu", 0, KernelThreading::Serial, 1);
    custom.factory = [&](const Node&, const KernelSelectionContext&){
        class PoolProbeKernel final : public OpKernel{
        public:
            explicit PoolProbeKernel(bool& saw_pool) : saw_pool_(saw_pool){}

            void compute(OpKernelContext& context) const override{
                saw_pool_ = context.intra_op_thread_pool() != nullptr;
                relu_out(context.input(0), context.output());
            }

        private:
            bool& saw_pool_;
        };
        return std::make_unique<PoolProbeKernel>(serial_kernel_saw_pool);
    };

    SessionOptions options;
    options.intra_op_threads = 4;
    InferenceSession session(options);
    session.kernel_registry().register_kernel(
        OpType::ReLU,
        std::move(custom)
    );
    session.load(std::move(graph), "input", {1, 4}, "output");
    session.initialize();

    expect(
        session.plan().nodes().front().kernel_selection.kernel_name == "test.session_relu",
        "custom session candidate participates in normal compilation"
    );
    const Tensor output = session.run(Tensor({1, 4}, {-1.0f, 2.0f, -3.0f, 4.0f}));
    expect(output.data()[1] == 2.0f, "custom session candidate executes");
    expect(!serial_kernel_saw_pool, "serial kernels do not receive the intra-op pool");
    expect_throw([&](){ (void)session.kernel_registry(); }, "before InferenceSession initialization");
}

void test_layout_contract_fails_at_the_boundary(){
    Graph relu;
    relu.add_node("relu", OpType::ReLU, {"input"}, "output");
    InferenceSession runtime;
    runtime.load(std::move(relu), "input", {2, 2}, "output");
    runtime.initialize();
    Tensor strided_input(make_cpu_buffer(4 * sizeof(float)), 0, {2, 2}, {1, 2});
    expect_throw(
        [&](){ (void)runtime.run(strided_input); },
        "compiled kernels require contiguous"
    );

    Graph linear;
    linear.set_tensor(
        "weight",
        Tensor(make_cpu_buffer(16 * sizeof(float)), 0, {4, 4}, {1, 4})
    );
    linear.set_tensor("bias", Tensor({4}));
    linear.add_node("linear", OpType::Linear, {"input", "weight", "bias"}, "output");
    InferenceSession compile;
    compile.load(std::move(linear), "input", {2, 4}, "output");
    expect_throw([&](){ compile.initialize(); }, "contiguous initializer 'weight'");
    expect(compile.lifecycle() == SessionLifecycle::Loaded,
           "failed selection does not partially initialize the session");
}

}

int main(){
    const std::vector<std::pair<const char*, void(*)()>> tests{
        {"selector_filters_then_ranks_deterministically",
         test_selector_filters_then_ranks_deterministically},
        {"builtin_selection_is_cached_observable_and_correct",
         test_builtin_selection_is_cached_observable_and_correct},
        {"selection_happens_once_during_build", test_selection_happens_once_during_build},
        {"session_exposes_preinitialization_kernel_registry",
         test_session_exposes_preinitialization_kernel_registry},
        {"layout_contract_fails_at_the_boundary", test_layout_contract_fails_at_the_boundary}
    };

    for(const auto& [name, test] : tests){
        try{
            test();
            std::cout << "[PASS] " << name << "\n";
        }catch(const std::exception& error){
            std::cerr << "[FAIL] " << name << ": " << error.what() << "\n";
            return 1;
        }
    }
    std::cout << "Passed " << tests.size() << " tests.\n";
}

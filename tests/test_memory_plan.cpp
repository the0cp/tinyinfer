#include "execution_frame.h"
#include "executor.h"
#include "inference_session.h"
#include "kernel_registry.h"
#include "memory_plan.h"
#include "operator_registry.h"
#include "ops.h"
#include "session_state.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
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
    throw std::runtime_error("Expected exception: " + needle);
}

Graph make_relu_chain(size_t node_count){
    Graph graph;
    std::string input = "input";

    for(size_t i = 0; i < node_count; i++){
        const std::string output = "v" + std::to_string(i);
        graph.add_node("relu_" + std::to_string(i), OpType::ReLU, {input}, output);
        input = output;
    }

    return graph;
}

std::unique_ptr<InferenceSession> make_chain_session(
    size_t node_count,
    SessionOptions options = {}
){
    auto session = std::make_unique<InferenceSession>(options);
    session->load(
        make_relu_chain(node_count),
        "input",
        {1, 4},
        "v" + std::to_string(node_count - 1)
    );
    session->initialize();
    return session;
}

void expect_result(const Tensor& tensor){
    const std::vector<float> expected{0.0f, 2.0f, 0.0f, 4.0f};
    expect(tensor.shape() == Shape({1, 4}), "output shape");

    for(size_t i = 0; i < expected.size(); i++){
        expect(std::fabs(tensor.data()[i] - expected[i]) < 1e-6f, "output value");
    }
}

void test_sequential_reuse_and_inclusive_lifetimes(){
    std::unique_ptr<InferenceSession> session = make_chain_session(4);
    const ExecutionPlan& execution = session->plan();
    const MemoryPlan& memory = session->session_state().memory_plan();

    const ValueAllocation& v0 = memory.allocation(execution.value_index("v0"));
    const ValueAllocation& v1 = memory.allocation(execution.value_index("v1"));
    const ValueAllocation& v2 = memory.allocation(execution.value_index("v2"));
    const ValueAllocation& output = memory.allocation(execution.value_index("v3"));

    expect(memory.policy() == MemoryPlanningPolicy::SequentialReuse, "sequential policy");
    expect(memory.planned_value_count() == 4, "all intermediates are planned");
    expect(memory.buffer_count() == 3, "four values use three physical blocks");
    expect(memory.reuse_count() == 1, "one block reuse");
    expect(memory.arena_bytes() == 3 * 4 * sizeof(float), "arena byte count");

    expect(v0.buffer != v1.buffer, "producer and same-node consumer output overlap");
    expect(v0.buffer == v2.buffer, "expired v0 block is reused by v2");
    expect(output.buffer != v0.buffer && output.buffer != v1.buffer, "output is dedicated");

    for(const ValueAllocation* allocation : {&v0, &v1, &v2, &output}){
        expect(allocation->byte_offset % allocation->alignment == 0, "aligned offset");
    }
}

void test_plan_is_deterministic(){
    std::unique_ptr<InferenceSession> first = make_chain_session(6);
    std::unique_ptr<InferenceSession> second = make_chain_session(6);

    const std::string first_dump = first->session_state().dump_memory_plan();
    const std::string second_dump = second->session_state().dump_memory_plan();
    expect(first_dump == second_dump, "identical graphs must produce identical plans");
    expect(first_dump.find("v0[0,1]") != std::string::npos, "dump includes lifetime");
}

void test_planned_and_unplanned_results_match(){
    SessionOptions unplanned_options;
    unplanned_options.enable_memory_planning = false;

    std::unique_ptr<InferenceSession> planned = make_chain_session(4);
    std::unique_ptr<InferenceSession> unplanned = make_chain_session(4, unplanned_options);
    Tensor input({1, 4}, {-1.0f, 2.0f, -3.0f, 4.0f});

    RunProfile planned_profile;
    RunProfile unplanned_profile;
    Tensor planned_output = planned->run(input, planned_profile);
    Tensor unplanned_output = unplanned->run(input, unplanned_profile);

    expect_result(planned_output);
    expect_result(unplanned_output);
    expect(planned_profile.managed_buffer_allocations == 3, "planned physical blocks");
    expect(planned_profile.managed_allocated_bytes == 3 * 4 * sizeof(float), "planned bytes");
    expect(planned_profile.planned_value_count == 4, "profile planned values");
    expect(planned_profile.memory_reuse_count == 1, "profile reuse count");
    expect(unplanned_profile.managed_buffer_allocations == 4, "one unplanned block per node");
    expect(unplanned_profile.managed_allocated_bytes == 4 * 4 * sizeof(float), "unplanned bytes");
    expect(unplanned_profile.memory_reuse_count == 0, "unplanned has no reuse");
}

void test_parallel_uses_conservative_dedicated_blocks(){
    SessionOptions options;
    options.execution_mode = ExecutionMode::Parallel;
    options.inter_op_threads = 2;

    std::unique_ptr<InferenceSession> session = make_chain_session(4, options);
    const MemoryPlan& memory = session->session_state().memory_plan();

    expect(memory.policy() == MemoryPlanningPolicy::Dedicated, "parallel policy");
    expect(memory.planned_value_count() == 4, "parallel planned values");
    expect(memory.buffer_count() == 4, "parallel values have dedicated blocks");
    expect(memory.reuse_count() == 0, "parallel reuse is disabled until happens-before analysis");
    expect_result(session->run(Tensor({1, 4}, {-1.0f, 2.0f, -3.0f, 4.0f})));
}

void test_outputs_survive_their_run_arena(){
    std::unique_ptr<InferenceSession> session = make_chain_session(4);
    Tensor first = session->run(Tensor({1, 4}, {-1.0f, 2.0f, -3.0f, 4.0f}));
    const Buffer* first_buffer = first.buffer().get();
    Tensor second = session->run(Tensor({1, 4}, {5.0f, -6.0f, 7.0f, -8.0f}));

    expect_result(first);
    expect(first_buffer != second.buffer().get(), "live outputs from different runs do not alias");
    expect(second.data()[0] == 5.0f && second.data()[2] == 7.0f, "second output values");
}

void test_concurrent_runs_have_private_arenas(){
    std::unique_ptr<InferenceSession> session = make_chain_session(4);
    constexpr size_t run_count = 8;
    std::vector<Tensor> outputs;
    outputs.reserve(run_count);
    for(size_t i = 0; i < run_count; i++){
        outputs.emplace_back(Shape{1, 4});
    }

    std::vector<std::thread> threads;
    for(size_t i = 0; i < run_count; i++){
        threads.emplace_back([&, i](){
            const float value = static_cast<float>(i + 1);
            outputs[i] = session->run(Tensor({1, 4}, {value, -1.0f, value, -1.0f}));
        });
    }
    for(std::thread& thread : threads){
        thread.join();
    }

    for(size_t i = 0; i < run_count; i++){
        expect(outputs[i].data()[0] == static_cast<float>(i + 1), "concurrent output value");
        for(size_t j = i + 1; j < run_count; j++){
            expect(outputs[i].buffer().get() != outputs[j].buffer().get(), "per-run output storage");
        }
    }
}

void test_zero_byte_buffers_are_valid(){
    InferenceSession session;
    session.load(make_relu_chain(2), "input", {1, 0}, "v1");
    session.initialize();

    Tensor output = session.run(Tensor({1, 0}, {}));
    const MemoryPlan& memory = session.session_state().memory_plan();
    expect(output.numel() == 0, "zero-element result");
    expect(memory.buffer_count() == 2, "zero-byte lifetimes still receive Buffer objects");
    expect(memory.arena_bytes() == 0, "zero-byte arena size");
}

class LegacyReluKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        context.set_output(relu(context.input(0)));
    }
};

void test_legacy_set_output_is_a_correctness_fallback(){
    OperatorRegistry operators;
    KernelRegistry kernels(false);
    kernels.register_kernel(
        OpType::ReLU,
        [](const Node&){ return std::make_unique<LegacyReluKernel>(); }
    );

    Graph graph = make_relu_chain(1);
    graph.resolve("input", {1, 4}, "v0", operators);
    SessionState state = SessionState::build(
        std::move(graph),
        operators,
        kernels,
        MemoryPlanningPolicy::SequentialReuse
    );

    RunProfiler profiler(state.execution_plan());
    ExecutionFrame frame(state, &profiler);
    Tensor input({1, 4}, {-1.0f, 2.0f, -3.0f, 4.0f});
    frame.bind_input(state.execution_plan().input_index(), input);
    SequentialExecutor executor;
    executor.execute(state, frame, &profiler);
    profiler.finish(true);

    expect_result(frame.fetch(state.execution_plan().output_index()));
    const RunProfile profile = profiler.snapshot();
    expect(profile.managed_buffer_allocations == 1, "only the arena block is runtime-managed");
    expect(profile.legacy_output_submissions == 1, "legacy submission is a separate event");
    expect(profile.planned_value_count == 1, "legacy output still lands in planned storage");
}

void test_policy_validation(){
    auto session = make_chain_session(4);
    expect_throw([&](){
        (void)MemoryPlanner::build(session->plan(), static_cast<MemoryPlanningPolicy>(99));
    }, "Unknown memory planning policy");

    ExecutionFrame frame(session->session_state());
    Tensor input({1, 4}, {1, 2, 3, 4});
    frame.bind_input(session->plan().input_index(), input);
    ThreadPool pool(2);
    ParallelExecutor executor(pool);
    expect_throw([&](){ executor.execute(session->session_state(), frame); }, "sequential-reuse");
    expect(!frame.has_value(session->plan().value_index("v0")), "reject before any kernel starts");

    static_assert(!std::is_constructible_v<Arena, MemoryPlan&&>);
    static_assert(!std::is_move_assignable_v<Arena>);
}

void test_output_context_contract(){
    auto session = make_chain_session(4);
    const auto& state = session->session_state();
    ExecutionFrame frame(state);
    const auto& node = state.execution_plan().nodes().front();
    OpKernelContext context(node, frame);
    Tensor& first = context.output();
    expect(&context.output() == &first, "output requests are idempotent");
    first.fill(7.0f);
    context.validate_output();
    expect_throw([&](){ context.set_output(Tensor({1, 4})); }, "already");
    first = Tensor({1, 4});
    expect_throw([&](){ context.validate_output(); }, "replaced");

    ExecutionFrame legacy_frame(state);
    OpKernelContext legacy(node, legacy_frame);
    legacy.set_output(Tensor({1, 4}));
    expect_throw([&](){ (void)legacy.output(); }, "mix");
    legacy.validate_output();
}

class ReplacingKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        Tensor& output = context.output();
        output = Tensor(output.shape());
    }
};

void test_executors_validate_kernel_binding(){
    OperatorRegistry operators;
    KernelRegistry kernels(false);
    kernels.register_kernel(OpType::ReLU, [](const Node&){ return std::make_unique<ReplacingKernel>(); });
    Graph graph = make_relu_chain(1);
    graph.resolve("input", {1, 4}, "v0", operators);
    auto state = SessionState::build(std::move(graph), operators, kernels, MemoryPlanningPolicy::Dedicated);
    Tensor input({1, 4});
    ExecutionFrame sequential_frame(state);
    sequential_frame.bind_input(state.execution_plan().input_index(), input);
    SequentialExecutor sequential;
    expect_throw([&](){ sequential.execute(state, sequential_frame); }, "replaced");

    ExecutionFrame parallel_frame(state);
    parallel_frame.bind_input(state.execution_plan().input_index(), input);
    ThreadPool pool(2);
    ParallelExecutor parallel(pool);
    expect_throw([&](){ parallel.execute(state, parallel_frame); }, "replaced");
}

void test_legacy_alias_is_not_an_allocation(){
    for(bool planned : {false, true}){
        SessionOptions options;
        options.enable_memory_planning = planned;
        auto session = make_chain_session(1, options);
        const auto& state = session->session_state();
        RunProfiler profiler(state.execution_plan());
        ExecutionFrame frame(state, &profiler);
        Tensor input({1, 4}, {1, 2, 3, 4});
        frame.bind_input(state.execution_plan().input_index(), input);
        OpKernelContext context(state.execution_plan().nodes().front(), frame);
        context.set_output(input); // shared handle, no new payload allocation
        context.validate_output();
        profiler.finish(true);
        const auto profile = profiler.snapshot();
        expect(profile.managed_buffer_allocations == (planned ? 1u : 0u), "do not infer allocation from submission");
        expect(profile.legacy_output_submissions == 1, "count the submission");
        expect(frame.fetch(state.execution_plan().output_index()).data()[2] == 3, "copy/alias result");
    }
}

void test_out_alias_and_empty_dimensions(){
    Tensor input({1, 2}, {2, 3});
    Tensor weight({2, 2}, {1, 0, 0, 1});
    Tensor bias({2}, {1, 1});
    expect_throw([&](){ linear_out(input, weight, bias, input); }, "overlap");
    expect(input.data()[0] == 2 && input.data()[1] == 3, "reject before clearing input");
    expect_throw([&](){ relu_out(input, input); }, "overlap");
    expect_throw([&](){ add_out(input, input, input); }, "overlap");
    expect_throw([&](){ softmax_out(input, input); }, "overlap");

    auto storage = make_cpu_buffer(6 * sizeof(float));
    Tensor left(storage, 0, {1, 2});
    Tensor partial(storage, sizeof(float), {1, 2});
    Tensor disjoint(storage, 2 * sizeof(float), {1, 2});
    left.fill(3.0f);
    expect_throw([&](){ relu_out(left, partial); }, "overlap");
    relu_out(left, disjoint);
    expect(disjoint.data()[1] == 3.0f, "disjoint views in same buffer are valid");

    Tensor strided(storage, 0, {1, 2}, {4, 2});
    disjoint.fill(9.0f);
    expect_throw([&](){ relu_out(strided, disjoint); }, "contiguous");
    expect(disjoint.data()[0] == 9.0f, "layout rejected before writes");

    Tensor empty_x({2, 0});
    Tensor empty_weight({0, 2});
    Tensor output({2, 2});
    output.fill(99.0f);
    linear_out(empty_x, empty_weight, bias, output);
    for(size_t i = 0; i < output.numel(); i++){
        expect(output.data()[i] == 1.0f, "zero reduction produces bias, not stale storage");
    }
    Tensor zero_columns({2, 0});
    Tensor zero_bias({0});
    linear_out(Tensor({2, 2}), Tensor({2, 0}), zero_bias, zero_columns);
}

void test_mixed_size_reuse(){
    Graph graph;
    graph.set_tensor("wide_weight", Tensor({2, 4}, {1, 0, 1, 0, 0, 1, 0, 1}));
    graph.set_tensor("wide_bias", Tensor({4}));
    graph.set_tensor("small_weight", Tensor({4, 2}, {1, 0, 0, 1, 1, 0, 0, 1}));
    graph.set_tensor("small_bias", Tensor({2}, {1, 1}));
    graph.add_node("wide", OpType::Linear, {"input", "wide_weight", "wide_bias"}, "wide");
    graph.add_node("relu", OpType::ReLU, {"wide"}, "hidden");
    graph.add_node("small", OpType::Linear, {"hidden", "small_weight", "small_bias"}, "small");
    graph.add_node("output", OpType::ReLU, {"small"}, "output");
    InferenceSession session;
    session.load(std::move(graph), "input", {1, 2}, "output");
    session.initialize();
    const auto& memory = session.session_state().memory_plan();
    const auto wide = memory.allocation(session.plan().value_index("wide"));
    const auto small = memory.allocation(session.plan().value_index("small"));
    expect(wide.buffer == small.buffer, "large expired block serves smaller output");
    expect(wide.size_bytes > small.size_bytes, "logical size differs from capacity");
    Tensor output = session.run(Tensor({1, 2}, {2, 3}));
    expect(output.data()[0] == 5 && output.data()[1] == 7, "Linear overwrites reused storage");
}

struct TestCase{
    const char* name;
    void (*run)();
};

}

int main(){
    const std::vector<TestCase> tests{
        {"policy_validation", test_policy_validation},
        {"output_context_contract", test_output_context_contract},
        {"executors_validate_kernel_binding", test_executors_validate_kernel_binding},
        {"legacy_alias_is_not_an_allocation", test_legacy_alias_is_not_an_allocation},
        {"out_alias_and_empty_dimensions", test_out_alias_and_empty_dimensions},
        {"mixed_size_reuse", test_mixed_size_reuse},
        {"sequential_reuse_and_inclusive_lifetimes", test_sequential_reuse_and_inclusive_lifetimes},
        {"plan_is_deterministic", test_plan_is_deterministic},
        {"planned_and_unplanned_results_match", test_planned_and_unplanned_results_match},
        {"parallel_uses_conservative_dedicated_blocks", test_parallel_uses_conservative_dedicated_blocks},
        {"outputs_survive_their_run_arena", test_outputs_survive_their_run_arena},
        {"concurrent_runs_have_private_arenas", test_concurrent_runs_have_private_arenas},
        {"zero_byte_buffers_are_valid", test_zero_byte_buffers_are_valid},
        {"legacy_set_output_is_a_correctness_fallback", test_legacy_set_output_is_a_correctness_fallback}
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

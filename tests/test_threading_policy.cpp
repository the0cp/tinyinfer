#include "inference_session.h"
#include "ops.h"
#include "thread_pool.h"
#include "threading_policy.h"

#include <atomic>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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
        if(std::string(error.what()).find(needle) == std::string::npos){
            throw std::runtime_error(
                "Exception did not contain '" + needle + "': " + error.what()
            );
        }
        return;
    }
    throw std::runtime_error("Expected exception containing: " + needle);
}

void expect_same_tensor(const Tensor& actual, const Tensor& expected){
    expect(actual.shape() == expected.shape(), "tensor shape mismatch");
    for(size_t index = 0; index < actual.numel(); index++){
        if(std::fabs(actual.data()[index] - expected.data()[index]) > 1e-5f){
            throw std::runtime_error("tensor value mismatch at index " + std::to_string(index));
        }
    }
}

void test_threading_plan_resolution(){
    SessionOptions sequential;
    sequential.execution_mode = ExecutionMode::Sequential;
    sequential.inter_op_threads = 8;
    sequential.intra_op_threads = 4;

    const ThreadingPlan sequential_plan = ThreadingPlan::build(sequential, 16);
    expect(sequential_plan.effective_execution_mode() == ExecutionMode::Sequential,
           "sequential mode stays sequential");
    expect(sequential_plan.effective_inter_op_threads() == 1, "sequential inter-op is one");
    expect(sequential_plan.effective_intra_op_threads() == 4, "sequential intra-op request");
    expect(!sequential_plan.nested_parallelism_suppressed(), "nothing is suppressed");

    sequential.intra_op_threads = 0;
    expect(
        ThreadingPlan::build(sequential, 6).effective_intra_op_threads() == 6,
        "zero resolves to hardware concurrency"
    );
    expect(
        ThreadingPlan::build(sequential, 0).effective_intra_op_threads() == 1,
        "unknown hardware concurrency falls back to one"
    );

    SessionOptions parallel;
    parallel.execution_mode = ExecutionMode::Parallel;
    parallel.inter_op_threads = 3;
    parallel.intra_op_threads = 7;
    const ThreadingPlan parallel_plan = ThreadingPlan::build(parallel, 16);
    expect(parallel_plan.effective_inter_op_threads() == 3, "parallel inter-op request");
    expect(parallel_plan.effective_intra_op_threads() == 1, "nested intra-op is disabled");
    expect(parallel_plan.nested_parallelism_suppressed(), "suppression is observable");
    expect(parallel_plan.dump().find("effective_intra_op_threads=1") != std::string::npos,
           "dump reports effective policy");

    parallel.inter_op_threads = 1;
    const ThreadingPlan single_inter = ThreadingPlan::build(parallel, 16);
    expect(single_inter.requested_execution_mode() == ExecutionMode::Parallel,
           "requested mode remains observable");
    expect(single_inter.effective_execution_mode() == ExecutionMode::Sequential,
           "one inter-op worker normalizes to sequential execution");
    expect(single_inter.effective_intra_op_threads() == 7,
           "sequential fallback preserves useful intra-op parallelism");
    expect(!single_inter.nested_parallelism_suppressed(),
           "no nested parallelism exists after normalization");
}

void test_parallel_for_contract(){
    ThreadPool pool(3);
    std::vector<std::atomic<size_t>> visits(10);

    pool.parallel_for(10, 4, [&](size_t begin, size_t end){
        expect(begin < end, "task ranges must be non-empty");
        for(size_t index = begin; index < end; index++){
            visits[index].fetch_add(1, std::memory_order_relaxed);
        }
    });

    for(const auto& count : visits){
        expect(count.load(std::memory_order_relaxed) == 1, "each item visited exactly once");
    }

    expect_throw(
        [&](){
            pool.parallel_for(8, 4, [](size_t begin, size_t){
                if(begin == 0){
                    throw std::runtime_error("group failure");
                }
            });
        },
        "group failure"
    );

    std::atomic<size_t> recovery_count{0};
    pool.parallel_for(17, 3, [&](size_t begin, size_t end){
        recovery_count.fetch_add(end - begin, std::memory_order_relaxed);
    });
    expect(recovery_count.load() == 17, "one failed group does not poison the pool");

    std::atomic<bool> nested_wait_rejected{false};
    pool.enqueue([&](){
        try{
            pool.parallel_for(1, 1, [](size_t, size_t){});
        }catch(const std::logic_error&){
            nested_wait_rejected.store(true, std::memory_order_relaxed);
        }
    });
    pool.wait();
    expect(nested_wait_rejected.load(), "same-pool nested wait is rejected");
}

void test_concurrent_task_groups_are_isolated(){
    ThreadPool pool(4);
    std::atomic<size_t> counts[2]{};
    std::exception_ptr failures[2];
    const size_t work[2]{257, 193};

    std::thread callers[2];
    for(size_t caller = 0; caller < 2; caller++){
        callers[caller] = std::thread([&, caller](){
            try{
                pool.parallel_for(work[caller], 7, [&](size_t begin, size_t end){
                    counts[caller].fetch_add(end - begin, std::memory_order_relaxed);
                });
            }catch(...){
                failures[caller] = std::current_exception();
            }
        });
    }
    for(auto& caller : callers){
        caller.join();
    }
    for(size_t caller = 0; caller < 2; caller++){
        if(failures[caller]){
            std::rethrow_exception(failures[caller]);
        }
        expect(counts[caller].load() == work[caller], "task groups do not share completion state");
    }
}

std::vector<float> make_values(size_t count, float scale){
    std::vector<float> values(count);
    for(size_t index = 0; index < count; index++){
        const int centered = static_cast<int>(index % 13) - 6;
        values[index] = static_cast<float>(centered) * scale;
    }
    return values;
}

Graph make_linear_graph(size_t input_features, size_t output_features){
    Graph graph;
    graph.set_tensor(
        "weight",
        Tensor(
            {input_features, output_features},
            make_values(input_features * output_features, 0.01f)
        )
    );
    graph.set_tensor("bias", Tensor({output_features}, make_values(output_features, 0.02f)));
    graph.add_node("linear", OpType::Linear, {"input", "weight", "bias"}, "output");
    return graph;
}

void test_session_policy_and_concurrent_runs(){
    constexpr size_t batch = 32;
    constexpr size_t input_features = 64;
    constexpr size_t output_features = 48;
    Tensor input({batch, input_features}, make_values(batch * input_features, 0.03f));

    SessionOptions sequential_options;
    sequential_options.intra_op_threads = 4;
    InferenceSession sequential(sequential_options);
    sequential.load(
        make_linear_graph(input_features, output_features),
        "input",
        {batch, input_features},
        "output"
    );
    sequential.initialize();
    expect(sequential.threading_plan().effective_intra_op_threads() == 4,
           "session exposes compiled threading plan");
    expect(sequential.session_state().requires_intra_op_thread_pool(),
           "test graph selects an intra-op kernel");
    const Tensor expected = sequential.run(input);

    std::vector<std::thread> callers;
    std::vector<std::exception_ptr> failures(6);
    for(size_t caller = 0; caller < failures.size(); caller++){
        callers.emplace_back([&, caller](){
            try{
                expect_same_tensor(sequential.run(input), expected);
            }catch(...){
                failures[caller] = std::current_exception();
            }
        });
    }
    for(auto& caller : callers){
        caller.join();
    }
    for(const auto& failure : failures){
        if(failure){
            std::rethrow_exception(failure);
        }
    }

    SessionOptions parallel_options;
    parallel_options.execution_mode = ExecutionMode::Parallel;
    parallel_options.inter_op_threads = 2;
    parallel_options.intra_op_threads = 4;
    InferenceSession parallel(parallel_options);
    parallel.load(
        make_linear_graph(input_features, output_features),
        "input",
        {batch, input_features},
        "output"
    );
    parallel.initialize();
    expect(parallel.threading_plan().effective_intra_op_threads() == 1,
           "parallel graph execution suppresses nested intra-op work");
    expect(parallel.threading_plan().nested_parallelism_suppressed(),
           "suppression is visible to callers");
    expect_same_tensor(parallel.run(input), expected);

    parallel_options.inter_op_threads = 1;
    InferenceSession single_inter(parallel_options);
    single_inter.load(
        make_linear_graph(input_features, output_features),
        "input",
        {batch, input_features},
        "output"
    );
    single_inter.initialize();
    expect(single_inter.threading_plan().effective_execution_mode() == ExecutionMode::Sequential,
           "single inter-op worker uses the sequential executor");
    expect(single_inter.threading_plan().effective_intra_op_threads() == 4,
           "single inter-op worker keeps the requested intra-op budget");
    expect(single_inter.plan().nodes().front().kernel_selection.threading ==
               KernelThreading::IntraOp,
           "normalized plan can select a threaded kernel");
    expect_same_tensor(single_inter.run(input), expected);
}

}

int main(){
    const std::vector<std::pair<const char*, void (*)()>> tests{
        {"threading_plan_resolution", test_threading_plan_resolution},
        {"parallel_for_contract", test_parallel_for_contract},
        {"concurrent_task_groups_are_isolated", test_concurrent_task_groups_are_isolated},
        {"session_policy_and_concurrent_runs", test_session_policy_and_concurrent_runs}
    };

    for(const auto& [name, run] : tests){
        try{
            run();
            std::cout << "[PASS] " << name << "\n";
        }catch(const std::exception& error){
            std::cerr << "[FAIL] " << name << ": " << error.what() << "\n";
            return 1;
        }
    }

    std::cout << "Passed " << tests.size() << " tests.\n";
    return 0;
}

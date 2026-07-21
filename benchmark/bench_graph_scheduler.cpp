#include "execution_context.h"
#include "graph.h"
#include "tensor.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace tinyinfer;

namespace{

using Clock = std::chrono::steady_clock;

struct BenchmarkResult{
    double median_microseconds = 0.0;
    double minimum_microseconds = 0.0;
    double maximum_microseconds = 0.0;
    double checksum = 0.0;
};

Tensor make_input(const Shape& shape){
    Tensor input(shape);

    for(size_t i = 0; i < input.numel(); i++){
        const int pattern = static_cast<int>(i % 29) - 14;
        input.data()[i] = static_cast<float>(pattern) * 0.125f;
    }

    return input;
}

void assert_same_tensor(const Tensor& expected, const Tensor& actual){
    if(expected.shape() != actual.shape()){
        throw std::runtime_error("Benchmark executors produced different output shapes.");
    }

    for(size_t i = 0; i < expected.numel(); i++){
        if(std::fabs(expected.data()[i] - actual.data()[i]) > 1e-5f){
            throw std::runtime_error("Benchmark executors produced different output values.");
        }
    }
}

template<typename Func>
BenchmarkResult measure(
    size_t warmup_runs,
    size_t sample_count,
    size_t runs_per_sample,
    Func&& run_once
){
    if(sample_count == 0 || runs_per_sample == 0){
        throw std::invalid_argument("Benchmark sample counts must be positive.");
    }

    for(size_t i = 0; i < warmup_runs; i++){
        (void)run_once();
    }

    std::vector<double> samples;
    samples.reserve(sample_count);

    double checksum = 0.0;
    size_t invocation = 0;

    for(size_t sample = 0; sample < sample_count; sample++){
        const auto start = Clock::now();

        for(size_t run = 0; run < runs_per_sample; run++){
            Tensor output = run_once();
            const size_t sample_index = (invocation * 7919 + 17) % output.numel();
            checksum += output.data()[sample_index];
            invocation++;
        }

        const auto end = Clock::now();
        const double total_microseconds =
            std::chrono::duration<double, std::micro>(end - start).count();

        samples.push_back(
            total_microseconds / static_cast<double>(runs_per_sample)
        );
    }

    std::sort(samples.begin(), samples.end());

    return BenchmarkResult{
        samples[samples.size() / 2],
        samples.front(),
        samples.back(),
        checksum
    };
}

std::vector<size_t> worker_counts(){
    const unsigned int reported = std::thread::hardware_concurrency();
    const size_t hardware_threads = reported == 0 ? 1 : reported;

    std::vector<size_t> counts{1};

    for(size_t candidate : {2UL, 4UL, 8UL}){
        if(candidate <= hardware_threads){
            counts.push_back(candidate);
        }
    }

    return counts;
}

void print_header(
    const std::string& name,
    const Shape& shape,
    size_t node_count,
    size_t span_nodes,
    size_t sample_count,
    size_t runs_per_sample
){
    const double average_parallelism =
        static_cast<double>(node_count) / static_cast<double>(span_nodes);

    std::cout << "\nCase: " << name << "\n"
              << "  shape: [" << shape[0] << ", " << shape[1] << "]\n"
              << "  work: " << node_count << " nodes\n"
              << "  span: " << span_nodes << " nodes\n"
              << "  work/span: " << std::fixed << std::setprecision(2)
              << average_parallelism << "\n"
              << "  samples: " << sample_count << "\n"
              << "  runs/sample: " << runs_per_sample << "\n\n"
              << std::left << std::setw(18) << "executor"
              << std::right << std::setw(13) << "median us"
              << std::setw(11) << "min us"
              << std::setw(11) << "max us"
              << std::setw(11) << "speedup"
              << std::setw(14) << "checksum" << "\n";
}

void print_result(
    const std::string& executor,
    const BenchmarkResult& result,
    double sequential_microseconds
){
    const double speedup = sequential_microseconds / result.median_microseconds;

    std::cout << std::left << std::setw(18) << executor
              << std::right << std::setw(13) << std::fixed << std::setprecision(2)
              << result.median_microseconds
              << std::setw(11) << result.minimum_microseconds
              << std::setw(11) << result.maximum_microseconds
              << std::setw(10) << speedup << "x"
              << std::setw(14) << std::setprecision(4) << result.checksum
              << "\n";
}

void build_chain(Graph& graph, size_t depth){
    if(depth == 0){
        throw std::invalid_argument("Chain depth must be positive.");
    }

    std::string input_name = "input";

    for(size_t i = 0; i < depth; i++){
        const bool is_last = i + 1 == depth;
        const std::string output_name =
            is_last ? "output" : "chain_" + std::to_string(i);

        graph.add_node(
            "relu_" + std::to_string(i),
            OpType::ReLU,
            {input_name},
            output_name
        );

        input_name = output_name;
    }
}

void build_branch_reduce(Graph& graph, size_t branch_count){
    if(branch_count < 2){
        throw std::invalid_argument("Branch count must be at least two.");
    }

    std::vector<std::string> current_level;
    current_level.reserve(branch_count);

    for(size_t i = 0; i < branch_count; i++){
        const std::string output_name = "branch_" + std::to_string(i);

        graph.add_node(
            "branch_relu_" + std::to_string(i),
            OpType::ReLU,
            {"input"},
            output_name
        );

        current_level.push_back(output_name);
    }

    size_t reduction_level = 0;
    size_t add_index = 0;

    while(current_level.size() > 1){
        std::vector<std::string> next_level;
        next_level.reserve((current_level.size() + 1) / 2);

        for(size_t i = 0; i < current_level.size(); i += 2){
            if(i + 1 == current_level.size()){
                next_level.push_back(current_level[i]);
                continue;
            }

            const bool is_final_add = current_level.size() == 2;
            const std::string output_name = is_final_add
                ? "output"
                : "reduce_" + std::to_string(reduction_level) + "_" +
                    std::to_string(add_index);

            graph.add_node(
                "add_" + std::to_string(add_index),
                OpType::Add,
                {current_level[i], current_level[i + 1]},
                output_name
            );

            next_level.push_back(output_name);
            add_index++;
        }

        current_level = std::move(next_level);
        reduction_level++;
    }

    if(current_level.front() != "output"){
        graph.add_node(
            "final_relu",
            OpType::ReLU,
            {current_level.front()},
            "output"
        );
    }
}

size_t ceil_log2(size_t value){
    size_t result = 0;
    size_t power = 1;

    while(power < value){
        power *= 2;
        result++;
    }

    return result;
}

void benchmark_graph(
    const std::string& name,
    Graph& graph,
    const Shape& shape,
    size_t span_nodes,
    size_t warmup_runs,
    size_t sample_count,
    size_t runs_per_sample
){
    Tensor input = make_input(shape);
    ExecutionPlan plan = graph.compile("input", shape, "output");

    ExecutionContext sequential_context;
    Tensor expected = graph.run(plan, sequential_context, input);

    for(size_t workers : worker_counts()){
        ThreadPool validation_pool(workers);
        ExecutionContext validation_context;
        Tensor actual = graph.run_parallel(plan, validation_context, input, validation_pool);
        assert_same_tensor(expected, actual);
    }

    print_header(
        name,
        shape,
        graph.num_nodes(),
        span_nodes,
        sample_count,
        runs_per_sample
    );

    ExecutionContext sequential_benchmark_context;
    const BenchmarkResult sequential = measure(
        warmup_runs,
        sample_count,
        runs_per_sample,
        [&](){
            return graph.run(plan, sequential_benchmark_context, input);
        }
    );

    print_result("sequential", sequential, sequential.median_microseconds);

    for(size_t workers : worker_counts()){
        ThreadPool pool(workers);
        ExecutionContext parallel_context;

        const BenchmarkResult parallel = measure(
            warmup_runs,
            sample_count,
            runs_per_sample,
            [&](){
                return graph.run_parallel(plan, parallel_context, input, pool);
            }
        );

        print_result(
            "parallel-" + std::to_string(workers),
            parallel,
            sequential.median_microseconds
        );
    }
}

}

int main(){
    const unsigned int reported = std::thread::hardware_concurrency();

    std::cout << "tinyinfer graph scheduler benchmark\n"
              << "hardware_concurrency: " << reported << "\n"
              << "clock: std::chrono::steady_clock\n";

    {
        Graph graph;
        constexpr size_t depth = 16;
        build_chain(graph, depth);

        benchmark_graph(
            "chain-16-small",
            graph,
            {1, 4096},
            depth,
            5,
            7,
            20
        );
    }

    {
        Graph graph;
        constexpr size_t branches = 8;
        build_branch_reduce(graph, branches);

        benchmark_graph(
            "branch-8-small",
            graph,
            {1, 4096},
            1 + ceil_log2(branches),
            5,
            7,
            20
        );
    }

    {
        Graph graph;
        constexpr size_t branches = 8;
        build_branch_reduce(graph, branches);

        benchmark_graph(
            "branch-8-large",
            graph,
            {1, 262144},
            1 + ceil_log2(branches),
            2,
            7,
            2
        );
    }

    return 0;
}

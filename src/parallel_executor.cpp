#include "executor.h"

#include "op_kernel.h"
#include "profiler.h"

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace tinyinfer{

namespace{

struct ParallelRunState{
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<size_t> remaining_dependencies;
    std::vector<size_t> remaining_uses;
    size_t completed_nodes = 0;
    size_t in_flight_tasks = 0;
    bool cancelled = false;
    std::exception_ptr first_failure;
};

std::string exception_message(const std::exception_ptr& failure){
    try{
        std::rethrow_exception(failure);
    }catch(const std::exception& error){
        return error.what();
    }catch(...){
        return "non-standard exception";
    }
}

}

ParallelExecutor::ParallelExecutor(ThreadPool& pool) : pool_(pool){}

void ParallelExecutor::execute(
    const SessionState& session_state,
    ExecutionFrame& frame,
    RunProfiler* profiler
) const{
    if(session_state.memory_plan().policy() == MemoryPlanningPolicy::SequentialReuse){
        throw std::logic_error("ParallelExecutor cannot execute a sequential-reuse MemoryPlan.");
    }

    if(pool_.is_current_worker_thread()){
        throw std::logic_error(
            "ParallelExecutor cannot be entered from a worker of its own ThreadPool."
        );
    }

    const ExecutionPlan& plan = session_state.execution_plan();

    if(plan.nodes().empty()){
        return;
    }

    ParallelRunState state;
    state.remaining_dependencies.reserve(plan.node_count());
    state.remaining_uses.resize(plan.value_count(), 0);

    std::vector<size_t> initial_ready;

    for(size_t position = 0; position < plan.node_count(); position++){
        const size_t dependencies = plan.nodes()[position].dependency_count;
        state.remaining_dependencies.push_back(dependencies);

        if(dependencies == 0){
            initial_ready.push_back(position);
        }
    }

    for(ValueIndex index = 0; static_cast<size_t>(index) < plan.value_count(); index++){
        state.remaining_uses[index] = plan.value_info(index).consumer_count;
    }

    if(initial_ready.empty()){
        throw std::logic_error("ParallelExecutor found no ready node in a non-empty plan.");
    }

    std::function<void(size_t)> schedule;

    schedule = [&](size_t position){
        {
            std::lock_guard<std::mutex> lock(state.mutex);

            if(state.cancelled){
                return;
            }

            state.in_flight_tasks++;
        }

        try{
            if(profiler){
                profiler->node_queued(position);
            }

            pool_.enqueue([&, position](){
                std::vector<ValueIndex> dead_values;
                std::vector<size_t> ready_consumers;
                bool should_run = true;

                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    should_run = !state.cancelled;
                }

                if(should_run){
                    try{
                        const NodeExecutionPlan& node = plan.nodes().at(position);

                        if(profiler){
                            profiler->node_started(position);
                        }

                        OpKernelContext context(node, frame);
                        session_state.kernel(node.kernel_index).compute(context);
                        context.validate_output();

                        {
                            std::lock_guard<std::mutex> lock(state.mutex);

                            if(!state.cancelled){
                                state.completed_nodes++;

                                for(ValueIndex input : node.inputs){
                                    const ValueInfo& info = plan.value_info(input);

                                    if(info.role != ValueRole::Intermediate || info.is_graph_output){
                                        continue;
                                    }

                                    size_t& remaining = state.remaining_uses.at(input);

                                    if(remaining == 0){
                                        throw std::logic_error(
                                            "ParallelExecutor intermediate use count underflow."
                                        );
                                    }

                                    remaining--;

                                    if(remaining == 0){
                                        dead_values.push_back(input);
                                    }
                                }

                                for(size_t consumer : node.consumers){
                                    size_t& remaining = state.remaining_dependencies.at(consumer);

                                    if(remaining == 0){
                                        throw std::logic_error(
                                            "ParallelExecutor dependency count underflow."
                                        );
                                    }

                                    remaining--;

                                    if(remaining == 0){
                                        ready_consumers.push_back(consumer);
                                    }
                                }
                            }
                        }

                        for(ValueIndex value : dead_values){
                            frame.release(value);
                        }

                        if(profiler){
                            profiler->node_completed(position);
                        }

                        for(size_t consumer : ready_consumers){
                            schedule(consumer);
                        }
                    }catch(...){
                        const std::exception_ptr failure = std::current_exception();

                        if(profiler){
                            try{
                                profiler->node_failed(position, exception_message(failure));
                            }catch(...){
                                // Preserve the original execution failure.
                            }
                        }

                        std::lock_guard<std::mutex> lock(state.mutex);

                        if(!state.first_failure){
                            state.first_failure = failure;
                        }

                        state.cancelled = true;
                    }
                }else if(profiler){
                    try{
                        profiler->node_cancelled(position);
                    }catch(...){
                        const std::exception_ptr failure = std::current_exception();
                        std::lock_guard<std::mutex> lock(state.mutex);

                        if(!state.first_failure){
                            state.first_failure = failure;
                        }

                        state.cancelled = true;
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(state.mutex);

                    if(state.in_flight_tasks == 0){
                        if(!state.first_failure){
                            state.first_failure = std::make_exception_ptr(
                                std::logic_error("ParallelExecutor task count underflow.")
                            );
                        }
                        state.cancelled = true;
                    }else{
                        state.in_flight_tasks--;
                    }

                    state.cv.notify_all();
                }
            });
        }catch(...){
            const std::exception_ptr failure = std::current_exception();

            if(profiler){
                try{
                    profiler->node_failed(position, exception_message(failure));
                }catch(...){
                    // Preserve the original enqueue failure.
                }
            }

            std::lock_guard<std::mutex> lock(state.mutex);

            if(state.in_flight_tasks > 0){
                state.in_flight_tasks--;
            }

            if(!state.first_failure){
                state.first_failure = failure;
            }

            state.cancelled = true;
            state.cv.notify_all();
        }
    };

    for(size_t position : initial_ready){
        schedule(position);
    }

    std::exception_ptr failure;
    size_t completed = 0;

    {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.cv.wait(lock, [&](){
            const bool success =
                state.completed_nodes == plan.node_count() && state.in_flight_tasks == 0;
            const bool failed = state.first_failure && state.in_flight_tasks == 0;
            return success || failed;
        });

        failure = state.first_failure;
        completed = state.completed_nodes;
    }

    if(failure){
        try{
            std::rethrow_exception(failure);
        }catch(const std::exception& error){
            throw std::runtime_error(
                "Parallel execution failed after " + std::to_string(completed) +
                " completed nodes: " + error.what()
            );
        }
    }

    if(completed != plan.node_count()){
        throw std::logic_error("ParallelExecutor did not complete every node.");
    }
}

}

#include "threading_policy.h"

#include <algorithm>
#include <sstream>

namespace tinyinfer{

namespace{

size_t resolve_thread_count(size_t requested, size_t hardware_threads) noexcept{
    if(requested != 0){
        return requested;
    }
    return std::max<size_t>(1, hardware_threads);
}

}

ThreadingPlan ThreadingPlan::build(const SessionOptions& options, size_t hardware_threads){
    ThreadingPlan plan;
    plan.requested_execution_mode_ = options.execution_mode;
    plan.requested_inter_op_threads_ = options.inter_op_threads;
    plan.requested_intra_op_threads_ = options.intra_op_threads;

    const size_t resolved_inter = resolve_thread_count(options.inter_op_threads, hardware_threads);
    const size_t resolved_intra = resolve_thread_count(options.intra_op_threads, hardware_threads);

    if(options.execution_mode == ExecutionMode::Parallel && resolved_inter > 1){
        plan.effective_execution_mode_ = ExecutionMode::Parallel;
        plan.effective_inter_op_threads_ = resolved_inter;
        plan.effective_intra_op_threads_ = 1;
        plan.nested_parallelism_suppressed_ = resolved_intra > 1;
    }else{
        plan.effective_execution_mode_ = ExecutionMode::Sequential;
        plan.effective_inter_op_threads_ = 1;
        plan.effective_intra_op_threads_ = resolved_intra;
    }

    return plan;
}

ExecutionMode ThreadingPlan::requested_execution_mode() const noexcept{
    return requested_execution_mode_;
}

ExecutionMode ThreadingPlan::effective_execution_mode() const noexcept{
    return effective_execution_mode_;
}

size_t ThreadingPlan::requested_inter_op_threads() const noexcept{
    return requested_inter_op_threads_;
}

size_t ThreadingPlan::requested_intra_op_threads() const noexcept{
    return requested_intra_op_threads_;
}

size_t ThreadingPlan::effective_inter_op_threads() const noexcept{
    return effective_inter_op_threads_;
}

size_t ThreadingPlan::effective_intra_op_threads() const noexcept{
    return effective_intra_op_threads_;
}

bool ThreadingPlan::nested_parallelism_suppressed() const noexcept{
    return nested_parallelism_suppressed_;
}

std::string ThreadingPlan::dump() const{
    std::ostringstream oss;
    oss << "ThreadingPlan: requested_mode="
        << (requested_execution_mode_ == ExecutionMode::Sequential ? "sequential" : "parallel")
        << ", effective_mode="
        << (effective_execution_mode_ == ExecutionMode::Sequential ? "sequential" : "parallel")
        << ", requested_inter_op_threads=" << requested_inter_op_threads_
        << ", requested_intra_op_threads=" << requested_intra_op_threads_
        << ", effective_inter_op_threads=" << effective_inter_op_threads_
        << ", effective_intra_op_threads=" << effective_intra_op_threads_
        << ", nested_parallelism_suppressed="
        << (nested_parallelism_suppressed_ ? "yes" : "no") << "\n";
    return oss.str();
}

}

#pragma once

#include "session_options.h"

#include <cstddef>
#include <string>

namespace tinyinfer{

// A resolved, immutable threading decision for one initialized session.
// SessionOptions stores user requests; ThreadingPlan stores what the runtime
// will actually use after applying execution-mode and hardware constraints.
class ThreadingPlan{
public:
    static ThreadingPlan build(const SessionOptions& options, size_t hardware_threads);

    ExecutionMode requested_execution_mode() const noexcept;
    ExecutionMode effective_execution_mode() const noexcept;
    size_t requested_inter_op_threads() const noexcept;
    size_t requested_intra_op_threads() const noexcept;
    size_t effective_inter_op_threads() const noexcept;
    size_t effective_intra_op_threads() const noexcept;
    bool nested_parallelism_suppressed() const noexcept;

    std::string dump() const;

private:
    ExecutionMode requested_execution_mode_ = ExecutionMode::Sequential;
    ExecutionMode effective_execution_mode_ = ExecutionMode::Sequential;
    size_t requested_inter_op_threads_ = 1;
    size_t requested_intra_op_threads_ = 1;
    size_t effective_inter_op_threads_ = 1;
    size_t effective_intra_op_threads_ = 1;
    bool nested_parallelism_suppressed_ = false;
};

}

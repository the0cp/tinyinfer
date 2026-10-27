#pragma once

#include <cstddef>

namespace tinyinfer{

enum class ExecutionMode{
    Sequential,
    Parallel
};

struct SessionOptions{
    ExecutionMode execution_mode = ExecutionMode::Sequential;
    // Zero means use the detected hardware concurrency. Inter-op parallelism
    // schedules independent graph nodes; intra-op parallelism splits one kernel.
    size_t inter_op_threads = 1;
    size_t intra_op_threads = 1;
    bool enable_graph_optimization = true;
    bool enable_memory_planning = true;
};

}

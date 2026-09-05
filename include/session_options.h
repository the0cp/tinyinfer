#pragma once

#include <cstddef>

namespace tinyinfer{

enum class ExecutionMode{
    Sequential,
    Parallel
};

struct SessionOptions{
    ExecutionMode execution_mode = ExecutionMode::Sequential;
    size_t inter_op_threads = 1;
    bool enable_graph_optimization = true;
    bool enable_memory_planning = true;
};

}

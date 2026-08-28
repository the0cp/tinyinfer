#pragma once

#include "operator_registry.h"
#include "tensor.h"
#include "value_index.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace tinyinfer{

enum class ValueRole{
    Input,
    Initializer,
    Intermediate
};

inline constexpr size_t lifetime_npos = static_cast<size_t>(-1);
// Represents an undefined lifetime position for values in the execution plan

struct ValueInfo{  // Represents metadata about a value in the execution plan
    std::string name;
    Shape shape;
    DataType dtype = DataType::Float32;
    ValueRole role = ValueRole::Intermediate;
    bool is_graph_output = false;
    size_t numel = 0;
    size_t byte_size = 0;
    size_t produced_at = lifetime_npos;
    size_t first_use = lifetime_npos;
    size_t last_use = lifetime_npos;
    size_t consumer_count = 0;
};

struct NodeExecutionPlan{  // Represents the execution plan for a single node in the graph
    NodeIndex source_node_index = invalid_node_index;
    std::string name;
    OpType op = OpType::ReLU;
    size_t kernel_index = 0;
    std::vector<ValueIndex> inputs;
    ValueIndex output = invalid_value_index;
    size_t dependency_count = 0;
    std::vector<size_t> consumers;
    std::vector<ValueIndex> release_after_execute;
};

class ExecutionPlan{
public:
    const ValueNameIndexMap& value_names() const noexcept;
    size_t value_count() const noexcept;
    size_t node_count() const noexcept;

    ValueIndex input_index() const noexcept;
    ValueIndex output_index() const noexcept;
    ValueIndex value_index(std::string_view name) const;

    const ValueInfo& value_info(ValueIndex index) const;
    const std::vector<ValueInfo>& values() const noexcept;
    const std::vector<NodeExecutionPlan>& nodes() const noexcept;

    const Shape& shape(ValueIndex index) const;
    const Shape& shape(std::string_view name) const;

    std::string dump() const;
    std::string dump_memory_plan() const;
    std::string dump_scheduler_plan() const;

private:
    friend class SessionState;  
    // Allow SessionState to construct and modify the ExecutionPlan

    ValueNameIndexMap value_names_;
    std::vector<ValueInfo> values_;
    std::vector<NodeExecutionPlan> nodes_;
    ValueIndex input_index_ = invalid_value_index;
    ValueIndex output_index_ = invalid_value_index;
};

}

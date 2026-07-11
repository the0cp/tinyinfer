#pragma once

#include "tensor.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyinfer{

class Graph;

using ShapeTable = std::unordered_map<std::string, Shape>;

inline constexpr size_t tensor_lifetime_npos = static_cast<size_t>(-1);

struct NodeScheduleInfo{
    size_t dependency_count = 0;
    std::vector<size_t> consumers;
};

struct TensorMemoryInfo{
    std::string name;
    Shape shape;
    size_t numel = 0;
    size_t byte_size = 0;

    bool is_input = false;
    bool is_output = false;
    bool is_constant = false;
    bool is_intermediate = false;

    size_t produced_at = tensor_lifetime_npos;
    size_t first_use = tensor_lifetime_npos;
    size_t last_use = tensor_lifetime_npos;
};

class ExecutionPlan{
public:
    const std::string& input_name() const noexcept;
    const std::string& output_name() const noexcept;

    const Shape& shape(const std::string& tensor_name) const;

    const ShapeTable& shapes() const noexcept;

    const std::vector<std::string>& execution_order() const noexcept;

    const std::vector<TensorMemoryInfo>& memory_infos() const noexcept;
    const std::vector<NodeScheduleInfo>& schedule_infos() const noexcept;

    size_t node_count() const noexcept;

private:
    friend class Graph;

    const Graph* owner_ = nullptr;
    size_t graph_revision_ = 0;

    std::string input_name_;
    std::string output_name_;

    ShapeTable shapes_;

    std::vector<size_t> node_indices_;
    std::vector<std::string> execution_order_;
    std::vector<TensorMemoryInfo> memory_infos_;
    std::vector<NodeScheduleInfo> schedule_infos_;
};

}
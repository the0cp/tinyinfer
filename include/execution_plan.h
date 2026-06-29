#pragma once

#include "tensor.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyinfer{

class Graph;

using ShapeTable = std::unordered_map<std::string, Shape>;

class ExecutionPlan{
public:
    const std::string& input_name() const noexcept;
    const std::string& output_name() const noexcept;

    const Shape& shape(const std::string& tensor_name) const;

    const ShapeTable& shapes() const noexcept;

    const std::vector<std::string>& execution_order() const noexcept;

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
};

}
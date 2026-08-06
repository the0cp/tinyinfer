#pragma once

#include "operator_registry.h"
#include "tensor.h"
#include "value_index.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tinyinfer{

using ShapeTable = std::unordered_map<std::string, Shape>;

struct Node{
    std::string name;
    OpType op;
    std::vector<std::string> inputs;
    std::string output;

    bool operator==(const Node&) const = default;
};

class PassManager;
class SessionState;

class Graph{
public:
    Graph() = default;
    Graph(const Graph&) = default;
    Graph& operator=(const Graph&) = default;
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    void set_tensor(std::string name, Tensor tensor);

    void add_node(
        std::string name,
        OpType op,
        std::vector<std::string> inputs,
        std::string output
    );

    void resolve(
        std::string input_name,
        Shape input_shape,
        std::string output_name,
        const OperatorRegistry& registry
    );

    bool is_resolved() const noexcept;

    const std::string& input_name() const;
    const Shape& input_shape() const;
    const std::string& output_name() const;

    const Tensor& constant(const std::string& name) const;
    bool has_constant(const std::string& name) const noexcept;

    const std::unordered_map<std::string, Tensor>& constants() const noexcept;
    const std::vector<Node>& nodes() const noexcept;
    const std::vector<NodeIndex>& topological_order() const;
    const ShapeTable& shapes() const;
    const Shape& shape(const std::string& value_name) const;

    size_t num_nodes() const noexcept;

    std::string dump(const OperatorRegistry& registry) const;
    std::string dump_constants() const;

private:
    friend class PassManager;
    friend class SessionState;

    void replace_nodes(std::vector<Node> nodes);
    void invalidate_resolve_state() noexcept;
    void require_resolved() const;

    std::unordered_map<std::string, Tensor> constants_;
    std::vector<Node> nodes_;

    bool resolved_ = false;
    std::string input_name_;
    Shape input_shape_;
    std::string output_name_;
    ShapeTable shapes_;
    std::vector<NodeIndex> topological_order_;
};

}

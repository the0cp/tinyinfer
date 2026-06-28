#pragma once

#include "operator_registry.h"
#include "tensor.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tinyinfer{

using ShapeTable = std::unordered_map<std::string, Shape>;

struct Node{
    std::string name;
    OpType op;
    std::vector<std::string> inputs;
    std::string output;
};

class Graph{
public:
    void set_tensor(std::string name, Tensor tensor);

    void add_node(
        std::string name,
        OpType op,
        std::vector<std::string> input,
        std::string output
    );

    void validate(
        const std::string& input_name,
        const std::string& output_name
    ) const;

    void infer_shapes(
        const std::string& input_name,
        const Shape& input_shape
    );

    const Shape& inferred_shape(const std::string& name) const;

    void execute();

    void execute_topological();

    Tensor forward(
        const std::string& input_name,
        const Tensor& input,
        const std::string& output_name
    );

    const Tensor& tensor(const std::string& name) const;

    size_t num_nodes() const;
    bool has_tensor(const std::string& name) const;

    std::string dump() const;
    std::string dump_tensors() const;
    std::string dump_execution_order() const;
    std::string dump_shapes() const;

    const std::vector<std::string>& last_execution_order() const;

private:
    OperatorRegistry registry_;
    
    std::unordered_map<std::string, Tensor> tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> last_execution_order_;

    ShapeTable last_inferred_shapes_;

    const Tensor& get_tensor_or_throw(const std::string& name) const;

    bool inputs_ready(const Node& node) const;
    void execute_node(const Node& node);
    void clear_node_outputs();

    Shape infer_node_output_shape(
        const Node& node,
        const ShapeTable& shapes
    ) const;
};

}
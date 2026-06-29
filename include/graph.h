#pragma once

#include "operator_registry.h"
#include "tensor.h"
#include "execution_plan.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tinyinfer{

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

    ExecutionPlan compile(
        const std::string& input_name,
        const Shape& input_shape,
        const std::string& output_name
    ) const;

    Tensor run(
        const ExecutionPlan& plan,
        const Tensor& input
    );

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
    std::string dump_plan(const ExecutionPlan& plan) const;

private:
    OperatorRegistry registry_;

    std::unordered_map<std::string, Tensor> constants_;
    std::unordered_map<std::string, Tensor> workspace_;
    
    std::vector<Node> nodes_;

    size_t revision_ = 0;

    const Tensor& get_tensor_or_throw(const std::string& name) const;

    void validate_structure(
        const std::string& input_name,
        const std::string& output_name
    ) const;

    void store_runtime_tensor(std::string name, Tensor tensor);

    Shape infer_node_output_shape(
        const Node& node,
        const ShapeTable& shapes
    ) const;

    void execute_node(const Node& node);
};

}
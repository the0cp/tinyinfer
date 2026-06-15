#pragma once

#include "tensor.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tinyinfer{

enum class OpType{
    Linear,
    ReLU,
    Softmax,
    Add
};

const char* op_type_to_string(OpType op);

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

    const std::vector<std::string>& last_execution_order() const;

private:
    std::unordered_map<std::string, Tensor> tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> last_execution_order_;

    const Tensor& get_tensor_or_throw(const std::string& name) const;

    bool inputs_ready(const Node& node) const;
    void execute_node(const Node& node);
    void clear_node_outputs();
};

}
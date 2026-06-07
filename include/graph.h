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
    Softmax
};

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

    Tensor forward(
        const std::string& input_name,
        const Tensor& input,
        const std::string& output_name
    );

    const Tensor& tensor(const std::string& name) const;

    size_t num_nodes() const;
    bool has_tensor(const std::string& name) const;

private:
    std::unordered_map<std::string, Tensor> tensors_;
    std::vector<Node> nodes_;

    const Tensor& get_tensor_or_throw(const std::string& name) const;
};

}
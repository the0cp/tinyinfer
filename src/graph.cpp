#include "graph.h"
#include "ops.h"

#include <stdexcept>

namespace tinyinfer{

void Graph::set_tensor(std::string name, Tensor tensor){
    tensors_.insert_or_assign(std::move(name), std::move(tensor));
}

void Graph::add_node(
    std::string name,
    OpType op,
    std::vector<std::string> inputs,
    std::string output
){
    nodes_.push_back(Node{
        std::move(name),
        op,
        std::move(inputs),
        std::move(output)
    });
}

const Tensor& Graph::get_tensor_or_throw(const std::string& name) const{
    auto it = tensors_.find(name);

    if(it == tensors_.end()){
        throw std::runtime_error("Tensor not found: " + name);
    }

    return it->second;
}

const Tensor& Graph::tensor(const std::string& name) const{
    return get_tensor_or_throw(name);
}

bool Graph::has_tensor(const std::string& name) const{
    return tensors_.find(name) != tensors_.end();
}

size_t Graph::num_nodes() const{
    return nodes_.size();
}

void Graph::execute_node(const Node& node){
    switch(node.op){
        case OpType::Linear:{
            if(node.inputs.size() != 3){
                throw std::runtime_error(
                    "Linear node expects 3 inputs: x, weight, bias"
                );
            }

            const Tensor& x = get_tensor_or_throw(node.inputs[0]);
            const Tensor& weight = get_tensor_or_throw(node.inputs[1]);
            const Tensor& bias = get_tensor_or_throw(node.inputs[2]);

            Tensor out = linear(x, weight, bias);

            set_tensor(node.output, std::move(out));
            break;
        }
        case OpType::ReLU:{
            if(node.inputs.size() != 1){
                throw std::runtime_error(
                    "ReLU node expects 1 input"
                );
            }

            const Tensor& x = get_tensor_or_throw(node.inputs[0]);

            Tensor out = relu(x);

            set_tensor(node.output, std::move(out));
            break;
        }
        case OpType::Softmax:{
            if(node.inputs.size() != 1){
                throw std::runtime_error(
                    "Softmax node expects 1 input"
                );
            }

            const Tensor& x = get_tensor_or_throw(node.inputs[0]);

            Tensor out = softmax(x);

            set_tensor(node.output, std::move(out));
            break;
        }
    }
}

void Graph::execute_topological(){
    clear_node_outputs();

    std::vector<bool> executed(nodes_.size(), false);
    size_t executed_count = 0;

    while(executed_count < nodes_.size()){
        bool progress = false;

        for(size_t i = 0; i < nodes_.size(); i++){
            if(executed[i]){
                continue;
            }

            const Node& node = nodes_[i];

            if(!inputs_ready(node)){
                continue;
            }

            execute_node(node);

            executed[i] = true;
            executed_count++;
            progress = true;
        }

        if(!progress){
            throw std::runtime_error(
                "Graph execution failed: missing dependency or cycle"
            );
        }
    }
}

void Graph::execute(){
    clear_node_outputs();

    for(const Node& node : nodes_){
        execute_node(node);
    }
}

Tensor Graph::forward(
    const std::string& input_name,
    const Tensor& input,
    const std::string& output_name
){
    set_tensor(input_name, input);

    execute_topological();

    return tensor(output_name);
}

bool Graph::inputs_ready(const Node& node) const{
    for(const std::string& input : node.inputs){
        if(!has_tensor(input)){
            return false;
        }
    }

    return true;
}

void Graph::clear_node_outputs(){
    for(const Node& node : nodes_){
        tensors_.erase(node.output);
    }
}

}
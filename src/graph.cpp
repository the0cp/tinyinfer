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

void Graph::execute(){
    for(const Node& node : nodes_){
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
}

Tensor Graph::forward(
    const std::string& input_name,
    const Tensor& input,
    const std::string& output_name
){
    set_tensor(input_name, input);

    execute();

    return tensor(output_name);
}

}
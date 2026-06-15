#include "graph.h"
#include "ops.h"

#include <stdexcept>
#include <sstream>

namespace tinyinfer{

namespace{

std::string join_strings(const std::vector<std::string>& values){
    std::ostringstream oss;

    for(size_t i = 0; i < values.size(); i++){
        if(i > 0){
            oss << ", ";
        }

        oss << values[i];
    }

    return oss.str();
}

std::string shape_to_string(const Tensor& tensor){
    std::ostringstream oss;

    oss << "[";

    const auto& shape = tensor.shape();

    for(size_t i = 0; i < shape.size(); i++){
        if(i > 0){
            oss << ", ";
        }

        oss << shape[i];
    }

    oss << "]";

    return oss.str();
}

}

const char* op_type_to_string(OpType op){
    switch(op){
        case OpType::Linear:
            return "Linear";
        case OpType::ReLU:
            return "ReLU";
        case OpType::Softmax:
            return "Softmax";
        case OpType::Add:
            return "Add";
    }

    return "Unknown";
}

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

const std::vector<std::string>& Graph::last_execution_order() const{
    return last_execution_order_;
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
        case OpType::Add:{
            if(node.inputs.size() != 2){
                throw std::runtime_error(
                    "Add node '" + node.name +
                    "' expects 2 inputs"
                );
            }

            const Tensor& a = get_tensor_or_throw(node.inputs[0]);
            const Tensor& b = get_tensor_or_throw(node.inputs[1]);

            Tensor out = add(a, b);

            set_tensor(node.output, std::move(out));
            break;
        }
    }
}

void Graph::execute_topological(){
    clear_node_outputs();
    last_execution_order_.clear();

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

            last_execution_order_.push_back(node.name);
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
    last_execution_order_.clear();

    for(const Node& node : nodes_){
        execute_node(node);
        last_execution_order_.push_back(node.name);
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

std::string Graph::dump() const{
    std::ostringstream oss;

    oss << "Graph:\n";

    if(nodes_.empty()){
        oss << " <empty>\n";
        return oss.str();
    }

    for(const Node& node : nodes_){
        oss << "  "
            << node.name
            << ": "
            << op_type_to_string(node.op)
            << "("
            << join_strings(node.inputs)
            << ") -> "
            << node.output
            << "\n";
    }

    return oss.str();
}

std::string Graph::dump_execution_order() const{
    std::ostringstream oss;

    oss << "Execution order:\n";

    if(last_execution_order_.empty()){
        oss << " <not executed>\n";
        return oss.str();
    }

    for(const std::string& name : last_execution_order_){
        oss << "  " << name << "\n";
    }

    return oss.str();
}

std::string Graph::dump_tensors() const{
    std::ostringstream oss;

    oss << "Tensors:\n";

    if(tensors_.empty()){
        oss << " <empty>\n";
        return oss.str();
    }

    for(const auto& [name, tensor] : tensors_){
        oss << "  "
            << name
            << ": shape="
            << shape_to_string(tensor)
            << ", numel="
            << tensor.numel()
            << "\n";
    }

    return oss.str();
}

}
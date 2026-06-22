#include "graph.h"
#include "ops.h"

#include <iterator>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

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

std::string shape_to_string(const Shape& shape){
    std::ostringstream oss;

    oss << "[";

    for(size_t i = 0; i < shape.size(); i++){
        if(i > 0){
            oss << ", ";
        }

        oss << shape[i];
    }

    oss << "]";

    return oss.str();
}

std::string set_to_string(const std::unordered_set<std::string>& values){
    std::ostringstream oss;

    oss << "{";

    for(auto it = values.begin(); it != values.end(); it++){
        oss << *it;
        if(std::next(it) != values.end()){
            oss << ", ";
        }
    }

    oss << "}";
    return oss.str();
}

std::vector<std::string> missing_inputs_for_node(
    const Node& node,
    const std::unordered_set<std::string>& available
){
    std::vector<std::string> missing;

    for(const std::string& input : node.inputs){
        if(!available.contains(input)){
            missing.push_back(input);
        }
    }

    return missing;
}

size_t expected_input_count(OpType op){
    switch(op){
        case OpType::Linear:
            return 3;
        case OpType::ReLU:
            return 1;
        case OpType::Add:
            return 2;
        case OpType::Softmax:
            return 1;
    }

    return 0;
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

void Graph::validate(
    const std::string& input_name,
    const std::string& output_name
) const{
    if(input_name.empty()){
        throw std::runtime_error("Graph validation failed: input name is empty");
    }

    if(output_name.empty()){
        throw std::runtime_error("Graph validation failed: output name is empty");
    }

    std::unordered_set<std::string> node_names;
    std::unordered_map<std::string, std::string> output_writers;
    std::unordered_set<std::string> node_outputs;

    for(const Node& node : nodes_){
        if(node.name.empty()){
            throw std::runtime_error("Graph validation failed: node name is empty");
        }

        if(!node_names.insert(node.name).second){
            throw std::runtime_error(
                "Graph validation failed: duplicate node name '" + 
                node.name + "'"
            );
        }

        if(node.output.empty()){
            throw std::runtime_error(
                "Graph validation failed: node '" + 
                node.name + "'" + "has empty output name"
            );
        }

        const size_t expected = expected_input_count(node.op);

        if(node.inputs.size() != expected){
            throw std::runtime_error(
                "Graph validation failed: node '" + 
                node.name + "'" +
                "(" + op_type_to_string(node.op) + ") expects " +
                std::to_string(expected) + " inputs, got " +
                std::to_string(node.inputs.size())
            );
        }

        auto [it, inserted] = output_writers.emplace(node.output, node.name);

        if(!inserted){
            throw std::runtime_error(
                "Graph validation failed: tensor '" + 
                node.output + "' is written by multiple nodes: " +
                "'" + it->second + "' and '" + node.name + "'"
            );
        }

        node_outputs.insert(node.output);
    }

    std::unordered_set<std::string> available;

    for(const auto& [name, tensor] : tensors_){
        (void)tensor;

        if(!node_outputs.contains(name)){
            available.insert(name);
        }
    }

    available.insert(input_name);

    std::vector<bool> executed(nodes_.size(), false);
    size_t executed_count = 0;

    while(executed_count < nodes_.size()){
        bool progress = false;

        for(size_t i = 0; i < nodes_.size(); i++){
            if(executed[i]){
                continue;
            }

            const Node& node = nodes_[i];
            const auto missing = missing_inputs_for_node(node, available);

            if(!missing.empty()){
                continue;
            }

            available.insert(node.output);
            executed[i] = true;
            executed_count++;
            progress = true;
        }

        if(!progress){
            std::ostringstream oss;

            oss << "Graph validation failed: unresovled dependency or cycle\n";
            oss << "Available tensors: " << set_to_string(available) << "\n";
            oss << "Pending nodes: \n";

            for(size_t i = 0; i < nodes_.size(); i++){
                if(executed[i]){
                    continue;
                }

                const Node& node = nodes_[i];
                const auto missing = missing_inputs_for_node(node, available);

                oss << "  "
                    << node.name
                    << ": "
                    << op_type_to_string(node.op)
                    << "("
                    << join_strings(node.inputs)
                    << ") ->"
                    << node.output;

                if(!missing.empty()){
                    oss << ", missing: " << join_strings(missing);
                }

                oss << "\n";
            }

            oss << "\n" << dump();

            throw std::runtime_error(oss.str());
        } 
    }

    if(!available.contains(output_name)){
        throw std::runtime_error(
            "Graph validation failed: requested output '" +
            output_name + "' is not produced by this graph"
            );
    }
}

Shape Graph::infer_node_output_shape(
    const Node& node,
    const ShapeTable& shapes
) const{
    switch(node.op){
        case OpType::Linear:{
            const Shape& x = shapes.at(node.inputs[0]);
            const Shape& weight = shapes.at(node.inputs[1]);
            const Shape& bias = shapes.at(node.inputs[2]);

            if(x.size() != 2){
                throw std::runtime_error(
                    "Shape inference failed: Linear node '" +
                    node.name + "' expects input to be 2D, got " +
                    shape_to_string(x)
                );
            }

            if(weight.size() != 2){
                throw std::runtime_error(
                    "Shape inference failed: Linear node '" +
                    node.name + "' expects weight to be 2D, got " +
                    shape_to_string(weight)
                );
            }

            if(bias.size() != 1){
                throw std::runtime_error(
                    "Shape inference failed: Linear node '" +
                    node.name + "' expects bias to be 1D, got " +
                    shape_to_string(bias)
                );
            }

            if(x[1] != weight[0]){
                throw std::runtime_error(
                    "Shape inference failed: Linear node '" +
                    node.name + "' cannot multiply input " +
                    shape_to_string(x) + " by weight " +
                    shape_to_string(weight)
                );
            }

            if(weight[1] != bias[0]){
                throw std::runtime_error(
                    "Shape inference failed: Linear node '" +
                    node.name + "' has weight output size " +
                    std::to_string(weight[1]) +
                    " but bias size " +
                    std::to_string(bias[0])
                );
            }

            return {x[0], weight[1]};
        }
        case OpType::Add:{
            const Shape& a = shapes.at(node.inputs[0]);
            const Shape& b = shapes.at(node.inputs[1]);

            if(a != b){
                throw std::runtime_error(
                    "Shape inference failed: Add node '" +
                    node.name + "' has incompatible inputs " +
                    shape_to_string(a) + " and " +
                    shape_to_string(b)
                );
            }

            return a;
        }
        case OpType::ReLU:{
            return shapes.at(node.inputs[0]);
        }
        case OpType::Softmax:{
            const Shape& x = shapes.at(node.inputs[0]);

            if(x.size() != 2){
                throw std::runtime_error(
                    "Shape inference failed: Softmax node '" +
                    node.name + "' expects a 2D input, got " +
                    shape_to_string(x)
                );
            }

            if(x[1] == 0){
                throw std::runtime_error(
                    "Shape inference failed: Softmax node '" +
                    node.name +
                    "' has an empty feature dimension"
                );
            }

            return x;
        }
    }

    throw std::runtime_error("Shape inference failed: unsupported operator");
}

void Graph::infer_shapes(
    const std::string& input_name,
    const Shape& input_shape
){
    last_inferred_shapes_.clear();

    ShapeTable shapes;
    std::unordered_set<std::string> node_outputs;

    for(const Node& node : nodes_){
        node_outputs.insert(node.output);
    }

    for(const auto& [name, tensor] : tensors_){
        if(!node_outputs.contains(name)){
            shapes.emplace(name, tensor.shape());
        }
    }

    shapes.insert_or_assign(input_name, input_shape);

    std::vector<bool> inferred(nodes_.size(), false);
    size_t inferred_count = 0;

    while(inferred_count < nodes_.size()){
        bool progress = false;

        for(size_t i = 0; i < nodes_.size(); i++){
            if(inferred[i]){
                continue;
            }

            const Node& node = nodes_[i];

            bool ready = true;

            for(const std::string& input : node.inputs){
                if(!shapes.contains(input)){
                    ready = false;
                    break;
                }
            }

            if(!ready){
                continue;
            }

            Shape output_shape = infer_node_output_shape(node, shapes);

            shapes.insert_or_assign(node.output, std::move(output_shape));
            
            inferred[i] = true;
            inferred_count++;
            progress = true;
        }

        if(!progress){
            throw std::runtime_error("Shape inference failed: unresolved dependency or cycle");
        }
    }

    last_inferred_shapes_ = std::move(shapes);
}

const Shape& Graph::inferred_shape(const std::string& name) const{
    auto it = last_inferred_shapes_.find(name);

    if(it == last_inferred_shapes_.end()){
        throw std::runtime_error("Inferred shape not found for tensor: " + name);
    }

    return it->second;
}

Tensor Graph::forward(
    const std::string& input_name,
    const Tensor& input,
    const std::string& output_name
){
    set_tensor(input_name, input);

    validate(input_name, output_name);

    infer_shapes(input_name, input.shape());

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
            << shape_to_string(tensor.shape())
            << ", numel="
            << tensor.numel()
            << "\n";
    }

    return oss.str();
}

std::string Graph::dump_shapes() const{
    std::ostringstream oss;

    oss << "Graph shapes:\n";

    if(last_inferred_shapes_.empty()){
        oss << "  <not inferred>\n";
        return oss.str();
    }

    for(const Node& node : nodes_){
        oss << "  "
            << node.name
            << ": "
            << op_type_to_string(node.op)
            << "(";

        for(size_t i = 0; i < node.inputs.size(); i++){
            if(i > 0){
                oss << ", ";
            }

            const std::string& input_name = node.inputs[i];

            oss << input_name
                << " "
                << shape_to_string(inferred_shape(input_name));
        }

        oss << ") -> "
            << node.output
            << " "
            << shape_to_string(inferred_shape(node.output))
            << "\n";
    }

    return oss.str();
}

}

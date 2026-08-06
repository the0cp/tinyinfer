#include "graph.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tinyinfer{

namespace{

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

NodeIndex checked_node_index(size_t index){
    if(index >= static_cast<size_t>(invalid_node_index)){
        throw std::overflow_error("Graph contains too many nodes for NodeIndex.");
    }

    return static_cast<NodeIndex>(index);
}

}

void Graph::set_tensor(std::string name, Tensor tensor){
    if(name.empty()){
        throw std::invalid_argument("Cannot store a constant tensor with an empty name.");
    }

    constants_.insert_or_assign(std::move(name), std::move(tensor));
    invalidate_resolve_state();
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

    invalidate_resolve_state();
}

void Graph::replace_nodes(std::vector<Node> nodes){
    nodes_ = std::move(nodes);
    invalidate_resolve_state();
}

void Graph::invalidate_resolve_state() noexcept{
    resolved_ = false;
    input_name_.clear();
    input_shape_.clear();
    output_name_.clear();
    shapes_.clear();
    topological_order_.clear();
}

void Graph::resolve(
    std::string input_name,
    Shape input_shape,
    std::string output_name,
    const OperatorRegistry& registry
){
    if(input_name.empty()){
        throw std::runtime_error("Graph resolve failed: input name is empty.");
    }

    if(output_name.empty()){
        throw std::runtime_error("Graph resolve failed: output name is empty.");
    }

    if(constants_.contains(input_name)){
        throw std::runtime_error(
            "Graph resolve failed: input '" + input_name + "' conflicts with an initializer."
        );
    }

    std::unordered_set<std::string> node_names;
    std::unordered_map<std::string, NodeIndex> producer_by_value;
    producer_by_value.reserve(nodes_.size());

    for(size_t i = 0; i < nodes_.size(); i++){
        const Node& node = nodes_[i];
        const NodeIndex node_index = checked_node_index(i);

        if(node.name.empty()){
            throw std::runtime_error("Graph resolve failed: node name is empty.");
        }

        if(!node_names.insert(node.name).second){
            throw std::runtime_error("Graph resolve failed: duplicate node name '" + node.name + "'.");
        }

        if(node.output.empty()){
            throw std::runtime_error("Graph resolve failed: node '" + node.name + "' has an empty output.");
        }

        if(node.output == input_name){
            throw std::runtime_error(
                "Graph resolve failed: node '" + node.name + "' overwrites graph input '" + input_name + "'."
            );
        }

        if(constants_.contains(node.output)){
            throw std::runtime_error(
                "Graph resolve failed: node '" + node.name + "' output '" + node.output +
                "' conflicts with an initializer."
            );
        }

        const OperatorSchema& schema = registry.get(node.op);

        if(node.inputs.size() != schema.input_count){
            throw std::runtime_error(
                "Graph resolve failed: node '" + node.name + "' (" + schema.name + ") expects " +
                std::to_string(schema.input_count) + " inputs, got " + std::to_string(node.inputs.size()) + "."
            );
        }

        for(const std::string& input : node.inputs){
            if(input.empty()){
                throw std::runtime_error(
                    "Graph resolve failed: node '" + node.name + "' has an empty input name."
                );
            }
        }

        if(!producer_by_value.emplace(node.output, node_index).second){
            throw std::runtime_error(
                "Graph resolve failed: value '" + node.output + "' has more than one producer."
            );
        }
    }

    std::vector<size_t> indegree(nodes_.size(), 0);
    std::vector<std::vector<NodeIndex>> consumers(nodes_.size());

    for(size_t i = 0; i < nodes_.size(); i++){
        const Node& node = nodes_[i];
        std::unordered_set<NodeIndex> unique_dependencies;

        for(const std::string& input : node.inputs){
            if(input == input_name || constants_.contains(input)){
                continue;
            }

            auto producer = producer_by_value.find(input);

            if(producer == producer_by_value.end()){
                throw std::runtime_error(
                    "Graph resolve failed: node '" + node.name + "' requires missing value '" + input + "'."
                );
            }

            if(unique_dependencies.insert(producer->second).second){
                consumers.at(producer->second).push_back(checked_node_index(i));
            }
        }

        indegree[i] = unique_dependencies.size();
    }

    std::deque<NodeIndex> ready;

    for(size_t i = 0; i < indegree.size(); i++){
        if(indegree[i] == 0){
            ready.push_back(checked_node_index(i));
        }
    }

    std::vector<NodeIndex> order;
    order.reserve(nodes_.size());

    while(!ready.empty()){
        const NodeIndex node_index = ready.front();
        ready.pop_front();
        order.push_back(node_index);

        for(NodeIndex consumer : consumers.at(node_index)){
            size_t& remaining = indegree.at(consumer);

            if(remaining == 0){
                throw std::logic_error("Graph resolve dependency count underflow.");
            }

            remaining--;

            if(remaining == 0){
                ready.push_back(consumer);
            }
        }
    }

    if(order.size() != nodes_.size()){
        std::ostringstream oss;
        oss << "Graph resolve failed: cycle detected. Pending nodes:";

        for(size_t i = 0; i < nodes_.size(); i++){
            if(indegree[i] != 0){
                oss << " " << nodes_[i].name;
            }
        }

        throw std::runtime_error(oss.str());
    }

    ShapeTable shapes;
    shapes.reserve(constants_.size() + nodes_.size() + 1);

    for(const auto& [name, tensor] : constants_){
        shapes.emplace(name, tensor.shape());
    }

    shapes.emplace(input_name, input_shape);

    for(NodeIndex node_index : order){
        const Node& node = nodes_.at(node_index);
        const OperatorSchema& schema = registry.get(node.op);
        ShapeInputs input_shapes;
        input_shapes.reserve(node.inputs.size());

        for(const std::string& input : node.inputs){
            auto shape = shapes.find(input);

            if(shape == shapes.end()){
                throw std::logic_error(
                    "Graph resolve invariant broken: shape for value '" + input + "' is unavailable."
                );
            }

            input_shapes.push_back(&shape->second);
        }

        Shape output_shape = schema.infer_shape(input_shapes, node.name);

        if(!shapes.emplace(node.output, std::move(output_shape)).second){
            throw std::logic_error(
                "Graph resolve invariant broken: output shape for value '" + node.output + "' already exists."
            );
        }
    }

    if(!shapes.contains(output_name)){
        throw std::runtime_error(
            "Graph resolve failed: requested output '" + output_name + "' is unavailable."
        );
    }

    input_name_ = std::move(input_name);
    input_shape_ = std::move(input_shape);
    output_name_ = std::move(output_name);
    shapes_ = std::move(shapes);
    topological_order_ = std::move(order);
    resolved_ = true;
}

bool Graph::is_resolved() const noexcept{
    return resolved_;
}

void Graph::require_resolved() const{
    if(!resolved_){
        throw std::logic_error("Graph has not been resolved.");
    }
}

const std::string& Graph::input_name() const{
    require_resolved();
    return input_name_;
}

const Shape& Graph::input_shape() const{
    require_resolved();
    return input_shape_;
}

const std::string& Graph::output_name() const{
    require_resolved();
    return output_name_;
}

const Tensor& Graph::constant(const std::string& name) const{
    auto it = constants_.find(name);

    if(it == constants_.end()){
        throw std::runtime_error("Initializer not found: " + name);
    }

    return it->second;
}

bool Graph::has_constant(const std::string& name) const noexcept{
    return constants_.contains(name);
}

const std::unordered_map<std::string, Tensor>& Graph::constants() const noexcept{
    return constants_;
}

const std::vector<Node>& Graph::nodes() const noexcept{
    return nodes_;
}

const std::vector<NodeIndex>& Graph::topological_order() const{
    require_resolved();
    return topological_order_;
}

const ShapeTable& Graph::shapes() const{
    require_resolved();
    return shapes_;
}

const Shape& Graph::shape(const std::string& value_name) const{
    require_resolved();
    auto it = shapes_.find(value_name);

    if(it == shapes_.end()){
        throw std::runtime_error("Resolved Graph has no shape for value '" + value_name + "'.");
    }

    return it->second;
}

size_t Graph::num_nodes() const noexcept{
    return nodes_.size();
}

std::string Graph::dump(const OperatorRegistry& registry) const{
    std::ostringstream oss;
    oss << "Graph:\n";

    if(nodes_.empty()){
        oss << "  <empty>\n";
        return oss.str();
    }

    for(const Node& node : nodes_){
        oss << "  " << node.name << ": " << registry.get(node.op).name << "("
            << join_strings(node.inputs) << ") -> " << node.output << "\n";
    }

    return oss.str();
}

std::string Graph::dump_constants() const{
    std::vector<std::string> names;
    names.reserve(constants_.size());

    for(const auto& [name, tensor] : constants_){
        (void)tensor;
        names.push_back(name);
    }

    std::sort(names.begin(), names.end());

    std::ostringstream oss;
    oss << "Initializers:\n";

    if(names.empty()){
        oss << "  <empty>\n";
        return oss.str();
    }

    for(const std::string& name : names){
        const Tensor& tensor = constants_.at(name);
        oss << "  " << name << ": shape=" << shape_to_string(tensor.shape())
            << ", numel=" << tensor.numel() << "\n";
    }

    return oss.str();
}

}

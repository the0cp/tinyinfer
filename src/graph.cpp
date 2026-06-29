#include "graph.h"

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

}

void Graph::set_tensor(std::string name, Tensor tensor){
    if(name.empty()){
        throw std::invalid_argument("Cannot store a constant tensor with an empty name.");
    }

    constants_.insert_or_assign(std::move(name), std::move(tensor));

    workspace_.clear();
    revision_++;
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

    workspace_.clear();
    revision_++;
}

void Graph::store_runtime_tensor(std::string name, Tensor tensor){
    workspace_.insert_or_assign(std::move(name), std::move(tensor));
}

const Tensor& Graph::get_tensor_or_throw(const std::string& name) const{
    auto rt_it = workspace_.find(name);
    if(rt_it != workspace_.end()){
        return rt_it->second;
    }

    auto const_it = constants_.find(name);
    if(const_it != constants_.end()){
        return const_it->second;
    }

    throw std::runtime_error("Tensor not found: " + name);
}

const Tensor& Graph::tensor(const std::string& name) const{
    return get_tensor_or_throw(name);
}

bool Graph::has_tensor(const std::string& name) const{
    return workspace_.contains(name) || constants_.contains(name);
}

size_t Graph::num_nodes() const{
    return nodes_.size();
}

void Graph::execute_node(const Node& node){
    const OperatorDefinition& definition = registry_.get(node.op);

    if(node.inputs.size() != definition.input_count){
        throw std::logic_error(
            "ExecutionPlan invariant broken at node '" + node.name + "'."
        );
    }

    TensorInputs inputs;
    inputs.reserve(node.inputs.size());

    for(const std::string& input_name : node.inputs){
        inputs.push_back(&get_tensor_or_throw(input_name));
    }

    try{
        Tensor output = definition.execute(inputs);
        store_runtime_tensor(node.output, std::move(output));
    }catch(const std::exception& error){
        throw std::runtime_error(
            "Execution failed at node '" + node.name + "' (" + definition.name + "'): " + error.what()
        );
    }
}

Tensor Graph::run(
    const ExecutionPlan& plan,
    const Tensor& input
){
    if(plan.owner_ != this){
        throw std::runtime_error("ExecutionPlan belongs to another Graph.");
    }

    if(plan.graph_revision_ != revision_){
        throw std::runtime_error("ExectuionPlan is stale, the Graph changed after compilation");
    }

    const Shape& expected_shape = plan.shape(plan.input_name_);

    if(input.shape() != expected_shape){
        throw std::runtime_error(
            "Input shape mismatch: expected " + shape_to_string(expected_shape) + 
            ", got " + shape_to_string(input.shape())
        );
    }

    workspace_.clear();

    store_runtime_tensor(plan.input_name_, input);

    for(size_t node_index : plan.node_indices_){
        if(node_index >= nodes_.size()){
            throw std::logic_error("ExecutionPlan contains an invalid node.");
        }

        execute_node(nodes_[node_index]);
    }

    return tensor(plan.output_name_);
}

void Graph::validate_structure(
    const std::string& input_name,
    const std::string& output_name
) const{
    if(input_name.empty()){
        throw std::runtime_error("Graph validation failed: input name is empty");
    }

    if(output_name.empty()){
        throw std::runtime_error("Graph validation failed: output name is empty");
    }

    if(constants_.contains(input_name)){
        throw std::runtime_error(
            "Graph compilation failed: input '" + input_name + "' conflicts with a constant tensor."
        );
    }

    std::unordered_set<std::string> node_names;
    std::unordered_map<std::string, std::string> output_writers;

    for(const Node& node : nodes_){
        if(node.name.empty()){
            throw std::runtime_error("Graph compilation failed: node name is empty");
        }

        if(!node_names.insert(node.name).second){
            throw std::runtime_error(
                "Graph compilation failed: duplicate node name '" + 
                node.name + "'"
            );
        }

        if(node.output.empty()){
            throw std::runtime_error(
                "Graph compilation failed: node '" + node.name + "'" + "has empty output name"
            );
        }

        if(node.output == input_name){
             throw std::runtime_error(
                "Graph compilation failed: node '" + node.name + "' overwrites graph input '" +
                input_name + "'"
            );
        }

        if(constants_.contains(node.output)){
            throw std::runtime_error(
                "Graph compilation failed: node '" + node.name + "' output '" +
                node.output + "' conflicts with a constant tensor"
            );
        }

        const OperatorDefinition& definition = registry_.get(node.op);

        const size_t expected = definition.input_count;

        if(node.inputs.size() != expected){
            throw std::runtime_error(
                "Graph compilation failed: node '" + 
                node.name + "'" +
                "(" + definition.name + ") expects " +
                std::to_string(expected) + " inputs, got " +
                std::to_string(node.inputs.size())
            );
        }

        for(const std::string& input : node.inputs){
            if(input.empty()){
                throw std::runtime_error(
                    "Graph compilation failed: node '" + node.name + "' has an empty input name"
                );
            }
        }

        auto [it, inserted] = output_writers.emplace(node.output, node.name);

        if(!inserted){
            throw std::runtime_error(
                "Graph compilation failed: tensor '" + 
                node.output + "' is written by multiple nodes: " +
                "'" + it->second + "' and '" + node.name + "'"
            );
        }
    }
}

Shape Graph::infer_node_output_shape(
    const Node& node,
    const ShapeTable& shapes
) const{
    const OperatorDefinition& definition = registry_.get(node.op);

    if(node.inputs.size() != definition.input_count){
        throw std::logic_error(
            "Graph invariant broken: node '" + node.name + "' has an invalid input count during shape inference"
        );
    }

    ShapeInputs input_shapes;
    input_shapes.reserve(node.inputs.size());

    for(const std::string& input_name : node.inputs){
        auto it = shapes.find(input_name);

        if(it == shapes.end()){
            throw std::logic_error(
                "Grpah invariant broken: shape for tensor '" +
                input_name +
                "' is not available"
            );
        }

        input_shapes.push_back(&it->second);
    }

    return definition.infer_shape(input_shapes, node.name);
}

ExecutionPlan Graph::compile(
    const std::string& input_name,
    const Shape& input_shape,
    const std::string& output_name
) const{
    validate_structure(input_name, output_name);

    ExecutionPlan plan;

    plan.owner_ = this;
    plan.graph_revision_ = revision_;
    plan.input_name_ = input_name;
    plan.output_name_ = output_name;

    ShapeTable shapes;

    for(const auto& [name, tensor] : constants_){
        shapes.emplace(name, tensor.shape());
    }

    shapes.emplace(input_name, input_shape);

    std::vector<bool> compiled(nodes_.size(), false);

    size_t compiled_count = 0;

    while(compiled_count < nodes_.size()){
        bool progress = false;

        for(size_t i = 0; i < nodes_.size(); i++){
            if(compiled[i]){
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

            if(!ready)  continue;

            Shape output_shape = infer_node_output_shape(node, shapes);

            auto [shape_it, inserted] = shapes.emplace(node.output, std::move(output_shape));
            (void)shape_it;

            if(!inserted){
                throw std::logic_error(
                    "Graph invariant broken: output shape for tensor '" + node.output + "' already exists."
                );
            }

            plan.node_indices_.push_back(i);
            plan.execution_order_.push_back(node.name);

            compiled[i] = true;
            compiled_count++;
            progress = true;
        }

        if(!progress){
            std::ostringstream oss;

            oss << "Graph compilation failed: unresolved dependency or cycle\n";

            oss << "Pending nodes:\n";

            for(size_t i = 0; i < nodes_.size(); i++){
                if(compiled[i]){
                    continue;
                }

                const Node& node = nodes_[i];

                oss << "  "
                    << node.name
                    << ", missing:";

                bool has_missing = false;

                for(const std::string& input : node.inputs){
                    if(!shapes.contains(input)){
                        oss << " " << input;
                        has_missing = true;
                    }
                }

                if(!has_missing){
                    oss << " <cycle>";
                }

                oss << "\n";
            }

            throw std::runtime_error(oss.str());
        }
    }

    if(!shapes.contains(output_name)){
        throw std::runtime_error(
            "Graph compilation failed: requested output '" + output_name + "' is unavailable."
        );
    }

    plan.shapes_ = std::move(shapes);

    return plan;
}

Tensor Graph::forward(
    const std::string& input_name,
    const Tensor& input,
    const std::string& output_name
){
    ExecutionPlan plan = compile(
        input_name,
        input.shape(),
        output_name
    );

    return run(plan, input);
}

std::string Graph::dump() const{
    std::ostringstream oss;
    oss << "Graph:\n";

    if(nodes_.empty()){
        oss << "  <empty>\n";
        return oss.str();
    }

    for(const Node& node : nodes_){
        const OperatorDefinition& definition = registry_.get(node.op);

        oss << "  " << node.name << ": "
            << definition.name << "("
            << join_strings(node.inputs) << ") -> "
            << node.output << "\n";
    }

    return oss.str();
}

std::string Graph::dump_tensors() const{
    std::ostringstream oss;
    oss << "Constants:\n";

    if(constants_.empty()){
        oss << "  <empty>\n";
    }else{
        for(const auto& [name, tensor] : constants_){
            oss << "  " << name
                << ": shape=" << shape_to_string(tensor.shape())
                << ", numel=" << tensor.numel() << "\n";
        }
    }

    oss << "Workspace:\n";

    if(workspace_.empty()){
        oss << "  <empty>\n";
    }else{
        for(const auto& [name, tensor] : workspace_){
            oss << "  " << name
                << ": shape=" << shape_to_string(tensor.shape())
                << ", numel=" << tensor.numel() << "\n";
        }
    }

    return oss.str();
}

std::string Graph::dump_plan(const ExecutionPlan& plan) const{
    if(plan.owner_ != this){
        throw std::runtime_error(
            "ExecutionPlan belongs to another Graph"
        );
    }

    if(plan.graph_revision_ != revision_){
        throw std::runtime_error(
            "Cannot dump a stale ExecutionPlan"
        );
    }

    std::ostringstream oss;
    oss << "Execution plan:\n";
    oss << "  input: " << plan.input_name_ << " "
        << shape_to_string(plan.shape(plan.input_name_)) << "\n";
    oss << "  output: " << plan.output_name_ << " "
        << shape_to_string(plan.shape(plan.output_name_)) << "\n";
    oss << "  nodes:\n";

    for(size_t node_index : plan.node_indices_){
        const Node& node = nodes_.at(node_index);
        const OperatorDefinition& definition = registry_.get(node.op);

        oss << "    " << node.name << ": "
            << definition.name << "(";

        for(size_t i = 0; i < node.inputs.size(); i++){
            if(i > 0){
                oss << ", ";
            }

            const std::string& input_name = node.inputs[i];
            oss << input_name << " "
                << shape_to_string(plan.shape(input_name));
        }

        oss << ") -> " << node.output << " "
            << shape_to_string(plan.shape(node.output)) << "\n";
    }

    return oss.str();
}

}

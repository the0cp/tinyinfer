#include "graph.h"

#include <algorithm>
#include <limits>
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

size_t checked_shape_numel(const Shape& shape){
    size_t total = 1;

    for(size_t dim : shape){
        if(dim != 0 && total > std::numeric_limits<size_t>::max() / dim){
            throw std::runtime_error("Shape element count overflows size_t.");
        }

        total *= dim;
    }

    return total;
}

size_t checked_byte_size(size_t numel){
    if(numel > std::numeric_limits<size_t>::max() / sizeof(float)){
        throw std::runtime_error("Tensor byte size overflows size_t.");
    }

    return numel * sizeof(float);
}

TensorMemoryInfo make_memory_info(
    std::string name,
    const Shape& shape,
    bool is_input,
    bool is_output,
    bool is_constant
){
    const size_t numel = checked_shape_numel(shape);

    TensorMemoryInfo info;
    info.name = std::move(name);
    info.shape = shape;
    info.numel = numel;
    info.byte_size = checked_byte_size(numel);
    info.is_input = is_input;
    info.is_output = is_output;
    info.is_constant = is_constant;
    info.is_intermediate = !is_input && !is_output && !is_constant;

    return info;
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

std::string memory_role_string(const TensorMemoryInfo& info){
    if(info.is_input){
        return "input";
    }

    if(info.is_output){
        return "output";
    }

    if(info.is_constant){
        return "constant";
    }

    return "intermediate";
}

std::string lifetime_index_to_string(size_t index){
    if(index == tensor_lifetime_npos){
        return "-";
    }

    return std::to_string(index);
}

void update_tensor_use(
    std::vector<TensorMemoryInfo>& memory_infos,
    const std::unordered_map<std::string, size_t>& memory_info_indices,
    const std::string& tensor_name,
    size_t node_position
){
    auto it = memory_info_indices.find(tensor_name);

    if(it == memory_info_indices.end()){
        throw std::logic_error(
            "ExecutionPlan memory analysis missing tensor '" + tensor_name + "'."
        );
    }

    TensorMemoryInfo& info = memory_infos.at(it->second);

    if(info.first_use == tensor_lifetime_npos){
        info.first_use = node_position;
    }

    info.last_use = node_position;
}

const TensorMemoryInfo& find_memory_info_or_throw(
    const ExecutionPlan& plan,
    const std::string& tensor_name
){
    for(const TensorMemoryInfo& info : plan.memory_infos()){
        if(info.name == tensor_name){
            return info;
        }
    }

    throw std::logic_error(
        "ExecutionPlan memory cleanup missing tensor '" + tensor_name + "'."
    );
}

}

void Graph::set_tensor(std::string name, Tensor tensor){
    if(name.empty()){
        throw std::invalid_argument("Cannot store a constant tensor with an empty name.");
    }

    constants_.insert_or_assign(std::move(name), std::move(tensor));

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

    revision_++;
}

void Graph::store_runtime_tensor(
    ExecutionContext& context,
    std::string name,
    Tensor tensor
) const{
    context.set_tensor(std::move(name), std::move(tensor));
}

const Tensor& Graph::get_tensor_or_throw(
    const ExecutionContext& context,
    const std::string& name
) const{
    if(context.has_tensor(name)){
        return context.tensor(name);
    }

    auto const_it = constants_.find(name);
    if(const_it != constants_.end()){
        return const_it->second;
    }

    throw std::runtime_error("Tensor not found: " + name);
}

const Tensor& Graph::constant(const std::string& name) const{
    auto it = constants_.find(name);

    if(it == constants_.end()){
        throw std::runtime_error("Constant tensor not found: " + name);
    }

    return it->second;
}

bool Graph::has_constant(const std::string& name) const{
    return constants_.contains(name);
}

size_t Graph::num_nodes() const{
    return nodes_.size();
}

void Graph::execute_node(ExecutionContext& context, const Node& node) const{
    const OperatorDefinition& definition = registry_.get(node.op);

    if(node.inputs.size() != definition.input_count){
        throw std::logic_error(
            "ExecutionPlan invariant broken at node '" + node.name + "'."
        );
    }

    TensorInputs inputs;
    inputs.reserve(node.inputs.size());

    for(const std::string& input_name : node.inputs){
        inputs.push_back(&get_tensor_or_throw(context, input_name));
    }

    try{
        Tensor output = definition.execute(inputs);
        store_runtime_tensor(context, node.output, std::move(output));
    }catch(const std::exception& error){
        throw std::runtime_error(
            "Execution failed at node '" + node.name + "' (" + definition.name + "'): " + error.what()
        );
    }
}

Tensor Graph::run(
    const ExecutionPlan& plan,
    ExecutionContext& context,
    const Tensor& input
) const{
    if(plan.owner_ != this){
        throw std::runtime_error("ExecutionPlan belongs to another Graph.");
    }

    if(plan.graph_revision_ != revision_){
        throw std::runtime_error("ExecutionPlan is stale, the Graph changed after compilation");
    }

    const Shape& expected_shape = plan.shape(plan.input_name_);

    if(input.shape() != expected_shape){
        throw std::runtime_error(
            "Input shape mismatch: expected " + shape_to_string(expected_shape) +
            ", got " + shape_to_string(input.shape())
        );
    }

    context.clear();
    store_runtime_tensor(context, plan.input_name_, input);

    for(size_t node_position = 0; node_position < plan.node_indices_.size(); node_position++){
        const size_t node_index = plan.node_indices_[node_position];

        if(node_index >= nodes_.size()){
            throw std::logic_error("ExecutionPlan contains an invalid node.");
        }

        const Node& node = nodes_[node_index];

        execute_node(context, node);

        for(const std::string& input_name : node.inputs){
            const TensorMemoryInfo& info = find_memory_info_or_throw(plan, input_name);

            if(info.is_intermediate && info.last_use == node_position){
                context.erase_tensor(input_name);
            }
        }
    }

    return context.tensor(plan.output_name_);
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

    plan.memory_infos_.push_back(make_memory_info(
        plan.input_name_,
        plan.shape(plan.input_name_),
        true,
        plan.input_name_ == plan.output_name_,
        false
    )); // input 

    std::vector<std::string> constant_names;
    constant_names.reserve(constants_.size());

    // Add constant tensors to memory info
    for(const auto& [name, tensor] : constants_){
        (void)tensor;
        constant_names.push_back(name);
    }

    // Sort constant names to ensure deterministic order in memory info
    std::sort(constant_names.begin(), constant_names.end());

    // Add constant tensors to memory info
    for(const std::string& name : constant_names){
        plan.memory_infos_.push_back(make_memory_info(
            name,
            plan.shape(name),
            false,
            name == plan.output_name_,
            true
        ));
    }

    // Add intermediate tensors to memory info
    for(size_t node_index : plan.node_indices_){
        const Node& node = nodes_[node_index];

        plan.memory_infos_.push_back(make_memory_info(
            node.output,
            plan.shape(node.output),
            false,
            node.output == plan.output_name_,
            false
        ));
    }

    std::unordered_map<std::string, size_t> memory_info_indices;
    memory_info_indices.reserve(plan.memory_infos_.size());

    for(size_t i = 0; i < plan.memory_infos_.size(); i++){
        const std::string& name = plan.memory_infos_[i].name;
        auto [it, inserted] = memory_info_indices.emplace(name, i);
        (void)it;

        if(!inserted){
            throw std::logic_error(
                "ExecutionPlan memory analysis found duplicate tensor '" + name + "'."
            );
        }
    }

    for(size_t node_position = 0; node_position < plan.node_indices_.size(); node_position++){
        const Node& node = nodes_.at(plan.node_indices_[node_position]);

        auto output_it = memory_info_indices.find(node.output);

        if(output_it == memory_info_indices.end()){
            throw std::logic_error(
                "ExecutionPlan memory analysis missing output tensor '" + node.output + "'."
            );
        }

        plan.memory_infos_.at(output_it->second).produced_at = node_position;

        for(const std::string& input : node.inputs){
            update_tensor_use(plan.memory_infos_, memory_info_indices, input, node_position);
        }
    }

    return plan;
}

Tensor Graph::forward(
    const std::string& input_name,
    const Tensor& input,
    const std::string& output_name
) const{
    ExecutionPlan plan = compile(
        input_name,
        input.shape(),
        output_name
    );

    ExecutionContext context;
    return run(plan, context, input);
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

std::string Graph::dump_constants() const{
    std::ostringstream oss;
    oss << "Constants:\n";

    if(constants_.empty()){
        oss << "  <empty>\n";
        return oss.str();
    }

    for(const auto& [name, tensor] : constants_){
        oss << "  " << name
            << ": shape=" << shape_to_string(tensor.shape())
            << ", numel=" << tensor.numel() << "\n";
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

std::string Graph::dump_memory_plan(const ExecutionPlan& plan) const{
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

    size_t input_bytes = 0;
    size_t constant_bytes = 0;
    size_t intermediate_bytes = 0;
    size_t output_bytes = 0;

    std::ostringstream oss;
    oss << "Memory plan:\n";
    oss << "  tensors:\n";

    for(const TensorMemoryInfo& info : plan.memory_infos_){
        oss << "    " << info.name
            << ": shape=" << shape_to_string(info.shape)
            << ", numel=" << info.numel
            << ", bytes=" << info.byte_size
            << ", role=" << memory_role_string(info)
            << ", produced_at=" << lifetime_index_to_string(info.produced_at)
            << ", first_use=" << lifetime_index_to_string(info.first_use)
            << ", last_use=" << lifetime_index_to_string(info.last_use) 
            << "\n";
        
        if(info.is_input){
            input_bytes += info.byte_size;
        }else if(info.is_output){
            output_bytes += info.byte_size;
        }else if(info.is_constant){
            constant_bytes += info.byte_size;
        }else if(info.is_intermediate){
            intermediate_bytes += info.byte_size;
        }
    }

    oss << "  memory usage:\n";
    oss << "    input: " << input_bytes << " bytes\n";
    oss << "    constant: " << constant_bytes << " bytes\n";
    oss << "    intermediate: " << intermediate_bytes << " bytes\n";
    oss << "    output: " << output_bytes << " bytes\n";

    return oss.str();
}

}

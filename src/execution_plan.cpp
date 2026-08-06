#include "execution_plan.h"

#include <sstream>
#include <stdexcept>

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

std::string role_to_string(ValueRole role){
    switch(role){
        case ValueRole::Input: return "input";
        case ValueRole::Initializer: return "initializer";
        case ValueRole::Intermediate: return "intermediate";
    }

    throw std::logic_error("Unknown ValueRole.");
}

std::string lifetime_to_string(size_t value){
    return value == lifetime_npos ? "-" : std::to_string(value);
}

}

const ValueNameIndexMap& ExecutionPlan::value_names() const noexcept{
    return value_names_;
}

size_t ExecutionPlan::value_count() const noexcept{
    return values_.size();
}

size_t ExecutionPlan::node_count() const noexcept{
    return nodes_.size();
}

ValueIndex ExecutionPlan::input_index() const noexcept{
    return input_index_;
}

ValueIndex ExecutionPlan::output_index() const noexcept{
    return output_index_;
}

ValueIndex ExecutionPlan::value_index(std::string_view name) const{
    return value_names_.get(name);
}

const ValueInfo& ExecutionPlan::value_info(ValueIndex index) const{
    if(static_cast<size_t>(index) >= values_.size()){
        throw std::out_of_range("ExecutionPlan ValueIndex is out of range.");
    }

    return values_[index];
}

const std::vector<ValueInfo>& ExecutionPlan::values() const noexcept{
    return values_;
}

const std::vector<NodeExecutionPlan>& ExecutionPlan::nodes() const noexcept{
    return nodes_;
}

const Shape& ExecutionPlan::shape(ValueIndex index) const{
    return value_info(index).shape;
}

const Shape& ExecutionPlan::shape(std::string_view name) const{
    return shape(value_index(name));
}

std::string ExecutionPlan::dump() const{
    std::ostringstream oss;
    oss << "Execution plan:\n";
    oss << "  input: " << value_names_.name(input_index_) << " " << shape_to_string(shape(input_index_)) << "\n";
    oss << "  output: " << value_names_.name(output_index_) << " " << shape_to_string(shape(output_index_)) << "\n";
    oss << "  nodes:\n";

    if(nodes_.empty()){
        oss << "    <empty>\n";
        return oss.str();
    }

    for(size_t position = 0; position < nodes_.size(); position++){
        const NodeExecutionPlan& node = nodes_[position];
        oss << "    [" << position << "] " << node.name << "(";

        for(size_t i = 0; i < node.inputs.size(); i++){
            if(i > 0){
                oss << ", ";
            }

            const ValueIndex input = node.inputs[i];
            oss << value_names_.name(input) << " " << shape_to_string(shape(input));
        }

        oss << ") -> " << value_names_.name(node.output) << " "
            << shape_to_string(shape(node.output)) << "\n";
    }

    return oss.str();
}

std::string ExecutionPlan::dump_memory_plan() const{
    size_t input_bytes = 0;
    size_t initializer_bytes = 0;
    size_t intermediate_bytes = 0;
    size_t output_bytes = 0;

    std::ostringstream oss;
    oss << "Value lifetime plan:\n";
    oss << "  values:\n";

    for(const ValueInfo& info : values_){
        oss << "    " << info.name
            << ": shape=" << shape_to_string(info.shape)
            << ", numel=" << info.numel
            << ", bytes=" << info.byte_size
            << ", role=" << role_to_string(info.role)
            << ", graph_output=" << (info.is_graph_output ? "yes" : "no")
            << ", produced_at=" << lifetime_to_string(info.produced_at)
            << ", first_use=" << lifetime_to_string(info.first_use)
            << ", last_use=" << lifetime_to_string(info.last_use)
            << ", consumers=" << info.consumer_count << "\n";

        switch(info.role){
            case ValueRole::Input: input_bytes += info.byte_size; break;
            case ValueRole::Initializer: initializer_bytes += info.byte_size; break;
            case ValueRole::Intermediate: intermediate_bytes += info.byte_size; break;
        }

        if(info.is_graph_output){
            output_bytes += info.byte_size;
        }
    }

    oss << "  logical bytes:\n";
    oss << "    input: " << input_bytes << "\n";
    oss << "    initializer: " << initializer_bytes << "\n";
    oss << "    intermediate: " << intermediate_bytes << "\n";
    oss << "    output: " << output_bytes << "\n";
    oss << "  note: buffer allocation/reuse is intentionally deferred to MemoryPlanner.\n";
    return oss.str();
}

std::string ExecutionPlan::dump_scheduler_plan() const{
    std::ostringstream oss;
    oss << "Scheduler plan:\n";

    if(nodes_.empty()){
        oss << "  <empty>\n";
        return oss.str();
    }

    for(size_t position = 0; position < nodes_.size(); position++){
        const NodeExecutionPlan& node = nodes_[position];
        oss << "  [" << position << "] " << node.name
            << ": dependencies=" << node.dependency_count << ", consumers=";

        if(node.consumers.empty()){
            oss << "<none>";
        }else{
            for(size_t i = 0; i < node.consumers.size(); i++){
                if(i > 0){
                    oss << ", ";
                }
                oss << "[" << node.consumers[i] << "] " << nodes_.at(node.consumers[i]).name;
            }
        }

        oss << "\n";
    }

    return oss.str();
}

}

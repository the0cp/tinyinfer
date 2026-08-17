#include "execution_frame.h"

#include <sstream>
#include <stdexcept>
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

}

ExecutionFrame::ExecutionFrame(const SessionState& session_state)
    : session_state_(session_state), values_(session_state.value_count()){
    for(ValueIndex index = 0; static_cast<size_t>(index) < values_.size(); index++){
        values_[index].borrowed = session_state_.initializer(index);
    }
}

void ExecutionFrame::bind_input(ValueIndex index, const Tensor& tensor){
    const ExecutionPlan& plan = session_state_.execution_plan();

    if(index != plan.input_index()){  // Check if the index corresponds to the graph input
        throw std::invalid_argument("ExecutionFrame can only bind the compiled graph input.");
    }

    const Shape& expected = plan.shape(index);

    if(tensor.shape() != expected){
        throw std::runtime_error(
            "Input shape mismatch for '" + std::string(plan.value_names().name(index)) +
            "': expected " + shape_to_string(expected) + ", got " + shape_to_string(tensor.shape()) + "."
        );
    }

    ValueSlot& slot = values_.at(index);

    if(slot.has_value()){
        throw std::logic_error("ExecutionFrame input slot is already bound.");
    }

    slot.borrowed = &tensor;
}

bool ExecutionFrame::has_value(ValueIndex index) const{
    return values_.at(index).has_value();
}

const Tensor& ExecutionFrame::value(ValueIndex index) const{
    const ValueSlot& slot = values_.at(index);

    if(slot.owned){
        return *slot.owned;
    }

    if(slot.borrowed){
        return *slot.borrowed;
    }

    throw std::logic_error(
        "ExecutionFrame value is unavailable: " +
        std::string(session_state_.execution_plan().value_names().name(index))
    );
}

void ExecutionFrame::set_value(ValueIndex index, Tensor tensor){
    ValueSlot& slot = values_.at(index);

    if(slot.has_value()){
        throw std::logic_error(
            "ExecutionFrame value already exists: " +
            std::string(session_state_.execution_plan().value_names().name(index))
        );
    }

    const Shape& expected = session_state_.execution_plan().shape(index);

    if(tensor.shape() != expected){
        throw std::logic_error(
            "Kernel produced an unexpected shape for value '" +
            std::string(session_state_.execution_plan().value_names().name(index)) + "'."
        );
    }

    slot.owned = std::move(tensor);
}

void ExecutionFrame::release(ValueIndex index){
    const ValueInfo& info = session_state_.execution_plan().value_info(index);

    if(info.role != ValueRole::Intermediate || info.is_graph_output){
        throw std::logic_error("Only non-output intermediate values may be released during execution.");
    }

    ValueSlot& slot = values_.at(index);
    slot.owned.reset();
    slot.borrowed = nullptr;
}

Tensor ExecutionFrame::fetch(ValueIndex index) const{
    return value(index);
}

std::string ExecutionFrame::dump_values() const{
    const ExecutionPlan& plan = session_state_.execution_plan();
    std::ostringstream oss;
    oss << "ExecutionFrame values:\n";

    for(ValueIndex index = 0; static_cast<size_t>(index) < values_.size(); index++){
        oss << "  [" << index << "] " << plan.value_names().name(index) << ": ";

        if(!values_[index].has_value()){
            oss << "<released>\n";
            continue;
        }

        const Tensor& tensor = value(index);
        oss << "shape=" << shape_to_string(tensor.shape())
            << ", numel=" << tensor.numel()
            << (values_[index].borrowed ? ", storage=borrowed" : ", storage=owned")
            << "\n";
    }

    return oss.str();
}

}

#include <tinyinfer/runtime/execution_frame.h>

#include <tinyinfer/runtime/profiler.h>

#include <sstream>
#include <stdexcept>

namespace tinyinfer{

ExecutionFrame::ExecutionFrame(const SessionState& session_state, RunProfiler* profiler)
    : session_state_(session_state), profiler_(profiler), values_(session_state.value_count()){
    const MemoryPlan& memory_plan = session_state_.memory_plan();
    if(memory_plan.enabled()){
        arena_.emplace(memory_plan);
        if(profiler_){
            profiler_->memory_plan_applied(
                memory_plan.buffer_count(),
                memory_plan.arena_bytes(),
                memory_plan.planned_value_count(),
                memory_plan.reuse_count()
            );
        }
    }

    for(ValueIndex index = 0; static_cast<size_t>(index) < values_.size(); index++){
        values_[index].borrowed = session_state_.initializer(index);
    }
}

Tensor& ExecutionFrame::allocate_output(ValueIndex index){
    ValueSlot& slot = values_.at(index);

    if(slot.has_value()){
        throw std::logic_error(
            "ExecutionFrame value already exists: " +
            std::string(session_state_.execution_plan().value_names().name(index))
        );
    }

    const ValueInfo& info = session_state_.execution_plan().value_info(index);
    if(info.role != ValueRole::Intermediate){
        throw std::logic_error("Only intermediate values may be allocated as kernel outputs.");
    }

    const MemoryPlan& memory_plan = session_state_.memory_plan();
    if(arena_ && memory_plan.has_allocation(index)){
        slot.owned.emplace(arena_->make_tensor(index, info));
    }else{
        slot.owned.emplace(info.shape);
        if(profiler_){
            profiler_->managed_buffer_allocated(info.byte_size);
        }
    }

    if(profiler_){
        profiler_->value_allocated(index, info.byte_size);
    }

    return *slot.owned;
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

    const ValueInfo& input_info = plan.value_info(index);
    if(tensor.dtype() != input_info.dtype){
        throw std::runtime_error(
            "Input dtype mismatch for '" + std::string(plan.value_names().name(index)) + "'."
        );
    }
    if(!tensor.is_contiguous()){
        throw std::runtime_error(
            "Input layout mismatch for '" + std::string(plan.value_names().name(index)) +
            "': compiled kernels require contiguous input."
        );
    }

    ValueSlot& slot = values_.at(index);

    if(slot.has_value()){
        throw std::logic_error("ExecutionFrame input slot is already bound.");
    }

    slot.borrowed = &tensor;
}

void ExecutionFrame::validate_output(ValueIndex index) const{
    const ValueInfo& info = session_state_.execution_plan().value_info(index);
    const Tensor& tensor = value(index);
    if(tensor.shape() != info.shape || tensor.dtype() != info.dtype || !tensor.is_contiguous()){
        throw std::logic_error("Kernel output violates its compiled shape, dtype or layout.");
    }
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

void ExecutionFrame::release(ValueIndex index){
    const ValueInfo& info = session_state_.execution_plan().value_info(index);

    if(info.role != ValueRole::Intermediate || info.is_graph_output){
        throw std::logic_error("Only non-output intermediate values may be released during execution.");
    }

    ValueSlot& slot = values_.at(index);

    if(slot.owned && profiler_){
        profiler_->value_released(index, info.byte_size);
    }

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
            << ", dtype=" << data_type_name(tensor.dtype())
            << ", numel=" << tensor.numel()
            << ", layout=" << (tensor.is_contiguous() ? "contiguous" : "strided")
            << ", byte_offset=" << tensor.byte_offset()
            << (values_[index].borrowed ? ", storage=borrowed" : ", storage=owned")
            << "\n";
    }

    return oss.str();
}

}

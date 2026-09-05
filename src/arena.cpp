#include "arena.h"

#include <stdexcept>

namespace tinyinfer{

Arena::Arena(const MemoryPlan& plan) : plan_(plan){
    if(!plan.enabled()){
        throw std::invalid_argument("Cannot create an Arena from a disabled MemoryPlan.");
    }

    buffers_.reserve(plan.buffer_count());
    for(const BufferPlan& buffer : plan.buffers()){
        if(static_cast<size_t>(buffer.id) != buffers_.size()){
            throw std::logic_error("MemoryPlan buffer ids must be dense and ordered.");
        }
        buffers_.push_back(make_cpu_buffer(buffer.size_bytes));
    }
}

Tensor Arena::make_tensor(ValueIndex value, const ValueInfo& info) const{
    const ValueAllocation& allocation = plan_.allocation(value);
    const BufferPlan& buffer_plan = plan_.buffers().at(allocation.buffer);

    if(allocation.size_bytes != info.byte_size || allocation.alignment < element_alignment(info.dtype)){
        throw std::logic_error("MemoryPlan allocation does not match ValueInfo.");
    }
    if(allocation.byte_offset > buffer_plan.size_bytes ||
       allocation.size_bytes > buffer_plan.size_bytes - allocation.byte_offset){
        throw std::logic_error("MemoryPlan allocation exceeds its buffer.");
    }

    return Tensor(
        buffers_.at(allocation.buffer),
        allocation.byte_offset,
        info.shape,
        info.dtype
    );
}

}

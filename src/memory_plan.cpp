#include "memory_plan.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tinyinfer{

namespace{

size_t checked_add(size_t left, size_t right){
    if(right > std::numeric_limits<size_t>::max() - left){
        throw std::overflow_error("MemoryPlan arena size overflows size_t.");
    }

    return left + right;
}

BufferId next_buffer_id(size_t buffer_count){
    if(buffer_count >= static_cast<size_t>(invalid_buffer_id)){
        throw std::overflow_error("MemoryPlan has too many buffers.");
    }

    return static_cast<BufferId>(buffer_count);
}

size_t lifetime_end(const ValueInfo& value){
    return value.last_use == lifetime_npos ? value.produced_at : value.last_use;
}

}

MemoryPlanningPolicy MemoryPlan::policy() const noexcept{
    return policy_;
}

bool MemoryPlan::enabled() const noexcept{
    return policy_ != MemoryPlanningPolicy::Disabled;
}

bool MemoryPlan::has_allocation(ValueIndex value) const{
    return static_cast<size_t>(value) < allocations_.size() && allocations_[value].has_value();
}

const ValueAllocation& MemoryPlan::allocation(ValueIndex value) const{
    if(!has_allocation(value)){
        throw std::out_of_range("MemoryPlan has no allocation for this value.");
    }

    return *allocations_[value];
}

const std::vector<BufferPlan>& MemoryPlan::buffers() const noexcept{
    return buffers_;
}

size_t MemoryPlan::planned_value_count() const noexcept{
    return planned_value_count_;
}

size_t MemoryPlan::buffer_count() const noexcept{
    return buffers_.size();
}

size_t MemoryPlan::reuse_count() const noexcept{
    assert(planned_value_count_ >= buffers_.size());
    return planned_value_count_ - buffers_.size();
}

size_t MemoryPlan::arena_bytes() const noexcept{
    return arena_bytes_;
}

std::string MemoryPlan::dump(const ExecutionPlan& execution_plan) const{
    if(execution_plan.value_count() != allocations_.size()){
        throw std::invalid_argument("MemoryPlan and ExecutionPlan value counts do not match.");
    }

    std::ostringstream oss;
    oss << "MemoryPlan: policy=";

    switch(policy_){
    case MemoryPlanningPolicy::Disabled:
        oss << "disabled";
        break;
    case MemoryPlanningPolicy::SequentialReuse:
        oss << "sequential-reuse";
        break;
    case MemoryPlanningPolicy::Dedicated:
        oss << "dedicated";
        break;
    }

    oss << ", planned_values=" << planned_value_count_
        << ", buffers=" << buffers_.size()
        << ", reuses=" << reuse_count()
        << ", arena_bytes=" << arena_bytes_ << "\n";

    for(const BufferPlan& buffer : buffers_){
        oss << "  buffer[" << buffer.id << "]: bytes=" << buffer.size_bytes
            << ", alignment=" << buffer.alignment
            << ", reusable=" << (buffer.reusable ? "yes" : "no")
            << ", values=";

        bool first = true;
        for(ValueIndex value = 0; static_cast<size_t>(value) < allocations_.size(); value++){
            if(!allocations_[value] || allocations_[value]->buffer != buffer.id){
                continue;
            }

            if(!first){
                oss << ", ";
            }
            first = false;

            const ValueInfo& info = execution_plan.value_info(value);
            oss << info.name << "[" << info.produced_at << ",";
            if(info.last_use == lifetime_npos){
                oss << info.produced_at;
            }else{
                oss << info.last_use;
            }
            oss << "]";
        }

        oss << "\n";
    }

    return oss.str();
}

MemoryPlan MemoryPlanner::build(const ExecutionPlan& execution_plan, MemoryPlanningPolicy policy){
    // Reject unknown policies rather than silently selecting sequential reuse.
    switch(policy){
    case MemoryPlanningPolicy::Disabled:
    case MemoryPlanningPolicy::SequentialReuse:
    case MemoryPlanningPolicy::Dedicated:
        break;
    default:
        throw std::invalid_argument("Unknown memory planning policy.");
    }

    MemoryPlan plan;
    plan.policy_ = policy;
    plan.allocations_.resize(execution_plan.value_count());

    if(policy == MemoryPlanningPolicy::Disabled){
        return plan;
    }

    auto create_buffer = [&](const ValueInfo& value, bool reusable){
        const BufferId id = next_buffer_id(plan.buffers_.size());
        const size_t alignment = element_alignment(value.dtype);
        plan.buffers_.push_back(BufferPlan{id, value.byte_size, alignment, reusable});
        plan.arena_bytes_ = checked_add(plan.arena_bytes_, value.byte_size);
        return id;
    };

    auto assign = [&](ValueIndex value, BufferId buffer){
        const ValueInfo& info = execution_plan.value_info(value);
        plan.allocations_[value] = ValueAllocation{
            value,
            buffer,
            0,
            info.byte_size,
            element_alignment(info.dtype)
        };
        plan.planned_value_count_++;
    };

    if(policy == MemoryPlanningPolicy::Dedicated){
        for(ValueIndex value = 0; static_cast<size_t>(value) < execution_plan.value_count(); value++){
            const ValueInfo& info = execution_plan.value_info(value);
            if(info.role == ValueRole::Intermediate){
                assign(value, create_buffer(info, false));
            }
        }
        return plan;
    }

    std::vector<ValueIndex> reusable_values;
    std::vector<ValueIndex> output_values;

    for(ValueIndex value = 0; static_cast<size_t>(value) < execution_plan.value_count(); value++){
        const ValueInfo& info = execution_plan.value_info(value);
        if(info.role != ValueRole::Intermediate){
            continue;
        }
        if(info.produced_at == lifetime_npos){
            throw std::logic_error("Intermediate value has no producer in MemoryPlanner.");
        }

        (info.is_graph_output ? output_values : reusable_values).push_back(value);
    }

    std::sort(reusable_values.begin(), reusable_values.end(), [&](ValueIndex left, ValueIndex right){
        const size_t left_position = execution_plan.value_info(left).produced_at;
        const size_t right_position = execution_plan.value_info(right).produced_at;
        return left_position != right_position ? left_position < right_position : left < right;
    });

    struct ActiveBuffer{
        BufferId buffer;
        size_t end;
    };

    std::vector<ActiveBuffer> active;
    std::vector<BufferId> available;

    for(ValueIndex value : reusable_values){
        const ValueInfo& info = execution_plan.value_info(value);
        const size_t begin = info.produced_at;

        auto active_it = active.begin();
        while(active_it != active.end()){
            // Lifetimes are inclusive: an input read by node N overlaps that
            // node's output, so only end < begin is safe to reuse.
            if(active_it->end < begin){
                available.push_back(active_it->buffer);
                active_it = active.erase(active_it);
            }else{
                ++active_it;
            }
        }

        const size_t required_alignment = element_alignment(info.dtype);
        auto best = available.end();

        for(auto candidate = available.begin(); candidate != available.end(); ++candidate){
            const BufferPlan& buffer = plan.buffers_.at(*candidate);
            if(buffer.size_bytes < info.byte_size || buffer.alignment < required_alignment){
                continue;
            }

            if(best == available.end()){
                best = candidate;
                continue;
            }

            const BufferPlan& current_best = plan.buffers_.at(*best);
            if(buffer.size_bytes < current_best.size_bytes ||
               (buffer.size_bytes == current_best.size_bytes && buffer.id < current_best.id)){
                best = candidate;
            }
        }

        BufferId buffer = invalid_buffer_id;
        if(best == available.end()){
            buffer = create_buffer(info, true);
        }else{
            buffer = *best;
            available.erase(best);
        }

        assign(value, buffer);
        active.push_back(ActiveBuffer{buffer, lifetime_end(info)});
    }

    // Graph outputs must remain valid after ExecutionFrame and Arena are gone.
    // A dedicated block prevents a later run-local value from aliasing them.
    for(ValueIndex value : output_values){
        const ValueInfo& info = execution_plan.value_info(value);
        assign(value, create_buffer(info, false));
    }

    return plan;
}

}

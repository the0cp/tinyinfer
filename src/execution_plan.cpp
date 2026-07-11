#include "execution_plan.h"

#include <stdexcept>

namespace tinyinfer{

const std::string& ExecutionPlan::input_name() const noexcept{
    return input_name_;
}

const std::string& ExecutionPlan::output_name() const noexcept{
    return output_name_;
}

const Shape& ExecutionPlan::shape(const std::string& tensor_name) const{
    auto it = shapes_.find(tensor_name);

    if(it == shapes_.end()){
        throw std::runtime_error(
            "ExecutionPlan has no shape for tensor '" +
            tensor_name
        );
    }

    return it->second;
}

const ShapeTable& ExecutionPlan::shapes() const noexcept{
    return shapes_;
}

const std::vector<std::string>& ExecutionPlan::execution_order() const noexcept{
    return execution_order_;
}

const std::vector<TensorMemoryInfo>& ExecutionPlan::memory_infos() const noexcept{
    return memory_infos_;
}

const std::vector<NodeScheduleInfo>& ExecutionPlan::schedule_infos() const noexcept{
    return schedule_infos_;
}

size_t ExecutionPlan::node_count() const noexcept{
    return node_indices_.size();
}

}
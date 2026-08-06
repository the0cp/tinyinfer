#include "op_kernel.h"

#include "execution_frame.h"
#include "execution_plan.h"

#include <stdexcept>
#include <utility>

namespace tinyinfer{

OpKernelContext::OpKernelContext(const NodeExecutionPlan& node_plan, ExecutionFrame& frame)
    : node_plan_(node_plan), frame_(frame){}

size_t OpKernelContext::input_count() const noexcept{
    return node_plan_.inputs.size();
}

const Tensor& OpKernelContext::input(size_t index) const{
    if(index >= node_plan_.inputs.size()){
        throw std::out_of_range("OpKernel input index is out of range.");
    }

    return frame_.value(node_plan_.inputs[index]);
}

void OpKernelContext::set_output(Tensor tensor){
    frame_.set_value(node_plan_.output, std::move(tensor));
}

}

#include "op_kernel.h"

#include "execution_frame.h"
#include "execution_plan.h"

#include <stdexcept>
#include <utility>

namespace tinyinfer{

OpKernelContext::OpKernelContext(
    const NodeExecutionPlan& node_plan,
    ExecutionFrame& frame,
    ThreadPool* intra_op_thread_pool
)
    : node_plan_(node_plan),
      frame_(frame),
      intra_op_thread_pool_(intra_op_thread_pool){}

size_t OpKernelContext::input_count() const noexcept{
    return node_plan_.inputs.size();
}

const Tensor& OpKernelContext::input(size_t index) const{
    if(index >= node_plan_.inputs.size()){
        throw std::out_of_range("OpKernel input index is out of range.");
    }

    return frame_.value(node_plan_.inputs[index]);
}

Tensor& OpKernelContext::output(){
    if(legacy_output_set_){
        throw std::logic_error("Cannot mix output() with set_output().");
    }
    if(!output_){
        output_ = &frame_.allocate_output(node_plan_.output);
        output_buffer_ = output_->buffer();
        output_offset_ = output_->byte_offset();
    }
    return *output_;
}

void OpKernelContext::set_output(Tensor tensor){
    if(output_ || legacy_output_set_){
        throw std::logic_error("Kernel output has already been requested or submitted.");
    }
    frame_.set_value(node_plan_.output, std::move(tensor));
    legacy_output_set_ = true;
}

void OpKernelContext::validate_output() const{
    frame_.validate_output(node_plan_.output);
    if(output_ && (output_->buffer() != output_buffer_ || output_->byte_offset() != output_offset_)){
        throw std::logic_error("Kernel replaced runtime-provided output storage.");
    }
}

ThreadPool* OpKernelContext::intra_op_thread_pool() const noexcept{
    return intra_op_thread_pool_;
}

}

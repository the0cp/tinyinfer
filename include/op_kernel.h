#pragma once

#include "tensor.h"

#include <cstddef>

namespace tinyinfer{

class ExecutionFrame;
struct NodeExecutionPlan;

class OpKernelContext{
public:
    OpKernelContext(const NodeExecutionPlan& node_plan, ExecutionFrame& frame);

    size_t input_count() const noexcept;
    const Tensor& input(size_t index) const;
    void set_output(Tensor tensor);

private:
    const NodeExecutionPlan& node_plan_;
    ExecutionFrame& frame_;
};

class OpKernel{
public:
    virtual ~OpKernel() = default;
    virtual void compute(OpKernelContext& context) const = 0;
};

}

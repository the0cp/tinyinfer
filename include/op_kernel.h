#pragma once

#include "tensor.h"

#include <cstddef>
#include <memory>

namespace tinyinfer{

class ExecutionFrame;
class ThreadPool;
struct NodeExecutionPlan;

class OpKernelContext{
public:
    OpKernelContext(
        const NodeExecutionPlan& node_plan,
        ExecutionFrame& frame,
        ThreadPool* intra_op_thread_pool = nullptr
    );

    size_t input_count() const noexcept;
    const Tensor& input(size_t index) const;
    // Repeated calls return the same output. Write all elements before returning
    // from compute(); do not replace its storage or retain activation handles.
    Tensor& output();
    // Legacy submission may copy into planned storage. Cannot mix with output().
    void set_output(Tensor tensor);
    // Called by executors after compute(), before consumers may run.
    void validate_output() const;
    ThreadPool* intra_op_thread_pool() const noexcept;

private:
    const NodeExecutionPlan& node_plan_;
    ExecutionFrame& frame_;
    ThreadPool* intra_op_thread_pool_ = nullptr;
    Tensor* output_ = nullptr;
    std::shared_ptr<Buffer> output_buffer_;
    size_t output_offset_ = 0;
    bool legacy_output_set_ = false;
};

class OpKernel{
public:
    virtual ~OpKernel() = default;
    // Synchronous: inputs are read-only and all output writes must complete
    // before return. Activation references/handles must not escape compute().
    virtual void compute(OpKernelContext& context) const = 0;
};

}

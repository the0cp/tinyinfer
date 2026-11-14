#include "builtin_kernels.h"

#include <tinyinfer/runtime/kernel_registry.h>
#include <tinyinfer/runtime/threading_policy.h>
#include <tinyinfer/compute/ops.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer{

namespace{

class SerialLinearKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        linear_out(context.input(0), context.input(1), context.input(2), context.output());
    }
};

class ThreadedLinearKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        ThreadPool* pool = context.intra_op_thread_pool();
        if(!pool || pool->size() < 2){
            throw std::logic_error(
                "The compiled threaded Linear kernel has no intra-op ThreadPool."
            );
        }
        threadpool_linear_out(
            context.input(0),
            context.input(1),
            context.input(2),
            context.output(),
            *pool,
            pool->size()
        );
    }
};

class ReluKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        relu_out(context.input(0), context.output());
    }
};

class SoftmaxKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        softmax_out(context.input(0), context.output());
    }
};

class AddKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        add_out(context.input(0), context.input(1), context.output());
    }
};

uint64_t as_cost(size_t value) noexcept{
    if constexpr(sizeof(size_t) > sizeof(uint64_t)){
        if(value > static_cast<size_t>(std::numeric_limits<uint64_t>::max())){
            return std::numeric_limits<uint64_t>::max();
        }
    }
    return static_cast<uint64_t>(value);
}

std::string require_arity(const KernelSelectionContext& context, size_t input_count){
    if(context.input_shapes.size() != input_count){
        return "expected " + std::to_string(input_count) + " inputs";
    }
    return {};
}

std::string require_linear_shape(const KernelSelectionContext& context){
    if(const std::string arity = require_arity(context, 3); !arity.empty()){
        return arity;
    }
    if(context.input_shapes[0].size() != 2 || context.input_shapes[1].size() != 2 ||
       context.input_shapes[2].size() != 1 || context.output_shape.size() != 2){
        return "requires rank-2 input/weight/output and rank-1 bias";
    }
    return {};
}

KernelMatch reject_or_accept(std::string rejection, uint64_t cost){
    return rejection.empty()
        ? KernelMatch::accept(cost)
        : KernelMatch::reject(std::move(rejection));
}

}

void register_builtin_kernels(KernelRegistry& registry){
    registry.register_kernel(
        OpType::Linear,
        KernelCandidate{
            .name = "cpu.linear.serial",
            .threading = KernelThreading::Serial,
            .priority = 0,
            .match = [](const KernelSelectionContext& context){
                if(const std::string shape = require_linear_shape(context); !shape.empty()){
                    return KernelMatch::reject(shape);
                }
                return KernelMatch::accept(estimate_linear_cost(
                    context.input_shapes[0][0],
                    context.input_shapes[0][1],
                    context.input_shapes[1][1],
                    1
                ));
            },
            .factory = [](const Node&, const KernelSelectionContext&){
                return std::make_unique<SerialLinearKernel>();
            }
        }
    );
    registry.register_kernel(
        OpType::Linear,
        KernelCandidate{
            .name = "cpu.linear.threaded",
            .threading = KernelThreading::IntraOp,
            .priority = 0,
            .match = [](const KernelSelectionContext& context){
                if(const std::string shape = require_linear_shape(context); !shape.empty()){
                    return KernelMatch::reject(shape);
                }
                return KernelMatch::accept(estimate_linear_cost(
                    context.input_shapes[0][0],
                    context.input_shapes[0][1],
                    context.input_shapes[1][1],
                    context.effective_intra_op_threads
                ));
            },
            .factory = [](const Node&, const KernelSelectionContext&){
                return std::make_unique<ThreadedLinearKernel>();
            }
        }
    );
    registry.register_kernel(
        OpType::ReLU,
        KernelCandidate{
            .name = "cpu.relu.reference",
            .threading = KernelThreading::Serial,
            .priority = 0,
            .match = [](const KernelSelectionContext& context){
                return reject_or_accept(
                    require_arity(context, 1),
                    as_cost(tensor_numel(context.output_shape))
                );
            },
            .factory = [](const Node&, const KernelSelectionContext&){
                return std::make_unique<ReluKernel>();
            }
        }
    );
    registry.register_kernel(
        OpType::Softmax,
        KernelCandidate{
            .name = "cpu.softmax.reference",
            .threading = KernelThreading::Serial,
            .priority = 0,
            .match = [](const KernelSelectionContext& context){
                return reject_or_accept(
                    require_arity(context, 1),
                    as_cost(tensor_numel(context.output_shape))
                );
            },
            .factory = [](const Node&, const KernelSelectionContext&){
                return std::make_unique<SoftmaxKernel>();
            }
        }
    );
    registry.register_kernel(
        OpType::Add,
        KernelCandidate{
            .name = "cpu.add.reference",
            .threading = KernelThreading::Serial,
            .priority = 0,
            .match = [](const KernelSelectionContext& context){
                return reject_or_accept(
                    require_arity(context, 2),
                    as_cost(tensor_numel(context.output_shape))
                );
            },
            .factory = [](const Node&, const KernelSelectionContext&){
                return std::make_unique<AddKernel>();
            }
        }
    );
}

}

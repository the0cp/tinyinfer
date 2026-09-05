#include "kernel_registry.h"

#include "ops.h"

#include <stdexcept>
#include <utility>

namespace tinyinfer{

namespace{

class LinearKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        const Tensor& input = context.input(0);
        const Tensor& weight = context.input(1);
        const Tensor& bias = context.input(2);
        Tensor& output = context.output();
        linear_out(input, weight, bias, output);
    }
};

class ReluKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        const Tensor& input = context.input(0);
        Tensor& output = context.output();
        relu_out(input, output);
    }
};

class SoftmaxKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        const Tensor& input = context.input(0);
        Tensor& output = context.output();
        softmax_out(input, output);
    }
};

class AddKernel final : public OpKernel{
public:
    void compute(OpKernelContext& context) const override{
        const Tensor& left = context.input(0);
        const Tensor& right = context.input(1);
        Tensor& output = context.output();
        add_out(left, right, output);
    }
};

}

KernelRegistry::KernelRegistry(bool register_builtins){
    if(!register_builtins){
        return;
    }

    register_kernel(OpType::Linear, [](const Node&){ return std::make_unique<LinearKernel>(); });
    register_kernel(OpType::ReLU, [](const Node&){ return std::make_unique<ReluKernel>(); });
    register_kernel(OpType::Softmax, [](const Node&){ return std::make_unique<SoftmaxKernel>(); });
    register_kernel(OpType::Add, [](const Node&){ return std::make_unique<AddKernel>(); });
}

void KernelRegistry::register_kernel(OpType type, KernelFactory factory){
    if(!factory){
        throw std::invalid_argument("Cannot register an empty kernel factory.");
    }

    if(!factories_.emplace(type, std::move(factory)).second){
        throw std::runtime_error("Kernel type is already registered.");
    }
}

std::unique_ptr<OpKernel> KernelRegistry::create_kernel(const Node& node) const{
    auto it = factories_.find(node.op);

    if(it == factories_.end()){
        throw std::runtime_error("No CPU kernel is registered for node '" + node.name + "'.");
    }

    std::unique_ptr<OpKernel> kernel = it->second(node);

    if(!kernel){
        throw std::runtime_error("Kernel factory returned null for node '" + node.name + "'.");
    }

    return kernel;
}

}

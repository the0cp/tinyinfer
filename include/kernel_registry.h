#pragma once

#include "graph.h"
#include "op_kernel.h"

#include <functional>
#include <memory>
#include <unordered_map>

namespace tinyinfer{

using KernelFactory = std::function<std::unique_ptr<OpKernel>(const Node&)>;

class KernelRegistry{
public:
    explicit KernelRegistry(bool register_builtins = true);

    void register_kernel(OpType type, KernelFactory factory);
    std::unique_ptr<OpKernel> create_kernel(const Node& node) const;

private:
    std::unordered_map<OpType, KernelFactory, OpTypeHash> factories_;
};

}

#pragma once

#include "graph.h"
#include "kernel_selection.h"
#include "op_kernel.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tinyinfer{

using KernelFactory = std::function<std::unique_ptr<OpKernel>(
    const Node&,
    const KernelSelectionContext&
)>;
using KernelMatcher = std::function<KernelMatch(const KernelSelectionContext&)>;

struct KernelCandidate{
    std::string name;
    KernelThreading threading = KernelThreading::Serial;
    // Higher priority wins only when estimated costs are equal.
    int priority = 0;
    KernelMatcher match;
    KernelFactory factory;
};

struct SelectedKernel{
    std::unique_ptr<OpKernel> kernel;
    KernelSelectionRecord record;
};

class KernelRegistry{
public:
    explicit KernelRegistry(bool register_builtins = true);

    void register_kernel(OpType type, KernelCandidate candidate);
    SelectedKernel select_kernel(
        const Node& node,
        const KernelSelectionContext& context
    ) const;

private:
    std::unordered_map<OpType, std::vector<KernelCandidate>, OpTypeHash> candidates_;
};

}

#pragma once

#include "execution_plan.h"
#include "graph.h"

#include <memory>
#include <vector>

namespace tinyinfer{

class KernelRegistry;
class OpKernel;
class OperatorRegistry;

class SessionState{
public:
    SessionState(const SessionState&) = delete;
    SessionState& operator=(const SessionState&) = delete;
    SessionState(SessionState&&) noexcept;
    SessionState& operator=(SessionState&&) noexcept;
    ~SessionState();

    static SessionState build(
        Graph graph,
        const OperatorRegistry& operator_registry,
        const KernelRegistry& kernel_registry
    );

    const Graph& graph() const noexcept;
    const ExecutionPlan& execution_plan() const noexcept;
    const OpKernel& kernel(size_t index) const;

    const Tensor* initializer(ValueIndex index) const;
    size_t value_count() const noexcept;

private:
    SessionState() = default;

    Graph graph_;
    ExecutionPlan execution_plan_;
    std::vector<std::unique_ptr<OpKernel>> kernels_;
};

}

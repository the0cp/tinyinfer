#pragma once

#include "memory_plan.h"
#include "tensor.h"

#include <memory>
#include <vector>

namespace tinyinfer{

// An Arena is private to one inference run. The plan may reuse a Buffer across
// non-overlapping values, while separate runs never share mutable storage.
class Arena{
public:
    // Borrows plan: it must outlive this Arena. Only SessionState-backed plans
    // should be used while a run is active.
    explicit Arena(const MemoryPlan& plan);
    Arena(MemoryPlan&&) = delete;
    Arena(const MemoryPlan&&) = delete;

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) noexcept = default;
    Arena& operator=(Arena&&) = delete;

    Tensor make_tensor(ValueIndex value, const ValueInfo& info) const;

private:
    const MemoryPlan& plan_;
    std::vector<std::shared_ptr<Buffer>> buffers_;
};

}

#pragma once

#include "execution_plan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tinyinfer{

using BufferId = uint32_t;
inline constexpr BufferId invalid_buffer_id = static_cast<BufferId>(-1);

enum class MemoryPlanningPolicy{
    // No static placement; the frame allocates outputs on demand.
    Disabled,
    // Reuse assumes the exact serial order in ExecutionPlan::nodes().
    SequentialReuse,
    // Preallocate one block per value; safe for either executor.
    Dedicated
};

struct BufferPlan{
    // Stored explicitly for diagnostics; must equal the buffers_ vector index.
    BufferId id = invalid_buffer_id;
    size_t size_bytes = 0;
    size_t alignment = 1;
    // Planner eligibility only. Arena never makes reuse decisions at runtime.
    bool reusable = false;
};

struct ValueAllocation{
    // Must equal the allocations_ vector index.
    ValueIndex value = invalid_value_index;
    BufferId buffer = invalid_buffer_id;
    size_t byte_offset = 0;
    size_t size_bytes = 0;
    size_t alignment = 1;
};

class MemoryPlan{
public:
    MemoryPlanningPolicy policy() const noexcept;
    bool enabled() const noexcept;
    bool has_allocation(ValueIndex value) const;
    const ValueAllocation& allocation(ValueIndex value) const;
    const std::vector<BufferPlan>& buffers() const noexcept;

    size_t planned_value_count() const noexcept;
    size_t buffer_count() const noexcept;
    size_t reuse_count() const noexcept;
    size_t arena_bytes() const noexcept;

private:
    friend class MemoryPlanner;
    friend class SessionState;

    // SessionState supplies the matching execution plan, not an arbitrary caller.
    std::string dump(const ExecutionPlan& execution_plan) const;

    MemoryPlanningPolicy policy_ = MemoryPlanningPolicy::Disabled;
    std::vector<std::optional<ValueAllocation>> allocations_;
    std::vector<BufferPlan> buffers_;
    size_t planned_value_count_ = 0;
    size_t arena_bytes_ = 0;
};

class MemoryPlanner{
public:
    static MemoryPlan build(
        const ExecutionPlan& execution_plan,
        MemoryPlanningPolicy policy
    );
};

}

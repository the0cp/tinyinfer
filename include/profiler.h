#pragma once

#include "execution_plan.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace tinyinfer{

enum class NodeProfileStatus{
    NotStarted,
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled
};

std::string_view node_profile_status_name(NodeProfileStatus status) noexcept;

enum class ValueProfileAction{
    Allocated,
    Released
};

std::string_view value_profile_action_name(ValueProfileAction action) noexcept;

struct NodeProfile{
    size_t plan_position = 0;
    NodeIndex source_node_index = invalid_node_index;
    std::string node_name;
    NodeProfileStatus status = NodeProfileStatus::NotStarted;
    uint64_t queued_ns = 0;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    uint64_t thread_id = 0;
    size_t output_bytes = 0;
    std::string error;

    uint64_t queue_duration_ns() const noexcept;
    uint64_t compute_duration_ns() const noexcept;
};

struct ValueProfileEvent{
    ValueIndex value_index = invalid_value_index;
    std::string value_name;
    ValueProfileAction action = ValueProfileAction::Allocated;
    uint64_t timestamp_ns = 0;
    size_t bytes = 0;
    size_t live_bytes_after = 0;
};

struct RunProfile{
    bool success = false;
    uint64_t duration_ns = 0;
    size_t owned_value_allocations = 0;
    size_t cumulative_allocated_bytes = 0;
    size_t peak_live_bytes = 0;
    size_t live_bytes_at_finish = 0;
    // Buffer objects created by Frame/Arena (including empty buffers), and
    // cumulative requested payload capacity. Excludes kernel-owned allocations,
    // metadata, allocator overhead and RSS; these are not peak-memory counters.
    size_t managed_buffer_allocations = 0;
    size_t managed_allocated_bytes = 0;
    size_t legacy_output_submissions = 0;
    size_t planned_value_count = 0;
    size_t memory_reuse_count = 0;
    std::string error;
    std::vector<NodeProfile> nodes;
    std::vector<ValueProfileEvent> value_events;

    std::string summary() const;
    std::string to_chrome_trace_json() const;
    void save_chrome_trace(const std::string& path) const;
};

class RunProfiler{
public:
    explicit RunProfiler(const ExecutionPlan& plan);

    RunProfiler(const RunProfiler&) = delete;
    RunProfiler& operator=(const RunProfiler&) = delete;

    void node_queued(size_t plan_position);
    void node_started(size_t plan_position);
    void node_completed(size_t plan_position);
    void node_failed(size_t plan_position, std::string error);
    void node_cancelled(size_t plan_position);

    void value_allocated(ValueIndex index, size_t bytes);
    void value_released(ValueIndex index, size_t bytes);
    void managed_buffer_allocated(size_t bytes);
    void legacy_output_submitted();
    void memory_plan_applied(
        size_t buffer_count,
        size_t arena_bytes,
        size_t planned_value_count,
        size_t reuse_count
    );

    void finish(bool success, std::string error = {});
    RunProfile snapshot() const;

private:
    using Clock = std::chrono::steady_clock;

    uint64_t now_ns() const;
    NodeProfile& node_at(size_t plan_position);

    const ExecutionPlan& plan_;
    const Clock::time_point run_start_;
    mutable std::mutex mutex_;
    RunProfile profile_;
    size_t current_live_bytes_ = 0;
    bool finished_ = false;
};

}

#include "profiler.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace tinyinfer{

namespace{

uint64_t nanoseconds_between(uint64_t begin, uint64_t end) noexcept{
    return end >= begin ? end - begin : 0;
}

std::string json_escape(std::string_view text){
    std::ostringstream oss;

    for(unsigned char character : text){
        switch(character){
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            if(character < 0x20){
                oss << "\\u"
                    << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(character)
                    << std::dec << std::setfill(' ');
            }else{
                oss << static_cast<char>(character);
            }
        }
    }

    return oss.str();
}

double ns_to_us(uint64_t nanoseconds) noexcept{
    return static_cast<double>(nanoseconds) / 1000.0;
}

}

std::string_view node_profile_status_name(NodeProfileStatus status) noexcept{
    switch(status){
    case NodeProfileStatus::NotStarted:
        return "not-started";
    case NodeProfileStatus::Queued:
        return "queued";
    case NodeProfileStatus::Running:
        return "running";
    case NodeProfileStatus::Completed:
        return "completed";
    case NodeProfileStatus::Failed:
        return "failed";
    case NodeProfileStatus::Cancelled:
        return "cancelled";
    }

    return "unknown";
}

std::string_view value_profile_action_name(ValueProfileAction action) noexcept{
    switch(action){
    case ValueProfileAction::Allocated:
        return "allocated";
    case ValueProfileAction::Released:
        return "released";
    }

    return "unknown";
}

uint64_t NodeProfile::queue_duration_ns() const noexcept{
    if(status == NodeProfileStatus::NotStarted || status == NodeProfileStatus::Queued){
        return 0;
    }

    return nanoseconds_between(queued_ns, start_ns);
}

uint64_t NodeProfile::compute_duration_ns() const noexcept{
    if(status != NodeProfileStatus::Completed && status != NodeProfileStatus::Failed){
        return 0;
    }

    return nanoseconds_between(start_ns, end_ns);
}

std::string RunProfile::summary() const{
    std::ostringstream oss;
    oss << "Run profile: success=" << (success ? "yes" : "no")
        << ", duration_us=" << std::fixed << std::setprecision(3) << ns_to_us(duration_ns)
        << "\n";
    oss << "  memory: owned_value_allocations=" << owned_value_allocations
        << ", cumulative_bytes=" << cumulative_allocated_bytes
        << ", peak_live_bytes=" << peak_live_bytes
        << ", live_bytes_at_finish=" << live_bytes_at_finish
        << "\n";
    oss << "  managed memory: buffer_allocations=" << managed_buffer_allocations
        << ", allocated_bytes=" << managed_allocated_bytes
        << ", legacy_submissions=" << legacy_output_submissions
        << ", planned_values=" << planned_value_count
        << ", reuses=" << memory_reuse_count
        << "\n";

    if(!error.empty()){
        oss << "  error: " << error << "\n";
    }

    oss << "  nodes:\n";

    for(const NodeProfile& node : nodes){
        oss << "    [" << node.plan_position << "] " << node.node_name
            << ": status=" << node_profile_status_name(node.status)
            << ", queue_us=" << ns_to_us(node.queue_duration_ns())
            << ", compute_us=" << ns_to_us(node.compute_duration_ns())
            << ", output_bytes=" << node.output_bytes
            << ", thread=" << node.thread_id;

        if(!node.error.empty()){
            oss << ", error=" << node.error;
        }

        oss << "\n";
    }

    oss << "  value events:\n";

    for(const ValueProfileEvent& event : value_events){
        oss << "    " << event.value_name
            << ": action=" << value_profile_action_name(event.action)
            << ", timestamp_us=" << ns_to_us(event.timestamp_ns)
            << ", bytes=" << event.bytes
            << ", live_bytes_after=" << event.live_bytes_after
            << "\n";
    }

    return oss.str();
}

std::string RunProfile::to_chrome_trace_json() const{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n";

    bool first_event = true;

    auto begin_event = [&](){
        if(!first_event){
            oss << ",\n";
        }
        first_event = false;
        oss << "    ";
    };

    begin_event();
    oss << "{\"name\":\"InferenceSession::run\",\"cat\":\"runtime\","
        << "\"ph\":\"X\",\"ts\":0.000,\"dur\":" << ns_to_us(duration_ns)
        << ",\"pid\":1,\"tid\":0,\"args\":{\"success\":"
        << (success ? "true" : "false") << "}}";

    for(const NodeProfile& node : nodes){
        if(node.status == NodeProfileStatus::NotStarted){
            continue;
        }

        if(node.status != NodeProfileStatus::Queued && node.start_ns >= node.queued_ns){
            begin_event();
            oss << "{\"name\":\"queue: " << json_escape(node.node_name)
                << "\",\"cat\":\"scheduler\",\"ph\":\"X\",\"ts\":"
                << ns_to_us(node.queued_ns) << ",\"dur\":"
                << ns_to_us(node.queue_duration_ns())
                << ",\"pid\":1,\"tid\":0}";
        }

        if(node.status == NodeProfileStatus::Completed || node.status == NodeProfileStatus::Failed){
            begin_event();
            oss << "{\"name\":\"" << json_escape(node.node_name)
                << "\",\"cat\":\"kernel\",\"ph\":\"X\",\"ts\":"
                << ns_to_us(node.start_ns) << ",\"dur\":"
                << ns_to_us(node.compute_duration_ns())
                << ",\"pid\":1,\"tid\":" << node.thread_id
                << ",\"args\":{\"status\":\""
                << node_profile_status_name(node.status)
                << "\",\"output_bytes\":" << node.output_bytes;

            if(!node.error.empty()){
                oss << ",\"error\":\"" << json_escape(node.error) << "\"";
            }

            oss << "}}";
        }
    }

    for(const ValueProfileEvent& event : value_events){
        begin_event();
        oss << "{\"name\":\"" << value_profile_action_name(event.action)
            << ": " << json_escape(event.value_name)
            << "\",\"cat\":\"memory\",\"ph\":\"i\",\"s\":\"t\",\"ts\":"
            << ns_to_us(event.timestamp_ns)
            << ",\"pid\":1,\"tid\":0,\"args\":{\"bytes\":"
            << event.bytes << ",\"live_bytes_after\":"
            << event.live_bytes_after << "}}";
    }

    oss << "\n  ]\n}\n";
    return oss.str();
}

void RunProfile::save_chrome_trace(const std::string& path) const{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);

    if(!output){
        throw std::runtime_error("Failed to open Chrome Trace output: " + path);
    }

    output << to_chrome_trace_json();

    if(!output){
        throw std::runtime_error("Failed to write Chrome Trace output: " + path);
    }
}

RunProfiler::RunProfiler(const ExecutionPlan& plan)
    : plan_(plan), run_start_(Clock::now()){
    profile_.nodes.reserve(plan.node_count());

    for(size_t position = 0; position < plan.node_count(); position++){
        const NodeExecutionPlan& node = plan.nodes()[position];
        profile_.nodes.push_back(NodeProfile{
            position,
            node.source_node_index,
            node.name,
            NodeProfileStatus::NotStarted,
            0,
            0,
            0,
            0,
            plan.value_info(node.output).byte_size,
            {}
        });
    }
}

void RunProfiler::node_queued(size_t plan_position){
    std::lock_guard<std::mutex> lock(mutex_);
    NodeProfile& node = node_at(plan_position);

    if(node.status != NodeProfileStatus::NotStarted){
        throw std::logic_error("Profiler node was queued more than once.");
    }

    node.queued_ns = now_ns();
    node.status = NodeProfileStatus::Queued;
}

void RunProfiler::node_started(size_t plan_position){
    std::lock_guard<std::mutex> lock(mutex_);
    NodeProfile& node = node_at(plan_position);

    if(node.status != NodeProfileStatus::Queued){
        throw std::logic_error("Profiler node started before it was queued.");
    }

    node.start_ns = now_ns();
    node.thread_id = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    node.status = NodeProfileStatus::Running;
}

void RunProfiler::node_completed(size_t plan_position){
    std::lock_guard<std::mutex> lock(mutex_);
    NodeProfile& node = node_at(plan_position);

    if(node.status != NodeProfileStatus::Running){
        throw std::logic_error("Profiler completed a node that was not running.");
    }

    node.end_ns = now_ns();
    node.status = NodeProfileStatus::Completed;
}

void RunProfiler::node_failed(size_t plan_position, std::string error){
    std::lock_guard<std::mutex> lock(mutex_);
    NodeProfile& node = node_at(plan_position);

    if(node.status == NodeProfileStatus::Queued){
        node.start_ns = now_ns();
        node.thread_id = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }else if(node.status != NodeProfileStatus::Running){
        throw std::logic_error("Profiler failed a node that was not queued or running.");
    }

    node.end_ns = now_ns();
    node.status = NodeProfileStatus::Failed;
    node.error = std::move(error);
}

void RunProfiler::node_cancelled(size_t plan_position){
    std::lock_guard<std::mutex> lock(mutex_);
    NodeProfile& node = node_at(plan_position);

    if(node.status != NodeProfileStatus::Queued){
        throw std::logic_error("Profiler cancelled a node that was not queued.");
    }

    node.status = NodeProfileStatus::Cancelled;
    node.end_ns = now_ns();
}

void RunProfiler::value_allocated(ValueIndex index, size_t bytes){
    std::lock_guard<std::mutex> lock(mutex_);

    if(profile_.owned_value_allocations == std::numeric_limits<size_t>::max()){
        throw std::overflow_error("Profiler allocation count overflow.");
    }

    if(profile_.cumulative_allocated_bytes > std::numeric_limits<size_t>::max() - bytes ||
       current_live_bytes_ > std::numeric_limits<size_t>::max() - bytes){
        throw std::overflow_error("Profiler byte counter overflow.");
    }

    profile_.owned_value_allocations++;
    profile_.cumulative_allocated_bytes += bytes;
    current_live_bytes_ += bytes;
    profile_.peak_live_bytes = std::max(profile_.peak_live_bytes, current_live_bytes_);
    profile_.value_events.push_back(ValueProfileEvent{
        index,
        std::string(plan_.value_names().name(index)),
        ValueProfileAction::Allocated,
        now_ns(),
        bytes,
        current_live_bytes_
    });
}

void RunProfiler::value_released(ValueIndex index, size_t bytes){
    std::lock_guard<std::mutex> lock(mutex_);

    if(bytes > current_live_bytes_){
        throw std::logic_error("Profiler live-byte counter underflow.");
    }

    current_live_bytes_ -= bytes;
    profile_.value_events.push_back(ValueProfileEvent{
        index,
        std::string(plan_.value_names().name(index)),
        ValueProfileAction::Released,
        now_ns(),
        bytes,
        current_live_bytes_
    });
}

void RunProfiler::managed_buffer_allocated(size_t bytes){
    std::lock_guard<std::mutex> lock(mutex_);

    if(profile_.managed_buffer_allocations == std::numeric_limits<size_t>::max() ||
       profile_.managed_allocated_bytes > std::numeric_limits<size_t>::max() - bytes){
        throw std::overflow_error("Profiler managed allocation counter overflow.");
    }

    profile_.managed_buffer_allocations++;
    profile_.managed_allocated_bytes += bytes;
}

void RunProfiler::legacy_output_submitted(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(profile_.legacy_output_submissions == std::numeric_limits<size_t>::max()){
        throw std::overflow_error("Profiler legacy submission counter overflow.");
    }
    profile_.legacy_output_submissions++;
}

void RunProfiler::memory_plan_applied(
    size_t buffer_count,
    size_t arena_bytes,
    size_t planned_value_count,
    size_t reuse_count
){
    std::lock_guard<std::mutex> lock(mutex_);

    if(profile_.managed_buffer_allocations != 0 || profile_.managed_allocated_bytes != 0){
        throw std::logic_error("Profiler received a MemoryPlan after managed allocation began.");
    }

    profile_.managed_buffer_allocations = buffer_count;
    profile_.managed_allocated_bytes = arena_bytes;
    profile_.planned_value_count = planned_value_count;
    profile_.memory_reuse_count = reuse_count;
}

void RunProfiler::finish(bool success, std::string error){
    std::lock_guard<std::mutex> lock(mutex_);

    if(finished_){
        throw std::logic_error("RunProfiler was finished more than once.");
    }

    profile_.success = success;
    profile_.duration_ns = now_ns();
    profile_.live_bytes_at_finish = current_live_bytes_;
    profile_.error = std::move(error);
    finished_ = true;
}

RunProfile RunProfiler::snapshot() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return profile_;
}

uint64_t RunProfiler::now_ns() const{
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - run_start_);
    return static_cast<uint64_t>(elapsed.count());
}

NodeProfile& RunProfiler::node_at(size_t plan_position){
    return profile_.nodes.at(plan_position);
}

}

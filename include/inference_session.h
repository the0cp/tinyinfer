#pragma once

#include "graph_optimizer.h"
#include "kernel_registry.h"
#include "model_format.h"
#include "operator_registry.h"
#include "profiler.h"
#include "session_options.h"
#include "session_state.h"
#include "thread_pool.h"

#include <memory>
#include <optional>
#include <string>

namespace tinyinfer{

enum class SessionLifecycle{
    Created,
    Loaded,
    Initialized
};

class InferenceSession{
public:
    explicit InferenceSession(SessionOptions options = {});

    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;
    InferenceSession(InferenceSession&&) = delete;
    InferenceSession& operator=(InferenceSession&&) = delete;

    void load(
        Graph graph,
        std::string input_name,
        Shape input_shape,
        std::string output_name
    );

    void load(LoadedGraph loaded_graph);
    void initialize();

    Tensor run(const Tensor& input) const;
    Tensor run(const Tensor& input, RunProfile& profile) const;

    SessionLifecycle lifecycle() const noexcept;
    PassManager& pass_manager();

    const Graph& graph() const;
    const ExecutionPlan& plan() const;
    const SessionState& session_state() const;
    const ModelMetadata& metadata() const;

private:
    void require_initialized() const;
    Tensor run_impl(const Tensor& input, RunProfile* profile) const;

    SessionOptions options_;
    SessionLifecycle lifecycle_ = SessionLifecycle::Created;

    OperatorRegistry operator_registry_;
    KernelRegistry kernel_registry_;
    PassManager pass_manager_;

    std::optional<Graph> loaded_graph_;
    std::string input_name_;
    Shape input_shape_;
    std::string output_name_;
    std::optional<ModelMetadata> metadata_;

    std::unique_ptr<SessionState> session_state_;
    std::unique_ptr<ThreadPool> inter_op_thread_pool_;
};

}

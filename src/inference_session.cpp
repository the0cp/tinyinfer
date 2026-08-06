#include "inference_session.h"

#include "execution_frame.h"
#include "executor.h"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <utility>

namespace tinyinfer{

InferenceSession::InferenceSession(SessionOptions options)
    : options_(options){
    if(options_.execution_mode == ExecutionMode::Parallel && options_.inter_op_threads == 0){
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        options_.inter_op_threads = std::max<size_t>(1, hardware_threads);
    }
}

void InferenceSession::load(
    Graph graph,
    std::string input_name,
    Shape input_shape,
    std::string output_name
){
    if(lifecycle_ == SessionLifecycle::Initialized){
        throw std::logic_error("Cannot load a new Graph into an initialized InferenceSession.");
    }

    loaded_graph_ = std::move(graph);
    input_name_ = std::move(input_name);
    input_shape_ = std::move(input_shape);
    output_name_ = std::move(output_name);
    metadata_.reset();
    session_state_.reset();
    inter_op_thread_pool_.reset();
    lifecycle_ = SessionLifecycle::Loaded;
}

void InferenceSession::load(LoadedGraph loaded_graph){
    ModelMetadata metadata = loaded_graph.metadata;

    load(
        std::move(loaded_graph.graph),
        metadata.input_name,
        metadata.input_shape,
        metadata.output_name
    );

    metadata_ = std::move(metadata);
}

void InferenceSession::initialize(){
    if(lifecycle_ != SessionLifecycle::Loaded || !loaded_graph_){
        throw std::logic_error("InferenceSession::initialize requires a loaded Graph.");
    }

    Graph candidate = *loaded_graph_;

    if(options_.enable_graph_optimization && pass_manager_.size() != 0){
        pass_manager_.run(
            candidate,
            GraphOptimizationContext{input_name_, input_shape_, output_name_},
            operator_registry_
        );
    }

    candidate.resolve(input_name_, input_shape_, output_name_, operator_registry_);
    auto new_state = std::make_unique<SessionState>(
        SessionState::build(std::move(candidate), operator_registry_, kernel_registry_)
    );

    std::unique_ptr<ThreadPool> new_pool;

    if(options_.execution_mode == ExecutionMode::Parallel){
        new_pool = std::make_unique<ThreadPool>(options_.inter_op_threads);
    }

    session_state_ = std::move(new_state);
    inter_op_thread_pool_ = std::move(new_pool);
    lifecycle_ = SessionLifecycle::Initialized;
}

Tensor InferenceSession::run(const Tensor& input) const{
    require_initialized();

    const SessionState& state = *session_state_;
    const ExecutionPlan& execution_plan = state.execution_plan();
    ExecutionFrame frame(state);
    frame.bind_input(execution_plan.input_index(), input);

    if(options_.execution_mode == ExecutionMode::Sequential){
        SequentialExecutor executor;
        executor.execute(state, frame);
    }else{
        if(!inter_op_thread_pool_){
            throw std::logic_error("Parallel InferenceSession has no inter-op ThreadPool.");
        }

        ParallelExecutor executor(*inter_op_thread_pool_);
        executor.execute(state, frame);
    }

    return frame.fetch(execution_plan.output_index());
}

SessionLifecycle InferenceSession::lifecycle() const noexcept{
    return lifecycle_;
}

PassManager& InferenceSession::pass_manager(){
    if(lifecycle_ == SessionLifecycle::Initialized){
        throw std::logic_error("Graph passes must be registered before InferenceSession initialization.");
    }

    return pass_manager_;
}

const Graph& InferenceSession::graph() const{
    if(session_state_){
        return session_state_->graph();
    }

    if(loaded_graph_){
        return *loaded_graph_;
    }

    throw std::logic_error("InferenceSession has no loaded Graph.");
}

const ExecutionPlan& InferenceSession::plan() const{
    return session_state().execution_plan();
}

const SessionState& InferenceSession::session_state() const{
    require_initialized();
    return *session_state_;
}

const ModelMetadata& InferenceSession::metadata() const{
    if(!metadata_){
        throw std::logic_error("InferenceSession was not loaded from a model package.");
    }

    return *metadata_;
}

void InferenceSession::require_initialized() const{
    if(lifecycle_ != SessionLifecycle::Initialized || !session_state_){
        throw std::logic_error("InferenceSession is not initialized.");
    }
}

}

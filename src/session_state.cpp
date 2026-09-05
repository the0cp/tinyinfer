#include "session_state.h"

#include "kernel_registry.h"
#include "op_kernel.h"
#include "operator_registry.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tinyinfer{

namespace{

ValueRole classify_value(const Graph& graph, const std::string& name){
    if(name == graph.input_name()){
        return ValueRole::Input;
    }

    if(graph.has_constant(name)){
        return ValueRole::Initializer;
    }

    return ValueRole::Intermediate;
}

}

SessionState::SessionState(SessionState&&) noexcept = default;
SessionState& SessionState::operator=(SessionState&&) noexcept = default;
SessionState::~SessionState() = default;

SessionState SessionState::build(
    Graph graph,
    const OperatorRegistry& operator_registry,
    const KernelRegistry& kernel_registry,
    MemoryPlanningPolicy memory_policy
){
    if(!graph.is_resolved()){
        throw std::logic_error("Cannot build SessionState from an unresolved Graph.");
    }

    SessionState state;
    state.graph_ = std::move(graph);
    ExecutionPlan& plan = state.execution_plan_;

    plan.input_index_ = plan.value_names_.add(state.graph_.input_name());

    std::vector<std::string> initializer_names;
    initializer_names.reserve(state.graph_.constants().size());

    for(const auto& [name, tensor] : state.graph_.constants()){
        (void)tensor;
        initializer_names.push_back(name);
    }

    std::sort(initializer_names.begin(), initializer_names.end());

    for(const std::string& name : initializer_names){
        plan.value_names_.add(name);
    }

    for(NodeIndex node_index : state.graph_.topological_order()){
        const Node& node = state.graph_.nodes().at(node_index);
        plan.value_names_.add(node.output);
    }

    plan.output_index_ = plan.value_names_.get(state.graph_.output_name());
    plan.values_.reserve(plan.value_names_.size());

    for(size_t i = 0; i < plan.value_names_.size(); i++){
        const ValueIndex index = static_cast<ValueIndex>(i);
        const std::string name(plan.value_names_.name(index));
        const Shape& shape = state.graph_.shape(name);
        const ValueRole role = classify_value(state.graph_, name);
        const DataType dtype = role == ValueRole::Initializer
            ? state.graph_.constant(name).dtype()
            : DataType::Float32;
        const size_t numel = tensor_numel(shape);

        plan.values_.push_back(ValueInfo{
            name,
            shape,
            dtype,
            role,
            name == state.graph_.output_name(),
            numel,
            tensor_bytes(shape, dtype),
            lifetime_npos,
            lifetime_npos,
            lifetime_npos,
            0
        });
    }

    plan.nodes_.reserve(state.graph_.topological_order().size());
    state.kernels_.reserve(state.graph_.topological_order().size());

    std::vector<size_t> producer_position(plan.value_count(), lifetime_npos);

    for(size_t position = 0; position < state.graph_.topological_order().size(); position++){
        const NodeIndex node_index = state.graph_.topological_order()[position];
        const Node& node = state.graph_.nodes().at(node_index);
        const OperatorSchema& schema = operator_registry.get(node.op);

        if(node.inputs.size() != schema.input_count){
            throw std::logic_error(
                "Resolved Graph invariant broken at node '" + node.name + "'."
            );
        }

        NodeExecutionPlan node_plan;
        node_plan.source_node_index = node_index;
        node_plan.name = node.name;
        node_plan.op = node.op;
        node_plan.kernel_index = state.kernels_.size();
        node_plan.output = plan.value_index(node.output);
        node_plan.inputs.reserve(node.inputs.size());

        std::unordered_set<size_t> unique_dependencies;

        for(const std::string& input_name : node.inputs){
            const ValueIndex input = plan.value_index(input_name);
            node_plan.inputs.push_back(input);

            const size_t producer = producer_position.at(input);
            if(producer != lifetime_npos){
                unique_dependencies.insert(producer);
            }
        }

        node_plan.dependency_count = unique_dependencies.size();
        state.kernels_.push_back(kernel_registry.create_kernel(node));
        plan.nodes_.push_back(std::move(node_plan));

        if(producer_position.at(plan.nodes_.back().output) != lifetime_npos){
            throw std::logic_error("ExecutionPlan found more than one producer for a value.");
        }

        producer_position.at(plan.nodes_.back().output) = position;
    }

    for(size_t consumer_position = 0; consumer_position < plan.nodes_.size(); consumer_position++){
        std::unordered_set<size_t> unique_dependencies;

        for(ValueIndex input : plan.nodes_[consumer_position].inputs){
            const size_t producer = producer_position.at(input);

            if(producer != lifetime_npos && unique_dependencies.insert(producer).second){
                if(producer >= consumer_position){
                    throw std::logic_error("ExecutionPlan contains a non-topological dependency.");
                }
                plan.nodes_[producer].consumers.push_back(consumer_position);
            }
        }
    }

    for(size_t position = 0; position < plan.nodes_.size(); position++){
        NodeExecutionPlan& node = plan.nodes_[position];
        ValueInfo& output = plan.values_.at(node.output);
        output.produced_at = position;

        for(ValueIndex input_index : node.inputs){
            ValueInfo& input = plan.values_.at(input_index);

            if(input.first_use == lifetime_npos){
                input.first_use = position;
            }

            input.last_use = position;
            input.consumer_count++;
        }
    }

    for(ValueIndex index = 0; static_cast<size_t>(index) < plan.values_.size(); index++){
        const ValueInfo& value = plan.values_[index];

        if(value.role == ValueRole::Intermediate && !value.is_graph_output &&
           value.last_use != lifetime_npos){
            plan.nodes_.at(value.last_use).release_after_execute.push_back(index);
        }
    }

    state.memory_plan_ = MemoryPlanner::build(plan, memory_policy);

    return state;
}

const Graph& SessionState::graph() const noexcept{
    return graph_;
}

const ExecutionPlan& SessionState::execution_plan() const noexcept{
    return execution_plan_;
}

const MemoryPlan& SessionState::memory_plan() const noexcept{
    return memory_plan_;
}

std::string SessionState::dump_memory_plan() const{
    return memory_plan_.dump(execution_plan_);
}

const OpKernel& SessionState::kernel(size_t index) const{
    if(index >= kernels_.size()){
        throw std::out_of_range("Kernel index is out of range.");
    }

    return *kernels_[index];
}

const Tensor* SessionState::initializer(ValueIndex index) const{
    const ValueInfo& info = execution_plan_.value_info(index);

    if(info.role != ValueRole::Initializer){
        return nullptr;
    }

    return &graph_.constant(info.name);
}

size_t SessionState::value_count() const noexcept{
    return execution_plan_.value_count();
}

}

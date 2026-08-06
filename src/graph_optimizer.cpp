#include "graph_optimizer.h"

#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tinyinfer{

bool GraphOptimizationReport::changed() const noexcept{
    for(const GraphPassReport& pass : passes_){
        if(pass.changed){
            return true;
        }
    }

    return false;
}

const std::vector<GraphPassReport>& GraphOptimizationReport::passes() const noexcept{
    return passes_;
}

std::string GraphOptimizationReport::dump() const{
    std::ostringstream oss;
    oss << "Optimization report:\n";

    if(passes_.empty()){
        oss << "  <no passes>\n";
        return oss.str();
    }

    for(const GraphPassReport& pass : passes_){
        oss << "  " << pass.name
            << ": changed=" << (pass.changed ? "yes" : "no")
            << ", nodes=" << pass.nodes_before << " -> " << pass.nodes_after << "\n";
    }

    return oss.str();
}

GraphRewriteContext::GraphRewriteContext(
    std::vector<Node>& nodes,
    const std::unordered_map<std::string, Tensor>& constants
) : nodes_(nodes), constants_(constants){}

const std::vector<Node>& GraphRewriteContext::nodes() const noexcept{
    return nodes_;
}

const std::unordered_map<std::string, Tensor>& GraphRewriteContext::constants() const noexcept{
    return constants_;
}

void GraphRewriteContext::replace_nodes(std::vector<Node> nodes){
    if(nodes_ == nodes){
        return;
    }

    nodes_ = std::move(nodes);
    changed_ = true;
}

bool GraphRewriteContext::changed() const noexcept{
    return changed_;
}

std::string_view DeadNodeEliminationPass::name() const noexcept{
    return "DeadNodeElimination";
}

void DeadNodeEliminationPass::run(
    GraphRewriteContext& graph,
    const GraphOptimizationContext& context
) const{
    const std::vector<Node>& nodes = graph.nodes();
    std::unordered_map<std::string, size_t> producer;
    producer.reserve(nodes.size());

    for(size_t i = 0; i < nodes.size(); i++){
        producer.emplace(nodes[i].output, i);
    }

    std::vector<std::string> worklist{context.output_name};
    std::unordered_set<size_t> live_nodes;

    while(!worklist.empty()){
        std::string value = std::move(worklist.back());
        worklist.pop_back();

        auto it = producer.find(value);

        if(it == producer.end() || !live_nodes.insert(it->second).second){
            continue;
        }

        for(const std::string& input : nodes[it->second].inputs){
            worklist.push_back(input);
        }
    }

    if(live_nodes.size() == nodes.size()){
        return;
    }

    std::vector<Node> kept;
    kept.reserve(live_nodes.size());

    for(size_t i = 0; i < nodes.size(); i++){
        if(live_nodes.contains(i)){
            kept.push_back(nodes[i]);
        }
    }

    graph.replace_nodes(std::move(kept));
}

void PassManager::add_pass(std::unique_ptr<GraphPass> pass){
    if(!pass){
        throw std::invalid_argument("Cannot add a null graph optimization pass.");
    }

    passes_.push_back(std::move(pass));
}

size_t PassManager::size() const noexcept{
    return passes_.size();
}

GraphOptimizationReport PassManager::run(
    Graph& graph,
    const GraphOptimizationContext& context,
    const OperatorRegistry& registry
) const{
    Graph working = graph;
    GraphOptimizationReport report;
    bool pipeline_changed = false;

    for(const std::unique_ptr<GraphPass>& pass : passes_){
        Graph candidate = working;
        const size_t nodes_before = candidate.nodes_.size();
        GraphRewriteContext rewrite(candidate.nodes_, candidate.constants_);
        pass->run(rewrite, context);

        if(rewrite.changed()){
            candidate.invalidate_resolve_state();
            candidate.resolve(
                context.input_name,
                context.input_shape,
                context.output_name,
                registry
            );
            working = std::move(candidate);
            pipeline_changed = true;
        }

        report.passes_.push_back(GraphPassReport{
            std::string(pass->name()),
            rewrite.changed(),
            nodes_before,
            working.nodes_.size()
        });
    }

    if(pipeline_changed){
        graph = std::move(working);
    }

    return report;
}

}

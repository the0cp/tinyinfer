#include "graph_optimizer.h"

#include <sstream>
#include <stdexcept>

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
            << ", nodes=" << pass.nodes_before
            << " -> " << pass.nodes_after << "\n";
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
    const GraphOptimizationContext& context
) const{
    GraphOptimizationReport report;
    std::vector<Node> working_nodes = graph.nodes_;
    bool pipeline_changed = false;

    for(const std::unique_ptr<GraphPass>& pass : passes_){
        const size_t nodes_before = working_nodes.size();
        std::vector<Node> candidate_nodes = working_nodes;

        GraphRewriteContext rewrite(candidate_nodes, graph.constants_);
        pass->run(rewrite, context);

        const bool pass_changed = rewrite.changed();

        if(pass_changed){
            std::swap(graph.nodes_, candidate_nodes);

            try{
                (void)graph.compile(
                    context.input_name,
                    context.input_shape,
                    context.output_name
                );
            }catch(...){
                std::swap(graph.nodes_, candidate_nodes);
                throw;
            }

            std::swap(graph.nodes_, candidate_nodes);
            working_nodes = std::move(candidate_nodes);
            pipeline_changed = true;
        }

        report.passes_.push_back(GraphPassReport{
            std::string(pass->name()),
            pass_changed,
            nodes_before,
            working_nodes.size()
        });
    }

    if(pipeline_changed){
        graph.nodes_ = std::move(working_nodes);
        graph.revision_++;
    }

    return report;
}

}

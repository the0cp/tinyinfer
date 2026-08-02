#pragma once

#include "graph.h"

#include <concepts>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tinyinfer{

struct GraphOptimizationContext{
    std::string input_name;
    Shape input_shape;
    std::string output_name;
};

struct GraphPassReport{
    std::string name;
    bool changed = false;
    size_t nodes_before = 0;
    size_t nodes_after = 0;
};

class GraphOptimizationReport{
public:
    bool changed() const noexcept;
    const std::vector<GraphPassReport>& passes() const noexcept;
    std::string dump() const;

private:
    friend class PassManager;

    std::vector<GraphPassReport> passes_;
};

class GraphRewriteContext{
public:
    const std::vector<Node>& nodes() const noexcept;
    const std::unordered_map<std::string, Tensor>& constants() const noexcept;

    void replace_nodes(std::vector<Node> nodes);
    bool changed() const noexcept;

private:
    friend class PassManager;

    GraphRewriteContext(
        std::vector<Node>& nodes,
        const std::unordered_map<std::string, Tensor>& constants
    );

    std::vector<Node>& nodes_;
    const std::unordered_map<std::string, Tensor>& constants_;
    bool changed_ = false;
};

class GraphPass{
public:
    virtual ~GraphPass() = default;

    virtual std::string_view name() const noexcept = 0;

    virtual void run(
        GraphRewriteContext& graph,
        const GraphOptimizationContext& context
    ) const = 0;
};

class PassManager{
public:
    void add_pass(std::unique_ptr<GraphPass> pass);

    template<std::derived_from<GraphPass> Pass, typename... Args>
    Pass& emplace_pass(Args&&... args){
        auto pass = std::make_unique<Pass>(std::forward<Args>(args)...);
        Pass& reference = *pass;
        add_pass(std::move(pass));
        return reference;
    }

    size_t size() const noexcept;

    GraphOptimizationReport run(
        Graph& graph,
        const GraphOptimizationContext& context
    ) const;

private:
    std::vector<std::unique_ptr<GraphPass>> passes_;
};

}

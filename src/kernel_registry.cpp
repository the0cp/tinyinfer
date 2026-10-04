#include "kernel_registry.h"

#include "builtin_kernels.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinyinfer{

KernelRegistry::KernelRegistry(bool register_builtins){
    if(register_builtins){
        register_builtin_kernels(*this);
    }
}

void KernelRegistry::register_kernel(OpType type, KernelCandidate candidate){
    if(candidate.name.empty()){
        throw std::invalid_argument("Cannot register a kernel with an empty name.");
    }
    if(!candidate.match){
        throw std::invalid_argument(
            "Kernel '" + candidate.name + "' has no matcher."
        );
    }
    if(!candidate.factory){
        throw std::invalid_argument(
            "Kernel '" + candidate.name + "' has an empty factory."
        );
    }

    std::vector<KernelCandidate>& candidates = candidates_[type];
    const auto duplicate = std::find_if(
        candidates.begin(),
        candidates.end(),
        [&](const KernelCandidate& registered){
            return registered.name == candidate.name;
        }
    );
    if(duplicate != candidates.end()){
        throw std::runtime_error(
            "Kernel name is already registered for this operator: " +
            candidate.name
        );
    }
    candidates.push_back(std::move(candidate));
}

SelectedKernel KernelRegistry::select_kernel(
    const Node& node,
    const KernelSelectionContext& context
) const{
    const auto candidates_it = candidates_.find(node.op);
    if(candidates_it == candidates_.end() || candidates_it->second.empty()){
        throw std::runtime_error("No CPU kernel is registered for node '" + node.name + "'.");
    }

    struct RankedCandidate{
        const KernelCandidate* candidate;
        uint64_t cost;
        size_t registration_order;
    };

    const std::vector<KernelCandidate>& candidates = candidates_it->second;
    std::vector<RankedCandidate> compatible;
    compatible.reserve(candidates.size());
    std::vector<std::string> rejections;
    rejections.reserve(candidates.size());

    for(size_t order = 0; order < candidates.size(); order++){
        const KernelCandidate& candidate = candidates[order];
        if(candidate.threading == KernelThreading::IntraOp &&
           context.effective_intra_op_threads < 2){
            rejections.push_back(
                candidate.name + ": requires at least two effective intra-op threads"
            );
            continue;
        }

        const KernelMatch match = candidate.match(context);
        if(!match.supported){
            rejections.push_back(
                candidate.name + ": " +
                (match.rejection.empty() ? "unsupported" : match.rejection)
            );
            continue;
        }
        compatible.push_back(RankedCandidate{&candidate, match.estimated_cost, order});
    }

    if(compatible.empty()){
        std::ostringstream message;
        message << "No compatible CPU kernel for node '" << node.name << "'.";
        for(const std::string& rejection : rejections){
            message << "\n  - " << rejection;
        }
        throw std::runtime_error(message.str());
    }

    std::sort(
        compatible.begin(),
        compatible.end(),
        [](const RankedCandidate& left, const RankedCandidate& right){
            if(left.cost != right.cost){
                return left.cost < right.cost;
            }
            if(left.candidate->priority != right.candidate->priority){
                return left.candidate->priority > right.candidate->priority;
            }
            return left.registration_order < right.registration_order;
        }
    );

    const RankedCandidate& best = compatible.front();
    std::string reason = "only compatible candidate";
    if(compatible.size() > 1){
        const RankedCandidate& runner_up = compatible[1];
        if(best.cost != runner_up.cost){
            reason = "lowest estimated cost: " + std::to_string(best.cost) + " < " +
                std::to_string(runner_up.cost);
        }else if(best.candidate->priority != runner_up.candidate->priority){
            reason = "equal cost; higher priority: " +
                std::to_string(best.candidate->priority) + " > " +
                std::to_string(runner_up.candidate->priority);
        }else{
            reason = "equal cost and priority; earlier registration wins";
        }
    }

    std::unique_ptr<OpKernel> kernel = best.candidate->factory(node, context);
    if(!kernel){
        throw std::runtime_error(
            "Kernel factory returned null for candidate '" + best.candidate->name + "'."
        );
    }

    return SelectedKernel{
        std::move(kernel),
        KernelSelectionRecord{
            best.candidate->name,
            best.candidate->threading,
            best.cost,
            std::move(reason)
        }
    };
}

}

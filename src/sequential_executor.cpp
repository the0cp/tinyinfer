#include "executor.h"

#include "op_kernel.h"

#include <exception>
#include <stdexcept>

namespace tinyinfer{

void SequentialExecutor::execute(
    const SessionState& session_state,
    ExecutionFrame& frame
) const{
    const ExecutionPlan& plan = session_state.execution_plan();

    for(const NodeExecutionPlan& node : plan.nodes()){
        try{
            OpKernelContext context(node, frame);
            session_state.kernel(node.kernel_index).compute(context);

            for(ValueIndex value : node.release_after_execute){
                frame.release(value);
            }
        }catch(const std::exception& error){
            throw std::runtime_error(
                "Sequential execution failed at node '" + node.name + "': " + error.what()
            );
        }
    }
}

}

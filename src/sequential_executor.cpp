#include "executor.h"

#include "op_kernel.h"
#include "profiler.h"

#include <exception>
#include <stdexcept>

namespace tinyinfer{

SequentialExecutor::SequentialExecutor(ThreadPool* intra_op_thread_pool)
    : intra_op_thread_pool_(intra_op_thread_pool){}

void SequentialExecutor::execute(
    const SessionState& session_state,
    ExecutionFrame& frame,
    RunProfiler* profiler
) const{
    const ExecutionPlan& plan = session_state.execution_plan();

    for(size_t position = 0; position < plan.node_count(); position++){
        const NodeExecutionPlan& node = plan.nodes()[position];

        if(profiler){
            profiler->node_queued(position);
            profiler->node_started(position);
        }

        try{
            ThreadPool* kernel_pool = nullptr;
            if(node.kernel_selection.threading == KernelThreading::IntraOp){
                if(!intra_op_thread_pool_){
                    throw std::logic_error(
                        "Compiled intra-op kernel has no runtime ThreadPool."
                    );
                }
                kernel_pool = intra_op_thread_pool_;
            }

            OpKernelContext context(node, frame, kernel_pool);
            session_state.kernel(node.kernel_index).compute(context);
            context.validate_output();

            for(ValueIndex value : node.release_after_execute){
                frame.release(value);
            }

            if(profiler){
                profiler->node_completed(position);
            }
        }catch(const std::exception& error){
            if(profiler){
                profiler->node_failed(position, error.what());
            }

            throw std::runtime_error(
                "Sequential execution failed at node '" + node.name + "': " + error.what()
            );
        }catch(...){
            if(profiler){
                profiler->node_failed(position, "non-standard exception");
            }

            throw;
        }
    }
}

}

target_sources(tinyinfer PRIVATE
    # Shared storage and concurrency
    core/buffer.cpp
    core/tensor.cpp
    core/thread_pool.cpp

    # Reusable CPU computation
    compute/ops.cpp

    # Static graph representation and optimization
    graph/value_index.cpp
    graph/operator_registry.cpp
    graph/graph.cpp
    graph/graph_optimizer.cpp

    # Graph compilation and execution
    runtime/threading_policy.cpp
    runtime/execution_plan.cpp
    runtime/memory_plan.cpp
    runtime/arena.cpp
    runtime/session_state.cpp
    runtime/profiler.cpp
    runtime/execution_frame.cpp
    runtime/op_kernel.cpp
    runtime/kernel_registry.cpp
    runtime/builtin_kernels.cpp
    runtime/sequential_executor.cpp
    runtime/parallel_executor.cpp
    runtime/inference_session.cpp
    runtime/model_loader.cpp

    # Stateful autoregressive inference
    llm/kv_cache.cpp
    llm/paged_kv_cache.cpp
    llm/attention.cpp
    llm/decoder_block.cpp
    llm/qwen2.cpp
)

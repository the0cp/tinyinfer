# tinyinfer

**A lightweight CPU inference runtime implemented in C++20.**

tinyinfer implements the core components of neural-network inference, from tensor storage and operator kernels to graph compilation, memory planning, parallel execution, and KV-cache management. The project focuses on explicit implementation choices, correctness tests, and reproducible performance experiments.

It currently supports a small set of float32 operators and CPU execution. The Attention and KV-cache components are reference implementations rather than a complete LLM serving system.

## Features

| Component                     | Implementation                                               |
| ----------------------------- | ------------------------------------------------------------ |
| **Tensor & operators**        | Shared buffers, strided views, zero-copy reshape, Linear, ReLU, Add, Softmax, and several MatMul kernels. |
| **Graph runtime**             | Dependency resolution, topological ordering, shape validation, dead-node elimination, and a compiled `ExecutionPlan`. |
| **Execution**                 | Sequential and dependency-driven parallel executors; configurable inter-op and intra-op threading. |
| **Memory planning**           | Tensor lifetime analysis and reusable activation buffers in sequential mode; dedicated buffers for parallel execution and a private arena per run. |
| **Kernel selection**          | Session-initialization-time selection of serial or threaded CPU kernels using shape-aware cost estimates. |
| **Model loading & profiling** | Native manifest/binary-weight format, per-node timing, allocation tracking, and Chrome trace export. |
| **Attention & KV cache**      | CPU causal Attention, prefill/decode, contiguous and paged KV storage, and reference-counted full-block prefix sharing. |

The execution pipeline is:

```text
Graph → validation / optimization → ExecutionPlan + MemoryPlan
      → ExecutionFrame → selected kernels → output Tensor
```

Paged KV cache is a separate reference component: request-local block tables map logical token positions to fixed-size physical blocks in a shared pool. `share_prefix_from()` reuses complete prefix blocks without copying K/V; per-block reference counts control reclamation. Prefix compatibility must be established by the caller.

## Build and test


```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure

./build-release/runtime_demo
./build-release/bench_paged_kv_cache
```

AddressSanitizer and UndefinedBehaviorSanitizer can be enabled with GCC/Clang:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DTINYINFER_ENABLE_ASAN_UBSAN=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

## Tests and benchmarks


| Benchmark                               |                        Baseline |                      Comparison | Result                   |
| --------------------------------------- | ------------------------------: | ------------------------------: | ------------------------ |
| MatMul, 512 × 512                       |     Serial `fast_matmul`: 13 ms |           8-thread MatMul: 6 ms | **2.2× faster**          |
| Memory planning, 12 ReLU nodes          |                    9.921 µs/run |                    6.092 µs/run | **38.6% lower latency**  |
| Managed buffer allocations, same graph  |                              12 |                               3 | **75% fewer**            |
| Cached Attention, 96 tokens             | Full recomputation: 39,049.7 µs | Contiguous KV cache: 1,289.2 µs | **30.0× faster**         |
| Kernel selection, `[8,512] × [512,512]` |              Serial: 200.117 µs |            Threaded: 131.221 µs | **1.53× faster**         |
| Full-block prefix sharing               |           20 occupied KV blocks |            8 occupied KV blocks | **60% fewer blocks**     |
| Small branching graph, 1,000 runs       |             Sequential: 12.1 ms |              Parallel: 170.3 ms | **Parallel ~14× slower** |

Performance depends on workload size. Parallel scheduling is counterproductive for tiny graph nodes, while kernel-level parallelism benefits sufficiently large matrix operations. With four tokens per block, the paged-cache benchmark achieves **94.7% mapped-slot utilization** on its five-request workload.

The **60%** prefix-sharing result measures unique occupied KV blocks, not process RSS; the block pool preallocates backing memory. The cached-Attention speedup compares incremental decoding against repeated full recomputation, not paged against contiguous KV storage. See [benchmark results](benchmark/RESULTS.md) for configurations, raw measurement definitions, and limitations.

To run every benchmark:

```bash
for name in matmul graph_scheduler session profiler tensor_storage memory_plan kernel_selection kv_cache paged_kv_cache; do
  ./build-release/bench_"$name"
done
```
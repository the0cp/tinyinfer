# Benchmark results

## 1. Operator kernels — 512 × 512 matrix multiplication

`benchmark/bench_matmul.cpp`; each variant executes once per invocation and reports whole milliseconds (coarse resolution).

| Kernel                      | Median of 7 runs |    Range |
| --------------------------- | ---------------: | -------: |
| `fast_matmul`               |            13 ms | 13–14 ms |
| `blocked_matmul(..., 32)`   |            63 ms | 62–68 ms |
| `parallel_matmul(..., 8)`   |             6 ms |   5–7 ms |
| `threadpool_matmul(..., 8)` |             6 ms |   4–9 ms |


## 2. Graph execution — small independent branches

`benchmark/bench_graph_scheduler.cpp`; eight ReLU branches accumulated with Add, run 1,000 times; parallel mode requests four inter-op workers.

| Mode       | Median elapsed for 1,000 inference calls |              Range |
| ---------- | ---------------------------------------: | -----------------: |
| Sequential |                                12.123 ms |   10.458–30.924 ms |
| Parallel   |                               170.305 ms | 161.308–198.893 ms |


## 3. Memory planning — 12-node ReLU chain

`benchmark/bench_memory_plan.cpp`; activation shape `[1, 4096]`; 21 batches of 50 inference calls per mode after warmup.

| Metric                                        |         Without plan | With sequential-reuse plan |
| --------------------------------------------- | -------------------: | -------------------------: |
| Median of batch-mean latency (7-run median)   |             9.921 μs |                   6.092 μs |
| P95 of batch-mean latency (7-run median)      |            13.186 μs |                   7.544 μs |
| Managed buffers allocated during a sample run |                   12 |                          3 |
| Managed allocation / arena payload size       | 196,608 B cumulative |             49,152 B arena |
| Planned value reuses                          |                    — |                          9 |


## 4. Compiled kernel selection

`benchmark/bench_kernel_selection.cpp`; cost-based selection is made once when building the session and varies with input shape and requested intra-op threads.

| Shape                  | 1-thread selection / median | 4-thread selection / median |
| ---------------------- | --------------------------- | --------------------------- |
| `[1,32] × [32,32]`     | serial / 0.544 μs           | serial / 0.537 μs           |
| `[4,32] × [32,32]`     | serial / 0.902 μs           | serial / 0.898 μs           |
| `[64,128] × [128,128]` | serial / 100.598 μs         | threaded / 78.909 μs        |
| `[256,64] × [64,64]`   | serial / 104.728 μs         | threaded / 80.411 μs        |
| `[8,512] × [512,512]`  | serial / 200.117 μs         | threaded / 131.221 μs       |


## 5. Cache-backed causal Attention

`benchmark/bench_kv_cache.cpp`; eight Query heads, head dimension 16, eight prompt tokens; compares full causal recomputation at each step with prefill once + contiguous KV cache for decode. The benchmark varies KV heads across MHA, GQA, and MQA. `cache_bytes` is calculated from Tensor shape, not measured RSS. `checksum_delta` is an auxiliary aggregate check; unit tests provide the element-wise comparisons.

| Tokens | Mode | Full recomputation | KV cached | Speedup | Cache bytes | Saving vs MHA |
|---:|---|---:|---:|---:|---:|---:|
| 32 | 8Q/8KV | 2,865.3 μs | 276.6 μs | 10.4× | 32,768 | 0% |
| 32 | 8Q/2KV | 2,727.3 μs | 322.2 μs | 8.5× | 8,192 | 75% |
| 32 | 8Q/1KV | 2,902.6 μs | 288.7 μs | 10.1× | 4,096 | 87.5% |
| 64 | 8Q/8KV | 22,433.8 μs | 1,126.9 μs | 19.9× | 65,536 | 0% |
| 64 | 8Q/2KV | 23,104.1 μs | 1,101.5 μs | 21.0× | 16,384 | 75% |
| 64 | 8Q/1KV | 22,239.8 μs | 1,088.4 μs | 20.4× | 8,192 | 87.5% |
| 96 | 8Q/8KV | 76,570.4 μs | 2,572.3 μs | 29.8× | 98,304 | 0% |
| 96 | 8Q/2KV | 73,550.0 μs | 2,472.1 μs | 29.8× | 24,576 | 75% |
| 96 | 8Q/1KV | 74,110.8 μs | 2,311.2 μs | 32.1× | 12,288 | 87.5% |

This is a three-sample median without a dedicated warm-up. It is useful as a local component snapshot, not as the final real-model performance methodology.


## 6. Paged KV utilization and shared prefixes

`benchmark/bench_paged_kv_cache.cpp`; metrics are computed from actual Cache/Pool state and are deterministic for the specified setup. Five requests have lengths 3, 7, 12, 19, 31, maximum 64 tokens each, four heads and head dimension 16.

| Tokens/block | Logical KV bytes | Mapped KV bytes | Utilization | Reduction vs per-request max reservation |
| -----------: | ---------------: | --------------: | ----------: | ---------------------------------------: |
|            1 |           36,864 |          36,864 |      100.0% |                                    77.5% |
|            4 |           36,864 |          38,912 |       94.7% |                                    76.2% |
|            8 |           36,864 |          45,056 |       81.8% |                                    72.5% |
|           16 |           36,864 |          57,344 |       64.3% |                                    65.0% |


## 7. Tensor views, session overhead and profiling

| Program                | Measured result (median of 7 invocations)                    | Interpretation                                               |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `bench_tensor_storage` | Owning tensor construction 359.29 ns vs metadata-only reshape 48.05 ns | Metadata-only views avoid allocating/copying; **different operations**, not equivalent benchmarks |
| `bench_session`        | 10.519 ms for 10,000 runs                                    | About 1.052 μs/run for a small graph in this environment     |
| `bench_profiler`       | Normal 0.941 μs vs profiled 1.583 μs                         | Approximately 68% higher median time for a tiny 3-node graph; instrumentation costs are visible |


## Reproduce

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure

for name in matmul graph_scheduler session profiler tensor_storage memory_plan kernel_selection kv_cache paged_kv_cache; do
  ./build-release/bench_"$name"
done
```

For memory-safety validation:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DTINYINFER_ENABLE_ASAN_UBSAN=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

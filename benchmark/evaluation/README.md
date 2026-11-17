# Evaluation

## Results

- Fashion-MNIST: **86.12%** accuracy and **86.15%** Macro-F1 on 10,000 test images.
- Fashion runtime check: maximum logit error **2.86e-06** on the exported sample.
- Qwen reference check: all greedy tokens match; maximum observed logit error **0.00104**.
- Qwen reference generation: 30 prompt tokens, 24 generated token ids including EOS, 23 cached Decode steps, stop=eos.
- Qwen reference output: `A key-value cache stores data in memory to quickly retrieve values based on keys, typically for quick access to data.`
- Qwen generation: median decode throughput **8.19 tokens/s** across 5 runs.
- Memory plan: median batch-mean latency **9.21** → **6.80 µs**.

## Method

- Correctness prompt: system=`You are a concise inference systems tutor.`; user=`Explain KV cache in one short sentence.`.
- Correctness generation stops at EOS, 64 is only the safety limit.
- Correctness compares Embedding, all 24 decoder layers, Final RMSNorm and final logits for Prefill and every cached Decode step.
- Performance prompt: system=`You are a helpful assistant.`; user=`Explain KV cache in one short sentence.`
- Fashion-MNIST quality is computed from the exported FP32 weights over the official 10,000-image test split. One exported sample is also checked through the C++ runtime.
- Qwen correctness compares prefill and cached-decode logits against the stored reference fixture and requires identical greedy tokens.
- Component benchmarks use their built-in warm-up and repeat policies. Memory-plan values are medians of five independent benchmark invocations. Paged-cache memory figures come from cache state rather than process RSS.

## Figures

- `figures/fashion_confusion_matrix.{png,svg}`
- `figures/fashion_class_f1.{png,svg}`
- `figures/qwen_logit_error.{png,svg}`
- `figures/qwen_layer_max_error.{png,svg}`
- `figures/qwen_layer_mean_error.{png,svg}`
- `figures/qwen_generation_latency.{png,svg}`
- `figures/kv_cache_speedup.{png,svg}`
- `figures/paged_kv_memory.{png,svg}`
- `figures/linear_threading_speedup.{png,svg}`
- `figures/memory_plan_latency.{png,svg}`

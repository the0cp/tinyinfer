#include "attention.h"
#include "kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace{

using namespace tinyinfer;

Tensor make_tensor(size_t tokens, size_t heads, size_t head_dim, float phase){
    Tensor tensor({tokens, heads, head_dim});
    for(size_t i = 0; i < tensor.numel(); i++){
        tensor.data()[i] = std::sin(static_cast<float>(i) * 0.013f + phase);
    }
    return tensor;
}

Tensor token_view(const Tensor& tensor, size_t first_token, size_t token_count){
    const size_t elements_per_token = tensor.shape()[1] * tensor.shape()[2];
    return Tensor(
        tensor.buffer(),
        tensor.byte_offset() + first_token * elements_per_token * sizeof(float),
        Shape{token_count, tensor.shape()[1], tensor.shape()[2]}
    );
}

template<typename Function>
double median_microseconds(Function&& function, size_t repeats, float& checksum){
    std::vector<double> samples;
    samples.reserve(repeats);
    checksum = 0.0f;

    (void)function();

    for(size_t repeat = 0; repeat < repeats; repeat++){
        const auto start = std::chrono::steady_clock::now();
        checksum += function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

float run_full_recomputation(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    size_t prompt_tokens
){
    float checksum = 0.0f;
    for(size_t token = prompt_tokens - 1; token < queries.shape()[0]; token++){
        const size_t prefix_tokens = token + 1;
        const Tensor output = causal_attention_reference(
            token_view(queries, 0, prefix_tokens),
            token_view(keys, 0, prefix_tokens),
            token_view(values, 0, prefix_tokens)
        );
        const size_t token_size = queries.shape()[1] * queries.shape()[2];
        checksum += output.data()[token * token_size];
    }
    return checksum;
}

float run_cached(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    size_t prompt_tokens
){
    ContiguousKvCache cache({queries.shape()[0], queries.shape()[1], queries.shape()[2]});
    Tensor output = attention_prefill(
        token_view(queries, 0, prompt_tokens),
        token_view(keys, 0, prompt_tokens),
        token_view(values, 0, prompt_tokens),
        cache
    );
    const size_t token_size = queries.shape()[1] * queries.shape()[2];
    float checksum = output.data()[(prompt_tokens - 1) * token_size];

    for(size_t token = prompt_tokens; token < queries.shape()[0]; token++){
        output = attention_decode(
            token_view(queries, token, 1),
            token_view(keys, token, 1),
            token_view(values, token, 1),
            cache
        );
        checksum += output.data()[0];
    }
    return checksum;
}

void run_case(size_t total_tokens){
    constexpr size_t heads = 4;
    constexpr size_t head_dim = 16;
    constexpr size_t prompt_tokens = 8;
    constexpr size_t repeats = 3;

    const Tensor queries = make_tensor(total_tokens, heads, head_dim, 0.1f);
    const Tensor keys = make_tensor(total_tokens, heads, head_dim, 0.2f);
    const Tensor values = make_tensor(total_tokens, heads, head_dim, 0.3f);

    float reference_checksum = 0.0f;
    const double reference_us = median_microseconds(
        [&](){ return run_full_recomputation(queries, keys, values, prompt_tokens); },
        repeats,
        reference_checksum
    );

    float cached_checksum = 0.0f;
    const double cached_us = median_microseconds(
        [&](){ return run_cached(queries, keys, values, prompt_tokens); },
        repeats,
        cached_checksum
    );

    const size_t cache_bytes = 2 * total_tokens * heads * head_dim * sizeof(float);
    std::cout << total_tokens << ',' << prompt_tokens << ','
              << std::fixed << std::setprecision(1)
              << reference_us << ',' << cached_us << ','
              << reference_us / cached_us << ',' << cache_bytes << ','
              << std::fabs(reference_checksum - cached_checksum) << '\n';
}

}

int main(){
    std::cout << "tokens,prompt_tokens,full_recompute_us,kv_cached_us,speedup,cache_bytes,checksum_delta\n";
    for(size_t tokens : {32U, 64U, 96U}){
        run_case(tokens);
    }
    return 0;
}

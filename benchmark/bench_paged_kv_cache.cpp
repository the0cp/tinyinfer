#include "paged_kv_cache.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
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

size_t divide_round_up(size_t value, size_t divisor){
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

void run_case(size_t block_tokens){
    constexpr size_t heads = 4;
    constexpr size_t head_dim = 16;
    constexpr size_t max_tokens_per_request = 64;
    const std::vector<size_t> lengths{3, 7, 12, 19, 31};

    const size_t required_blocks = std::accumulate(
        lengths.begin(),
        lengths.end(),
        size_t{0},
        [&](size_t total, size_t length){
            return total + divide_round_up(length, block_tokens);
        }
    );

    KvBlockPool pool({required_blocks, block_tokens, heads, head_dim});
    std::vector<std::unique_ptr<PagedKvCache>> caches;
    caches.reserve(lengths.size());

    size_t logical_bytes = 0;
    size_t paged_bytes = 0;
    for(size_t request = 0; request < lengths.size(); request++){
        const size_t tokens = lengths[request];
        auto cache = std::make_unique<PagedKvCache>(pool, max_tokens_per_request);
        const Tensor keys = make_tensor(tokens, heads, head_dim, 0.1f + request);
        const Tensor values = make_tensor(tokens, heads, head_dim, 0.2f + request);
        cache->append(keys, values);
        logical_bytes += cache->used_bytes();
        paged_bytes += cache->allocated_bytes();
        caches.push_back(std::move(cache));
    }

    const size_t contiguous_bytes =
        lengths.size() * 2 * max_tokens_per_request * heads * head_dim * sizeof(float);
    const double utilization = static_cast<double>(logical_bytes) / paged_bytes;
    const double saving = 1.0 - static_cast<double>(paged_bytes) / contiguous_bytes;

    std::cout << block_tokens << ',' << lengths.size() << ','
              << logical_bytes << ',' << paged_bytes << ',' << contiguous_bytes << ','
              << std::fixed << std::setprecision(3)
              << utilization << ',' << saving << '\n';
}

}

int main(){
    std::cout << "block_tokens,requests,logical_bytes,paged_bytes,contiguous_bytes,"
                 "paged_utilization,saving_vs_contiguous\n";
    for(size_t block_tokens : {1U, 4U, 8U, 16U}){
        run_case(block_tokens);
    }
    return 0;
}

#include "attention.h"
#include "paged_kv_cache.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace{

using namespace tinyinfer;

static_assert(!std::is_copy_constructible_v<KvBlockPool>);
static_assert(!std::is_move_constructible_v<KvBlockPool>);
static_assert(!std::is_copy_constructible_v<PagedKvCache>);
static_assert(!std::is_move_constructible_v<PagedKvCache>);

void expect(bool condition, const std::string& message){
    if(!condition){
        throw std::runtime_error("EXPECT failed: " + message);
    }
}

void expect_close(float actual, float expected, float tolerance = 1e-5f){
    if(std::fabs(actual - expected) > tolerance){
        throw std::runtime_error(
            "EXPECT_CLOSE failed: actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected)
        );
    }
}

template<typename Function>
void expect_throw(Function&& function, const std::string& needle){
    try{
        function();
    }catch(const std::exception& error){
        expect(
            std::string(error.what()).find(needle) != std::string::npos,
            "exception did not contain '" + needle + "': " + error.what()
        );
        return;
    }
    throw std::runtime_error("Expected exception containing: " + needle);
}

Tensor range_tensor(size_t tokens, size_t heads, size_t head_dim, float first){
    Tensor tensor({tokens, heads, head_dim});
    for(size_t i = 0; i < tensor.numel(); i++){
        tensor.data()[i] = first + static_cast<float>(i);
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

void expect_tensor_close(const Tensor& actual, const Tensor& expected){
    expect(actual.shape() == expected.shape(), "tensor shape mismatch");
    for(size_t i = 0; i < actual.numel(); i++){
        expect_close(actual.data()[i], expected.data()[i]);
    }
}

void test_pool_configuration_and_lifetime(){
    expect_throw([](){ KvBlockPool({0, 2, 1, 1}); }, "num_blocks");
    expect_throw([](){ KvBlockPool({1, 0, 1, 1}); }, "block_tokens");
    expect_throw([](){ KvBlockPool({1, 2, 0, 1}); }, "num_heads");
    expect_throw([](){ KvBlockPool({1, 2, 1, 0}); }, "head_dim");

    KvBlockPool pool({3, 2, 2, 2});
    expect(pool.num_blocks() == 3, "pool block count");
    expect(pool.free_blocks() == 3, "all blocks initially free");
    expect(pool.capacity_tokens() == 6, "pool token capacity");
    expect(pool.block_bytes() == 2 * 2 * 2 * 2 * sizeof(float), "block bytes");
    expect(pool.allocated_bytes() == 3 * pool.block_bytes(), "pool bytes");

    {
        PagedKvCache cache(pool, 4);
        Tensor token = range_tensor(1, 2, 2, 1.0f);
        cache.append(token, token);
        expect(pool.free_blocks() == 2, "cache acquired one block");
    }
    expect(pool.free_blocks() == 3, "destructor returned the block");

    expect_throw([&](){ PagedKvCache(pool, 0); }, "max_tokens");
    expect_throw([&](){ PagedKvCache(pool, 7); }, "pool capacity");
}

void test_append_across_block_boundaries(){
    KvBlockPool pool({3, 2, 2, 2});
    PagedKvCache cache(pool, 6);
    Tensor keys = range_tensor(5, 2, 2, 10.0f);
    Tensor values = range_tensor(5, 2, 2, 100.0f);

    cache.append(keys, values);
    expect(cache.size_tokens() == 5, "five tokens appended");
    expect(cache.used_blocks() == 3, "five tokens need three two-token blocks");
    expect(cache.block_table().size() == 3, "three logical block mappings");
    expect(cache.used_bytes() == 2 * 5 * 2 * 2 * sizeof(float), "logical bytes");
    expect(cache.allocated_bytes() == 3 * pool.block_bytes(), "reserved block bytes");

    for(size_t token = 0; token < 5; token++){
        for(size_t head = 0; head < 2; head++){
            for(size_t element = 0; element < 2; element++){
                const size_t source = token * 4 + head * 2 + element;
                expect_close(cache.key(token, head)[element], keys.data()[source]);
                expect_close(cache.value(token, head)[element], values.data()[source]);
            }
        }
    }
}

void test_fragmented_block_mapping(){
    KvBlockPool pool({4, 2, 1, 1});
    PagedKvCache first(pool, 4);
    PagedKvCache survivor(pool, 2);
    first.append(range_tensor(4, 1, 1, 1.0f), range_tensor(4, 1, 1, 11.0f));
    survivor.append(range_tensor(2, 1, 1, 21.0f), range_tensor(2, 1, 1, 31.0f));

    expect(first.block_table()[0] == 0 && first.block_table()[1] == 1, "initial blocks");
    expect(survivor.block_table()[0] == 2, "surviving middle allocation");
    first.clear();

    PagedKvCache reused(pool, 4);
    Tensor keys = range_tensor(4, 1, 1, 41.0f);
    Tensor values = range_tensor(4, 1, 1, 51.0f);
    reused.append(keys, values);

    expect(
        reused.block_table()[0] == 1 && reused.block_table()[1] == 0,
        "logical order is independent of physical block order"
    );
    for(size_t token = 0; token < 4; token++){
        expect_close(reused.key(token, 0)[0], keys.data()[token]);
        expect_close(reused.value(token, 0)[0], values.data()[token]);
    }
}

void test_exhaustion_and_capacity_are_transactional(){
    KvBlockPool pool({2, 2, 1, 2});
    PagedKvCache full(pool, 4);
    Tensor three = range_tensor(3, 1, 2, 1.0f);
    full.append(three, three);
    expect(pool.free_blocks() == 0, "first cache owns the pool");

    PagedKvCache waiting(pool, 2);
    Tensor one = range_tensor(1, 1, 2, 20.0f);
    expect_throw([&](){ waiting.append(one, one); }, "exhausted");
    expect(waiting.size_tokens() == 0, "exhaustion leaves token count unchanged");
    expect(waiting.used_blocks() == 0, "exhaustion acquires no partial blocks");

    full.clear();
    waiting.append(one, one);
    expect(waiting.size_tokens() == 1, "append succeeds after blocks return");

    Tensor two = range_tensor(2, 1, 2, 30.0f);
    expect_throw([&](){ waiting.append(two, two); }, "capacity");
    expect(waiting.size_tokens() == 1, "capacity failure leaves state unchanged");
    expect_close(waiting.key(0, 0)[0], one.data()[0]);
}

void test_paged_attention_matches_reference(){
    Tensor queries = range_tensor(4, 2, 3, 0.1f);
    Tensor keys = range_tensor(4, 2, 3, 1.1f);
    Tensor values = range_tensor(4, 2, 3, 2.1f);
    const Tensor reference = causal_attention_reference(queries, keys, values);

    KvBlockPool pool({2, 2, 2, 3});
    PagedKvCache cache(pool, 4);
    const Tensor prefill = attention_prefill(
        token_view(queries, 0, 3),
        token_view(keys, 0, 3),
        token_view(values, 0, 3),
        cache
    );
    expect_tensor_close(prefill, token_view(reference, 0, 3));

    const Tensor decoded = attention_decode(
        token_view(queries, 3, 1),
        token_view(keys, 3, 1),
        token_view(values, 3, 1),
        cache
    );
    expect_tensor_close(decoded, token_view(reference, 3, 1));
}

void test_attention_across_block_sizes(){
    Tensor queries = range_tensor(6, 2, 2, 0.25f);
    Tensor keys = range_tensor(6, 2, 2, 1.25f);
    Tensor values = range_tensor(6, 2, 2, 2.25f);
    const Tensor reference = causal_attention_reference(queries, keys, values);

    for(size_t block_tokens : {1U, 2U, 3U, 4U}){
        const size_t num_blocks =
            6 / block_tokens + (6 % block_tokens != 0 ? 1 : 0);
        KvBlockPool pool({num_blocks, block_tokens, 2, 2});
        PagedKvCache cache(pool, 6);

        const Tensor prefill = attention_prefill(
            token_view(queries, 0, 4),
            token_view(keys, 0, 4),
            token_view(values, 0, 4),
            cache
        );
        expect_tensor_close(prefill, token_view(reference, 0, 4));

        for(size_t token = 4; token < 6; token++){
            const Tensor decoded = attention_decode(
                token_view(queries, token, 1),
                token_view(keys, token, 1),
                token_view(values, token, 1),
                cache
            );
            expect_tensor_close(decoded, token_view(reference, token, 1));
        }
    }
}

void test_access_validation(){
    KvBlockPool pool({2, 2, 1, 2});
    PagedKvCache cache(pool, 4);
    Tensor token = range_tensor(1, 1, 2, 1.0f);
    cache.append(token, token);

    expect_throw([&](){ (void)cache.key(1, 0); }, "token index");
    expect_throw([&](){ (void)cache.value(0, 1); }, "head index");
    expect_throw(
        [&](){ cache.append(Tensor({1, 2, 1}), Tensor({1, 2, 1})); },
        "block pool"
    );
}

}

int main(){
    test_pool_configuration_and_lifetime();
    test_append_across_block_boundaries();
    test_fragmented_block_mapping();
    test_exhaustion_and_capacity_are_transactional();
    test_paged_attention_matches_reference();
    test_attention_across_block_sizes();
    test_access_validation();
    std::cout << "paged KV cache tests passed\n";
    return 0;
}

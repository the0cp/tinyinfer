#include "attention.h"
#include "kv_cache.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace{

using namespace tinyinfer;

static_assert(!std::is_copy_constructible_v<ContiguousKvCache>);
static_assert(!std::is_copy_assignable_v<ContiguousKvCache>);
static_assert(std::is_move_constructible_v<ContiguousKvCache>);
static_assert(std::is_move_assignable_v<ContiguousKvCache>);

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

Tensor token_view(const Tensor& tensor, size_t first_token, size_t token_count){
    const size_t elements_per_token = tensor.shape()[1] * tensor.shape()[2];
    const size_t byte_offset =
        tensor.byte_offset() + first_token * elements_per_token * sizeof(float);
    return Tensor(
        tensor.buffer(),
        byte_offset,
        Shape{token_count, tensor.shape()[1], tensor.shape()[2]}
    );
}

void expect_tensor_close(const Tensor& actual, const Tensor& expected){
    expect(actual.shape() == expected.shape(), "tensor shape mismatch");
    for(size_t i = 0; i < actual.numel(); i++){
        expect_close(actual.data()[i], expected.data()[i]);
    }
}

Tensor range_tensor(size_t tokens, size_t heads, size_t head_dim, float first){
    Tensor tensor({tokens, heads, head_dim});
    for(size_t i = 0; i < tensor.numel(); i++){
        tensor.data()[i] = first + static_cast<float>(i) * 0.1f;
    }
    return tensor;
}

void test_cache_append_and_clear(){
    ContiguousKvCache cache({3, 2, 2});
    Tensor keys({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor values({2, 2, 2}, {8, 7, 6, 5, 4, 3, 2, 1});

    cache.append(keys, values);
    expect(cache.size_tokens() == 2, "cache size after append");
    expect(cache.capacity_tokens() == 3, "cache capacity");
    expect(cache.used_bytes() == 2 * 2 * 2 * 2 * sizeof(float), "used bytes");
    expect(cache.allocated_bytes() == 2 * 3 * 2 * 2 * sizeof(float), "allocated bytes");
    expect_close(cache.key(1, 0)[1], 6.0f);
    expect_close(cache.value(0, 1)[0], 6.0f);

    cache.clear();
    expect(cache.size_tokens() == 0, "cache clear resets logical size");
    expect_throw([&](){ (void)cache.key(0, 0); }, "token index");
}

void test_cache_validation_and_transactional_overflow(){
    expect_throw([](){ ContiguousKvCache({0, 1, 1}); }, "max_tokens");
    expect_throw([](){ ContiguousKvCache({1, 0, 1}); }, "num_heads");
    expect_throw([](){ ContiguousKvCache({1, 1, 0}); }, "head_dim");

    ContiguousKvCache cache({2, 1, 2});
    Tensor first({1, 1, 2}, {1, 2});
    cache.append(first, first);

    Tensor too_many({2, 1, 2}, {3, 4, 5, 6});
    expect_throw([&](){ cache.append(too_many, too_many); }, "capacity");
    expect(cache.size_tokens() == 1, "overflow leaves cache unchanged");
    expect_close(cache.key(0, 0)[0], 1.0f);

    expect_throw(
        [&](){ cache.append(Tensor({1, 2, 1}), Tensor({1, 2, 1})); },
        "configuration"
    );
    expect_throw([&](){ (void)cache.value(0, 1); }, "head index");
}

void test_reference_attention_known_values(){
    Tensor queries({2, 1, 2}, {1, 0, 1, 0});
    Tensor keys({2, 1, 2}, {1, 0, 0, 1});
    Tensor values({2, 1, 2}, {2, 0, 0, 4});

    Tensor output = causal_attention_reference(queries, keys, values);
    expect_close(output.data()[0], 2.0f);
    expect_close(output.data()[1], 0.0f);

    const float first_weight = std::exp(1.0f / std::sqrt(2.0f));
    const float denominator = first_weight + 1.0f;
    expect_close(output.data()[2], 2.0f * first_weight / denominator);
    expect_close(output.data()[3], 4.0f / denominator);
}

void test_grouped_query_head_mapping(){
    Tensor queries({1, 4, 2}, {1, 0, 0, 1, 1, 1, 2, 1});
    Tensor keys({1, 2, 2}, {1, 0, 0, 1});
    Tensor values({1, 2, 2}, {10, 11, 20, 21});

    const Tensor output = causal_attention_reference(queries, keys, values);
    const std::vector<float> expected{10, 11, 10, 11, 20, 21, 20, 21};
    for(size_t i = 0; i < expected.size(); i++){
        expect_close(output.data()[i], expected[i]);
    }
}

void test_mha_gqa_and_mqa_prefill_decode(){
    constexpr size_t tokens = 4;
    constexpr size_t query_heads = 4;
    constexpr size_t head_dim = 2;
    const Tensor queries = range_tensor(tokens, query_heads, head_dim, 0.25f);

    for(size_t kv_heads : {4U, 2U, 1U}){
        const Tensor keys = range_tensor(tokens, kv_heads, head_dim, 1.25f);
        const Tensor values = range_tensor(tokens, kv_heads, head_dim, 2.25f);
        const Tensor reference = causal_attention_reference(queries, keys, values);

        ContiguousKvCache cache({tokens, kv_heads, head_dim});
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
        expect(cache.num_heads() == kv_heads, "cache stores only KV heads");
    }
}

void test_grouped_query_validation(){
    expect_throw(
        [](){
            (void)causal_attention_reference(
                Tensor({1, 3, 2}), Tensor({1, 2, 2}), Tensor({1, 2, 2})
            );
        },
        "divisible"
    );
    expect_throw(
        [](){
            (void)causal_attention_reference(
                Tensor({2, 4, 2}), Tensor({1, 2, 2}), Tensor({1, 2, 2})
            );
        },
        "token counts"
    );
    expect_throw(
        [](){
            (void)causal_attention_reference(
                Tensor({1, 4, 3}), Tensor({1, 2, 2}), Tensor({1, 2, 2})
            );
        },
        "head dimensions"
    );
    expect_throw(
        [](){
            (void)causal_attention_reference(
                Tensor({1, 4, 2}), Tensor({1, 2, 2}), Tensor({1, 1, 2})
            );
        },
        "K and V"
    );

    ContiguousKvCache wrong_cache({2, 1, 2});
    expect_throw(
        [&](){
            (void)attention_decode(
                Tensor({1, 4, 2}),
                Tensor({1, 2, 2}),
                Tensor({1, 2, 2}),
                wrong_cache
            );
        },
        "does not match"
    );
    expect(wrong_cache.size_tokens() == 0, "cache mismatch appends no KV token");
}

void test_prefill_and_decode_match_full_recomputation(){
    Tensor queries(
        {4, 2, 3},
        {
            1, 0, 1,  0, 1, 0,
            1, 1, 0,  1, 0, 1,
            0, 1, 1,  1, 1, 0,
            2, 1, 0,  0, 1, 2
        }
    );
    Tensor keys(
        {4, 2, 3},
        {
            1, 0, 0,  0, 1, 0,
            0, 1, 0,  0, 0, 1,
            0, 0, 1,  1, 1, 0,
            1, 1, 0,  0, 1, 1
        }
    );
    Tensor values(
        {4, 2, 3},
        {
            1, 2, 3,  4, 5, 6,
            2, 3, 4,  5, 6, 7,
            3, 4, 5,  6, 7, 8,
            4, 5, 6,  7, 8, 9
        }
    );

    const Tensor full = causal_attention_reference(queries, keys, values);
    ContiguousKvCache cache({4, 2, 3});
    const Tensor prefill = attention_prefill(
        token_view(queries, 0, 3),
        token_view(keys, 0, 3),
        token_view(values, 0, 3),
        cache
    );
    expect_tensor_close(prefill, token_view(full, 0, 3));

    const Tensor decoded = attention_decode(
        token_view(queries, 3, 1),
        token_view(keys, 3, 1),
        token_view(values, 3, 1),
        cache
    );
    expect_tensor_close(decoded, token_view(full, 3, 1));
    expect(cache.size_tokens() == 4, "decode appends one token");
}

void test_attention_phase_contracts(){
    Tensor token({1, 1, 2}, {1, 2});
    ContiguousKvCache cache({2, 1, 2});
    attention_prefill(token, token, token, cache);
    expect_throw(
        [&](){ (void)attention_prefill(token, token, token, cache); },
        "empty KV cache"
    );

    ContiguousKvCache decode_cache({3, 1, 2});
    Tensor two_tokens({2, 1, 2}, {1, 2, 3, 4});
    expect_throw(
        [&](){ (void)attention_decode(two_tokens, two_tokens, two_tokens, decode_cache); },
        "exactly one token"
    );
    expect(decode_cache.size_tokens() == 0, "invalid decode leaves cache unchanged");

    ContiguousKvCache wrong_shape({2, 2, 2});
    expect_throw(
        [&](){ (void)attention_decode(token, token, token, wrong_shape); },
        "does not match"
    );
}

}

int main(){
    test_cache_append_and_clear();
    test_cache_validation_and_transactional_overflow();
    test_reference_attention_known_values();
    test_grouped_query_head_mapping();
    test_mha_gqa_and_mqa_prefill_decode();
    test_grouped_query_validation();
    test_prefill_and_decode_match_full_recomputation();
    test_attention_phase_contracts();
    std::cout << "attention tests passed\n";
    return 0;
}

#include "ops.h"
#include "decoder_block.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace{

using namespace tinyinfer;

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

void expect_tensor_close(
    const Tensor& actual,
    const std::vector<float>& expected,
    float tolerance = 1e-5f
){
    expect(actual.numel() == expected.size(), "tensor element count");
    for(size_t i = 0; i < expected.size(); i++){
        expect_close(actual.data()[i], expected[i], tolerance);
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
    const size_t hidden_size = tensor.shape()[1];
    return Tensor(
        tensor.buffer(),
        tensor.byte_offset() + first_token * hidden_size * sizeof(float),
        {token_count, hidden_size}
    );
}

void test_transformer_ops(){
    const Tensor table(
        {3, 2},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
    );
    const std::vector<int64_t> token_ids{2, 0};
    expect_tensor_close(embedding(table, token_ids), {5.0f, 6.0f, 1.0f, 2.0f});
    expect_throw(
        [&](){
            const std::vector<int64_t> invalid{-1};
            (void)embedding(table, invalid);
        },
        "out of range"
    );

    const Tensor input({2, 2}, {1.0f, 2.0f, -3.0f, 4.0f});
    const Tensor weight({2}, {2.0f, 0.5f});
    const Tensor normalized = rms_norm(input, weight, 1e-6f);
    expect_close(normalized.data()[0], 2.0f / std::sqrt(2.5f + 1e-6f));
    expect_close(normalized.data()[1], 1.0f / std::sqrt(2.5f + 1e-6f));
    expect_close(normalized.data()[2], -6.0f / std::sqrt(12.5f + 1e-6f));
    expect_close(normalized.data()[3], 2.0f / std::sqrt(12.5f + 1e-6f));

    const Tensor gate({1, 3}, {-1.0f, 0.0f, 2.0f});
    const Tensor up({1, 3}, {3.0f, 4.0f, -2.0f});
    const Tensor activated = silu_mul(gate, up);
    expect_close(activated.data()[0], -3.0f / (1.0f + std::exp(1.0f)));
    expect_close(activated.data()[1], 0.0f);
    expect_close(activated.data()[2], -4.0f / (1.0f + std::exp(-2.0f)));

    const Tensor matrix({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    expect_tensor_close(linear(input, matrix), {7.0f, 10.0f, 9.0f, 10.0f});
    expect_throw([&](){ (void)rms_norm(input, Tensor({3}), 1e-6f); }, "shape");
    expect_throw([&](){ (void)silu_mul(gate, Tensor({1, 2})); }, "shape");
}

void test_half_split_rope_layout(){
    const Tensor input({1, 1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    expect_tensor_close(rope_half_split(input, 0, 100.0f), {1.0f, 2.0f, 3.0f, 4.0f});

    const float cosine_1 = std::cos(1.0f);
    const float sine_1 = std::sin(1.0f);
    const float cosine_point_1 = std::cos(0.1f);
    const float sine_point_1 = std::sin(0.1f);
    expect_tensor_close(
        rope_half_split(input, 1, 100.0f),
        {
            cosine_1 - 3.0f * sine_1,
            2.0f * cosine_point_1 - 4.0f * sine_point_1,
            3.0f * cosine_1 + sine_1,
            4.0f * cosine_point_1 + 2.0f * sine_point_1
        }
    );
    expect_throw([&](){ (void)rope_half_split(Tensor({1, 1, 3}), 0, 100.0f); }, "even");
}

// Oracle values were produced by a one-layer Qwen2Model in
// transformers==4.43.1 using eager float32 CPU inference. PyTorch stores
// Linear weights as [out, in], so the matrices below are transposed into
// tinyinfer's [in, out] layout.
DecoderBlockWeights make_weights(){
    return {
        Tensor({4}, {1.0f, 1.10000002f, 0.899999976f, 1.04999995f}),
        Tensor({4, 4}, {
            -0.199999988f, -0.0399999991f, 0.119999997f, -0.159999996f,
            -0.159999996f, 0.0f, 0.159999996f, -0.119999997f,
            -0.119999997f, 0.0399999991f, 0.199999988f, -0.0799999982f,
            -0.0799999982f, 0.0799999982f, -0.199999988f, -0.0399999991f
        }),
        Tensor({4}, {-0.159999996f, -0.119999997f, -0.0799999982f, -0.0399999991f}),
        Tensor({4, 2}, {
            -0.119999997f, 0.0399999991f, -0.0799999982f, 0.0799999982f,
            -0.0399999991f, 0.119999997f, 0.0f, 0.159999996f
        }),
        Tensor({2}, {-0.0799999982f, -0.0399999991f}),
        Tensor({4, 2}, {
            -0.0399999991f, 0.119999997f, 0.0f, 0.159999996f,
            0.0399999991f, 0.199999988f, 0.0799999982f, -0.199999988f
        }),
        Tensor({2}, {0.0f, 0.0399999991f}),
        Tensor({4, 4}, {
            0.0399999991f, 0.199999988f, -0.0799999982f, 0.0799999982f,
            0.0799999982f, -0.199999988f, -0.0399999991f, 0.119999997f,
            0.119999997f, -0.159999996f, 0.0f, 0.159999996f,
            0.159999996f, -0.119999997f, 0.0399999991f, 0.199999988f
        }),
        Tensor({4}, {0.949999988f, 1.0f, 1.04999995f, 0.899999976f}),
        Tensor({4, 8}, {
            0.0799999982f, -0.199999988f, -0.0399999991f, 0.119999997f,
            -0.159999996f, 0.0f, 0.159999996f, -0.119999997f,
            0.119999997f, -0.159999996f, 0.0f, 0.159999996f,
            -0.119999997f, 0.0399999991f, 0.199999988f, -0.0799999982f,
            0.159999996f, -0.119999997f, 0.0399999991f, 0.199999988f,
            -0.0799999982f, 0.0799999982f, -0.199999988f, -0.0399999991f,
            0.199999988f, -0.0799999982f, 0.0799999982f, -0.199999988f,
            -0.0399999991f, 0.119999997f, -0.159999996f, 0.0f
        }),
        Tensor({4, 8}, {
            0.119999997f, -0.159999996f, 0.0f, 0.159999996f,
            -0.119999997f, 0.0399999991f, 0.199999988f, -0.0799999982f,
            0.159999996f, -0.119999997f, 0.0399999991f, 0.199999988f,
            -0.0799999982f, 0.0799999982f, -0.199999988f, -0.0399999991f,
            0.199999988f, -0.0799999982f, 0.0799999982f, -0.199999988f,
            -0.0399999991f, 0.119999997f, -0.159999996f, 0.0f,
            -0.199999988f, -0.0399999991f, 0.119999997f, -0.159999996f,
            0.0f, 0.159999996f, -0.119999997f, 0.0399999991f
        }),
        Tensor({8, 4}, {
            0.159999996f, 0.0399999991f, -0.0799999982f, -0.199999988f,
            0.199999988f, 0.0799999982f, -0.0399999991f, -0.159999996f,
            -0.199999988f, 0.119999997f, 0.0f, -0.119999997f,
            -0.159999996f, 0.159999996f, 0.0399999991f, -0.0799999982f,
            -0.119999997f, 0.199999988f, 0.0799999982f, -0.0399999991f,
            -0.0799999982f, -0.199999988f, 0.119999997f, 0.0f,
            -0.0399999991f, -0.159999996f, 0.159999996f, 0.0399999991f,
            0.0f, -0.119999997f, 0.199999988f, 0.0799999982f
        })
    };
}

void test_block_matches_transformers_reference(){
    const DecoderBlockConfig config{4, 8, 2, 1, 1e-6f, 10000.0f};
    const DecoderBlockWeights weights = make_weights();
    const Tensor input(
        {3, 4},
        {
            0.2f, -0.1f, 0.4f, 0.3f,
            -0.3f, 0.5f, 0.1f, -0.2f,
            0.6f, 0.2f, -0.4f, 0.1f
        }
    );
    const std::vector<float> expected{
        0.24638021f, -0.132970303f, 0.395789713f, 0.348987848f,
        -0.231293008f, 0.436124235f, 0.0872438997f, -0.116716951f,
        0.647095561f, 0.123681463f, -0.381319821f, 0.164122835f
    };

    ContiguousKvCache full_cache({3, 1, 2});
    const Tensor full = decoder_block_prefill(input, config, weights, full_cache);
    expect_tensor_close(full, expected, 2e-5f);
    expect(full_cache.size_tokens() == 3, "full prefill cache size");

    ContiguousKvCache decode_cache({3, 1, 2});
    const Tensor prefix = decoder_block_prefill(
        token_view(input, 0, 2),
        config,
        weights,
        decode_cache
    );
    expect_tensor_close(
        prefix,
        std::vector<float>(expected.begin(), expected.begin() + 8),
        2e-5f
    );
    const Tensor decoded = decoder_block_decode(
        token_view(input, 2, 1),
        config,
        weights,
        decode_cache
    );
    expect_tensor_close(
        decoded,
        std::vector<float>(expected.begin() + 8, expected.end()),
        2e-5f
    );
    expect(decode_cache.size_tokens() == 3, "decode cache size");
}

void test_block_validation_precedes_cache_mutation(){
    const DecoderBlockConfig config{4, 8, 2, 1, 1e-6f, 10000.0f};
    DecoderBlockWeights weights = make_weights();
    weights.down_weight = Tensor({7, 4});
    ContiguousKvCache cache({2, 1, 2});
    expect_throw(
        [&](){
            (void)decoder_block_prefill(Tensor({1, 4}), config, weights, cache);
        },
        "down_weight"
    );
    expect(cache.size_tokens() == 0, "invalid block leaves cache empty");
}

void test_cache_preconditions_precede_projection(){
    const DecoderBlockConfig config{4, 8, 2, 1, 1e-6f, 10000.0f};
    const DecoderBlockWeights weights = make_weights();
    const Tensor one_token({1, 4}, {0.2f, -0.1f, 0.4f, 0.3f});

    ContiguousKvCache non_empty({2, 1, 2});
    (void)decoder_block_prefill(one_token, config, weights, non_empty);
    expect_throw(
        [&](){ (void)decoder_block_prefill(one_token, config, weights, non_empty); },
        "empty KV cache"
    );
    expect(non_empty.size_tokens() == 1, "rejected prefill leaves cache unchanged");

    ContiguousKvCache too_small({2, 1, 2});
    expect_throw(
        [&](){ (void)decoder_block_prefill(Tensor({3, 4}), config, weights, too_small); },
        "capacity"
    );
    expect(too_small.size_tokens() == 0, "oversized prefill leaves cache empty");

    ContiguousKvCache full({1, 1, 2});
    (void)decoder_block_prefill(one_token, config, weights, full);
    expect_throw(
        [&](){ (void)decoder_block_decode(one_token, config, weights, full); },
        "capacity"
    );
    expect(full.size_tokens() == 1, "rejected decode leaves cache unchanged");
}

}

int main(){
    test_transformer_ops();
    test_half_split_rope_layout();
    test_block_matches_transformers_reference();
    test_block_validation_precedes_cache_mutation();
    test_cache_preconditions_precede_projection();
    std::cout << "decoder block tests passed\n";
    return 0;
}

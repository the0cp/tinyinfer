#include <tinyinfer/llm/qwen2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

void expect_close(const Tensor& actual, const Tensor& expected, float tolerance = 1e-4f){
    expect(actual.shape() == expected.shape(), "tensor shapes match");
    for(size_t i = 0; i < actual.numel(); i++){
        const float actual_value = actual.data()[i];
        const float expected_value = expected.data()[i];
        if(
            !std::isfinite(actual_value) ||
            !std::isfinite(expected_value) ||
            std::fabs(actual_value - expected_value) > tolerance
        ){
            throw std::runtime_error(
                "EXPECT_CLOSE failed at " + std::to_string(i) +
                ": actual=" + std::to_string(actual_value) +
                ", expected=" + std::to_string(expected_value)
            );
        }
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

void test_expect_close_rejects_non_finite(){
    Tensor finite({1});
    Tensor non_finite({1});
    finite.data()[0] = 0.0f;
    non_finite.data()[0] = std::numeric_limits<float>::quiet_NaN();

    expect_throw(
        [&](){ expect_close(non_finite, finite); },
        "EXPECT_CLOSE failed"
    );
    expect_throw(
        [&](){ expect_close(finite, non_finite); },
        "EXPECT_CLOSE failed"
    );
}

struct Spec{
    std::string name;
    Shape shape;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

enum class Mutation{
    None,
    MissingTensor,
    WrongShape,
    WrongDtype,
    Overlap,
    ShortFile,
    UntiedEmbedding
};

uint64_t align64(uint64_t offset){
    return (offset + 63) / 64 * 64;
}

std::vector<Spec> tiny_specs(){
    const size_t hidden = 4;
    const size_t intermediate = 8;
    const size_t kv_size = 2;
    return {
        {"model.embed_tokens.weight", {16, hidden}},
        {"model.layers.0.input_layernorm.weight", {hidden}},
        {"model.layers.0.self_attn.q_proj.weight", {hidden, hidden}},
        {"model.layers.0.self_attn.q_proj.bias", {hidden}},
        {"model.layers.0.self_attn.k_proj.weight", {hidden, kv_size}},
        {"model.layers.0.self_attn.k_proj.bias", {kv_size}},
        {"model.layers.0.self_attn.v_proj.weight", {hidden, kv_size}},
        {"model.layers.0.self_attn.v_proj.bias", {kv_size}},
        {"model.layers.0.self_attn.o_proj.weight", {hidden, hidden}},
        {"model.layers.0.post_attention_layernorm.weight", {hidden}},
        {"model.layers.0.mlp.gate_proj.weight", {hidden, intermediate}},
        {"model.layers.0.mlp.up_proj.weight", {hidden, intermediate}},
        {"model.layers.0.mlp.down_proj.weight", {intermediate, hidden}},
        {"model.norm.weight", {hidden}}
    };
}

std::filesystem::path write_package(const std::string& name, Mutation mutation){
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / ("tinyinfer_qwen_" + name);
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    std::vector<Spec> specs = tiny_specs();
    std::ofstream weights(directory / "weights.bin", std::ios::binary);
    uint64_t offset = 0;
    for(size_t i = 0; i < specs.size(); i++){
        Spec& spec = specs[i];
        spec.offset = align64(offset);
        spec.bytes = tensor_numel(spec.shape) * sizeof(BFloat16);
        const std::vector<char> padding(static_cast<size_t>(spec.offset - offset), 0);
        weights.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        const std::vector<BFloat16> values(
            tensor_numel(spec.shape),
            BFloat16::from_float(static_cast<float>(i + 1))
        );
        weights.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(spec.bytes)
        );
        offset = spec.offset + spec.bytes;
    }
    weights.close();

    if(mutation == Mutation::ShortFile){
        std::filesystem::resize_file(directory / "weights.bin", offset - sizeof(BFloat16));
    }

    std::ofstream manifest(directory / "model.tiq");
    manifest << "TINYINFER_QWEN2 1\n"
             << "weights weights.bin\n"
             << "model_id test/qwen2\n"
             << "revision test-revision\n"
             << "source_sha256 " << std::string(64, '0') << "\n"
             << "model_type qwen2\n"
             << "hidden_act silu\n"
             << "rope_layout half_split\n"
             << "qkv_bias 1\n"
             << "tie_word_embeddings "
             << (mutation == Mutation::UntiedEmbedding ? 0 : 1) << "\n"
             << "alignment 64\n"
             << "vocab_size 16\n"
             << "hidden_size 4\n"
             << "intermediate_size 8\n"
             << "num_layers 1\n"
             << "query_heads 2\n"
             << "kv_heads 1\n"
             << "max_position_embeddings 32\n"
             << "rms_norm_epsilon 1e-6\n"
             << "rope_theta 10000\n"
             << "bos_token_id 1\n"
             << "eos_token_id 2\n";

    for(size_t i = 0; i < specs.size(); i++){
        if(mutation == Mutation::MissingTensor && i + 1 == specs.size()){
            continue;
        }
        Spec spec = specs[i];
        if(mutation == Mutation::WrongShape && spec.name.ends_with("q_proj.bias")){
            spec.shape = {3};
            spec.bytes = 3 * sizeof(BFloat16);
        }
        if(mutation == Mutation::Overlap && i == 1){
            spec.offset = 0;
        }
        manifest << "tensor " << spec.name << ' '
                 << (mutation == Mutation::WrongDtype && i == 0 ? "f32" : "bf16")
                 << ' ' << spec.shape.size();
        for(size_t dimension : spec.shape){
            manifest << ' ' << dimension;
        }
        manifest << ' ' << spec.offset << ' ' << spec.bytes << "\n";
    }
    manifest << "end\n";
    manifest.close();
    return directory / "model.tiq";
}

void test_loads_valid_package(){
    const std::filesystem::path manifest = write_package("valid", Mutation::None);
    const Qwen2Model model = load_qwen2_model(manifest);
    expect(model.model_id == "test/qwen2", "model id");
    expect(model.config.vocab_size == 16, "vocabulary size");
    expect(model.config.num_layers == 1, "layer count");
    expect(model.layers.size() == 1, "loaded layers");
    expect(model.token_embedding.shape() == Shape({16, 4}), "embedding shape");
    expect(model.final_norm.shape() == Shape({4}), "final norm shape");
    expect(model.token_embedding.dtype() == DataType::BFloat16, "embedding dtype");
    expect(model.layers[0].q_weight.dtype() == DataType::BFloat16, "layer dtype");
    expect(model.token_embedding.bfloat16_data()[0].to_float() == 1.0f, "embedding payload");
    expect(model.final_norm.bfloat16_data()[0].to_float() == 14.0f, "final norm payload");
    expect(
        model.token_embedding.buffer().get() == model.layers[0].q_weight.buffer().get(),
        "all tensors share one immutable weight buffer"
    );
    std::filesystem::remove_all(manifest.parent_path());
}

void test_rejects_invalid_packages(){
    struct Case{
        const char* name;
        Mutation mutation;
        const char* expected;
    };
    const std::vector<Case> cases{
        {"missing", Mutation::MissingTensor, "missing tensor"},
        {"shape", Mutation::WrongShape, "shape mismatch"},
        {"dtype", Mutation::WrongDtype, "invalid bfloat16"},
        {"overlap", Mutation::Overlap, "overlap"},
        {"short", Mutation::ShortFile, "weights size"},
        {"untied", Mutation::UntiedEmbedding, "tie_word_embeddings"}
    };

    for(const Case& test : cases){
        const std::filesystem::path manifest = write_package(test.name, test.mutation);
        expect_throw([&](){ (void)load_qwen2_model(manifest); }, test.expected);
        std::filesystem::remove_all(manifest.parent_path());
    }
}

void test_decoder_prefill_decode_and_reset(){
    const std::filesystem::path manifest = write_package("decoder", Mutation::None);
    const Qwen2Model model = load_qwen2_model(manifest);
    Qwen2Decoder decoder(model, 4);

    expect(decoder.capacity_tokens() == 4, "decoder capacity");
    expect(decoder.size_tokens() == 0, "new decoder cache is empty");
    expect_throw([&](){ (void)decoder.decode(1); }, "prefill first");
    expect_throw([&](){ (void)decoder.prefill({}); }, "at least one token");

    const int64_t invalid[]{16};
    expect_throw([&](){ (void)decoder.prefill(invalid); }, "out of range");
    expect(decoder.size_tokens() == 0, "invalid token leaves cache empty");

    const int64_t prompt[]{1, 2};
    const Tensor prefill_logits = decoder.prefill(prompt);
    expect(prefill_logits.shape() == Shape({1, 16}), "prefill logits shape");
    expect(decoder.size_tokens() == 2, "prefill cache size");
    expect_throw([&](){ (void)decoder.prefill(prompt); }, "empty KV cache");
    expect(decoder.size_tokens() == 2, "repeated prefill leaves cache unchanged");

    const Tensor cached_logits = decoder.decode(3);
    expect(decoder.size_tokens() == 3, "decode cache size");

    Qwen2Decoder full_recompute(model, 4);
    const int64_t full_sequence[]{1, 2, 3};
    const Tensor full_logits = full_recompute.prefill(full_sequence);
    expect_close(cached_logits, full_logits);

    decoder.reset();
    expect(decoder.size_tokens() == 0, "reset clears every layer cache");
    expect_close(decoder.prefill(prompt), prefill_logits);

    std::filesystem::remove_all(manifest.parent_path());
}

void test_decoder_rejects_invalid_model_and_capacity(){
    const std::filesystem::path manifest = write_package("decoder_invalid", Mutation::None);
    Qwen2Model model = load_qwen2_model(manifest);

    expect_throw([&](){ Qwen2Decoder decoder(model, 0); }, "capacity");
    expect_throw([&](){ Qwen2Decoder decoder(model, 33); }, "model limit");

    model.layers.clear();
    expect_throw([&](){ Qwen2Decoder decoder(model, 4); }, "layer count");
    std::filesystem::remove_all(manifest.parent_path());
}

void test_greedy_generation_stops_and_resets(){
    const std::filesystem::path manifest = write_package("generation", Mutation::None);
    const Qwen2Model model = load_qwen2_model(manifest);
    const int64_t prompt[]{1, 2};

    Qwen2Decoder oracle(model, 4);
    const Tensor first_logits = oracle.prefill(prompt);
    const int64_t first_token = static_cast<int64_t>(
        std::max_element(
            first_logits.data(),
            first_logits.data() + first_logits.numel()
        ) - first_logits.data()
    );

    Qwen2Decoder decoder(model, 4);
    expect(
        greedy_generate(decoder, prompt, 2, first_token) ==
            std::vector<int64_t>({first_token}),
        "EOS stops greedy generation"
    );
    expect(decoder.size_tokens() == 2, "stopped token is not decoded into cache");

    const std::vector<int64_t> generated = greedy_generate(decoder, prompt, 2, -1);
    expect(generated.size() == 2, "maximum token count stops generation");
    expect(generated.front() == first_token, "generation resets request state");
    expect(decoder.size_tokens() == 3, "only tokens needed by the next step enter cache");

    expect(
        greedy_generate(decoder, prompt, 3, -1).size() == 3,
        "the final generated token needs no cache slot"
    );
    expect(decoder.size_tokens() == 4, "generation uses exactly prompt plus N-1 slots");

    expect_throw(
        [&](){ (void)greedy_generate(decoder, prompt, 4, -1); },
        "capacity"
    );
    expect(decoder.size_tokens() == 4, "invalid request is rejected before reset");

    expect(greedy_generate(decoder, prompt, 0, -1).empty(), "zero-token generation");
    expect(decoder.size_tokens() == 0, "zero-token generation resets request state");

    std::filesystem::remove_all(manifest.parent_path());
}

}

int main(){
    try{
        test_expect_close_rejects_non_finite();
        test_loads_valid_package();
        test_rejects_invalid_packages();
        test_decoder_prefill_decode_and_reset();
        test_decoder_rejects_invalid_model_and_capacity();
        test_greedy_generation_stops_and_resets();
        std::cout << "qwen model tests passed\n";
        return 0;
    }catch(const std::exception& error){
        std::cerr << error.what() << '\n';
        return 1;
    }
}

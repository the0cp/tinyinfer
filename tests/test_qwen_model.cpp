#include "qwen_model.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
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

}

int main(){
    test_loads_valid_package();
    test_rejects_invalid_packages();
    std::cout << "qwen model loader tests passed\n";
    return 0;
}

#pragma once

#include "decoder_block.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tinyinfer{

// Native model boundary for the pinned Qwen2.5 checkpoint. Decoder math stays
// model-neutral in decoder_block.h; checkpoint names and metadata live here.
struct Qwen2Config{
    size_t vocab_size;
    size_t num_layers;
    size_t max_position_embeddings;
    int64_t bos_token_id;
    int64_t eos_token_id;
    DecoderBlockConfig block;
};

struct Qwen2Model{
    std::string model_id;
    std::string revision;
    std::string source_sha256;
    Qwen2Config config;
    Tensor token_embedding;
    std::vector<DecoderBlockWeights> layers;
    Tensor final_norm;
};

Qwen2Model load_qwen2_model(const std::filesystem::path& manifest_path);

}

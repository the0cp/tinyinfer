#pragma once

#include <tinyinfer/llm/decoder_block.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
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

class Qwen2Decoder{
public:
    Qwen2Decoder(Qwen2Model model, size_t max_tokens);

    Qwen2Decoder(const Qwen2Decoder&) = delete;
    Qwen2Decoder& operator=(const Qwen2Decoder&) = delete;
    // A moved-from decoder may only be destroyed or assigned a new decoder.
    Qwen2Decoder(Qwen2Decoder&&) noexcept = default;
    Qwen2Decoder& operator=(Qwen2Decoder&&) noexcept = default;

    Tensor prefill(std::span<const int64_t> token_ids);
    Tensor decode(int64_t token_id);
    void reset() noexcept;

    size_t size_tokens() const noexcept;
    size_t capacity_tokens() const noexcept;

private:
    Tensor last_token_logits(const Tensor& hidden_states) const;

    Qwen2Model model_;
    std::vector<ContiguousKvCache> caches_;
    size_t capacity_tokens_;
};

}

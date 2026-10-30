#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tokenizers{
class Tokenizer;
}

namespace tinyinfer{

class Qwen2Tokenizer{
public:
    Qwen2Tokenizer(Qwen2Tokenizer&&) noexcept;
    Qwen2Tokenizer& operator=(Qwen2Tokenizer&&) noexcept;
    ~Qwen2Tokenizer();

    Qwen2Tokenizer(const Qwen2Tokenizer&) = delete;
    Qwen2Tokenizer& operator=(const Qwen2Tokenizer&) = delete;

    std::vector<int64_t> encode(std::string_view text) const;
    std::string decode(
        std::span<const int64_t> token_ids,
        bool skip_special_tokens = false
    ) const;

private:
    Qwen2Tokenizer(
        std::unique_ptr<tokenizers::Tokenizer> tokenizer,
        int32_t first_special_token_id,
        size_t vocab_size
    );
    friend Qwen2Tokenizer load_qwen2_tokenizer(const std::filesystem::path& path);

    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    int32_t first_special_token_id_;
    size_t vocab_size_;
};

Qwen2Tokenizer load_qwen2_tokenizer(const std::filesystem::path& path);

std::string qwen2_chat_prompt(
    std::string_view system_prompt,
    std::string_view user_prompt
);

}

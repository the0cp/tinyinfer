#include <tinyinfer/llm/qwen2_tokenizer.h>

#include <tokenizers_cpp.h>

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace tinyinfer{

Qwen2Tokenizer::Qwen2Tokenizer(
    std::unique_ptr<tokenizers::Tokenizer> tokenizer,
    int32_t first_special_token_id,
    size_t vocab_size
) :
    tokenizer_(std::move(tokenizer)),
    first_special_token_id_(first_special_token_id),
    vocab_size_(vocab_size){}

Qwen2Tokenizer::Qwen2Tokenizer(Qwen2Tokenizer&&) noexcept = default;
Qwen2Tokenizer& Qwen2Tokenizer::operator=(Qwen2Tokenizer&&) noexcept = default;
Qwen2Tokenizer::~Qwen2Tokenizer() = default;

Qwen2Tokenizer load_qwen2_tokenizer(const std::filesystem::path& path){
    std::ifstream stream(path, std::ios::binary);
    if(!stream){
        throw std::runtime_error("Cannot open Qwen2 tokenizer: " + path.string());
    }

    std::string json_blob{
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
    };

    if(stream.bad()){
        throw std::runtime_error("Cannot read Qwen2 tokenizer: " + path.string());
    }

    if(json_blob.empty()){
        throw std::runtime_error("Qwen2 tokenizer.json is empty.");
    }

    std::unique_ptr<tokenizers::Tokenizer> tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    constexpr std::string_view required_tokens[]{
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>"
    };

    for(std::string_view token : required_tokens){
        if(tokenizer->TokenToId(std::string(token)) < 0){
            throw std::runtime_error("Qwen2 tokenizer is missing special token: " + std::string(token));
        }
    }

    const int32_t first_special_token_id = tokenizer->TokenToId(std::string(required_tokens[0]));
    const size_t vocab_size = tokenizer->GetVocabSize();

    return Qwen2Tokenizer(
        std::move(tokenizer),
        first_special_token_id,
        vocab_size
    );
}

std::vector<int64_t> Qwen2Tokenizer::encode(std::string_view text) const{
    const std::vector<int32_t> ids = tokenizer_->Encode(std::string(text));
    return {ids.begin(), ids.end()};
}

std::string Qwen2Tokenizer::decode(
    std::span<const int64_t> token_ids,
    bool skip_special_tokens
)const{
    std::vector<int32_t> ids;
    ids.reserve(token_ids.size());
    for(int64_t id : token_ids){
        if(!skip_special_tokens || id < first_special_token_id_ || static_cast<size_t>(id) >= vocab_size_){
            ids.push_back(static_cast<int32_t>(id));
        }
    }
    return tokenizer_->Decode(ids);
}

std::string qwen2_chat_prompt(
    std::string_view system_prompt,
    std::string_view user_prompt
){
    std::string prompt;
    prompt.reserve(system_prompt.size() + user_prompt.size() + 96);
    prompt += "<|im_start|>system\n";
    prompt += system_prompt;
    prompt += "<|im_end|>\n<|im_start|>user\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";

    return prompt;
}

}

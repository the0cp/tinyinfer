#include <tinyinfer/llm/qwen2.h>
#include <tinyinfer/llm/qwen2_tokenizer.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace{

struct ModelCase{
    std::vector<int64_t> prompt;
    std::vector<int64_t> generated;
    size_t vocab_size;
    std::vector<float> logits;
};

template<typename Value>
void read_value(std::ifstream& stream, Value& value, const char* field){
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if(!stream){
        throw std::runtime_error(std::string("Cannot read model case ") + field + ".");
    }
}

void read_bytes(
    std::ifstream& stream,
    void* destination,
    size_t byte_count,
    const char* field
){
    if(byte_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())){
        throw std::runtime_error(std::string("Model case ") + field + " is too large.");
    }
    stream.read(
        reinterpret_cast<char*>(destination),
        static_cast<std::streamsize>(byte_count)
    );
    if(!stream){
        throw std::runtime_error(std::string("Cannot read model case ") + field + ".");
    }
}

size_t checked_product(uint32_t left, uint32_t right, const char* field){
    if(
        left != 0 &&
        static_cast<size_t>(right) > std::numeric_limits<size_t>::max() / left
    ){
        throw std::runtime_error(std::string("Model case ") + field + " overflows size_t.");
    }
    return static_cast<size_t>(left) * right;
}

ModelCase load_case(const std::filesystem::path& path){
    if constexpr(std::endian::native != std::endian::little){
        throw std::runtime_error("Model case requires a little-endian platform.");
    }

    std::ifstream stream(path, std::ios::binary);
    if(!stream){
        throw std::runtime_error("Cannot open model case: " + path.string());
    }

    char magic[8]{};
    uint32_t version = 0;
    uint32_t prompt_tokens = 0;
    uint32_t vocab_size = 0;
    uint32_t steps = 0;
    stream.read(magic, sizeof(magic));
    read_value(stream, version, "version");
    read_value(stream, prompt_tokens, "prompt token count");
    read_value(stream, vocab_size, "vocabulary size");
    read_value(stream, steps, "step count");

    if(std::memcmp(magic, "TIQMC001", sizeof(magic)) != 0 || version != 1){
        throw std::runtime_error("Unsupported model case format.");
    }
    if(prompt_tokens == 0 || vocab_size == 0 || steps == 0){
        throw std::runtime_error("Model case dimensions must be positive.");
    }

    const size_t logit_count = checked_product(steps, vocab_size, "logit count");
    if(
        prompt_tokens > std::numeric_limits<size_t>::max() / sizeof(int64_t) ||
        steps > std::numeric_limits<size_t>::max() / sizeof(int64_t) ||
        logit_count > std::numeric_limits<size_t>::max() / sizeof(float)
    ){
        throw std::runtime_error("Model case byte count overflows size_t.");
    }

    std::vector<int64_t> prompt(prompt_tokens);
    std::vector<int64_t> generated(steps);
    std::vector<float> logits(logit_count);
    read_bytes(stream, prompt.data(), prompt.size() * sizeof(int64_t), "prompt");
    read_bytes(
        stream,
        generated.data(),
        generated.size() * sizeof(int64_t),
        "generated tokens"
    );
    read_bytes(stream, logits.data(), logits.size() * sizeof(float), "logits");
    if(!stream || stream.peek() != std::ifstream::traits_type::eof()){
        throw std::runtime_error("Model case size does not match its header.");
    }

    return {
        std::move(prompt),
        std::move(generated),
        vocab_size,
        std::move(logits)
    };
}

void check_logits(
    const tinyinfer::Tensor& actual,
    const float* expected,
    size_t vocab_size,
    size_t step
){
    if(actual.shape() != tinyinfer::Shape({1, vocab_size})){
        throw std::runtime_error("Qwen2 logits shape mismatch.");
    }

    constexpr float absolute_tolerance = 1e-3f;
    constexpr float relative_tolerance = 1e-3f;
    float maximum_error = 0.0f;
    double error_sum = 0.0;
    for(size_t token = 0; token < vocab_size; token++){
        const float actual_value = actual.data()[token];
        const float expected_value = expected[token];
        const float error = std::fabs(actual_value - expected_value);
        const float allowed = absolute_tolerance +
            relative_tolerance * std::fabs(expected_value);
        if(
            !std::isfinite(actual_value) ||
            !std::isfinite(expected_value) ||
            error > allowed
        ){
            throw std::runtime_error(
                "Logit mismatch at step " + std::to_string(step) +
                ", token " + std::to_string(token) +
                ": actual=" + std::to_string(actual_value) +
                ", expected=" + std::to_string(expected_value) +
                ", allowed=" + std::to_string(allowed)
            );
        }
        maximum_error = std::max(maximum_error, error);
        error_sum += error;
    }

    std::cout << "step " << step
              << ": max_abs_error=" << maximum_error
              << ", mean_abs_error=" << error_sum / vocab_size << "\n";
}

int64_t argmax(const tinyinfer::Tensor& logits){
    const float* begin = logits.data();
    return static_cast<int64_t>(
        std::max_element(begin, begin + logits.numel()) - begin
    );
}

}

int main(int argc, char** argv){
    if(argc != 4){
        std::cerr
            << "usage: check_qwen_model <model.tiq> <tokenizer.json> <model_case.bin>\n";
        return 2;
    }

    try{
        const ModelCase test = load_case(argv[3]);
        const tinyinfer::Qwen2Tokenizer tokenizer =
            tinyinfer::load_qwen2_tokenizer(argv[2]);
        if(
            tokenizer.encode("Hello, TinyInfer!") !=
            std::vector<int64_t>({9707, 11, 47974, 641, 802, 0})
        ){
            throw std::runtime_error("Qwen2 ASCII token ids do not match the reference.");
        }
        if(
            tokenizer.encode("a23.''-Z'cc.43") !=
            std::vector<int64_t>({64, 17, 18, 3159, 24293, 57, 6, 638, 13, 19, 18}) ||
            tokenizer.encode("  hello") != std::vector<int64_t>({220, 23811})
        ){
            throw std::runtime_error("Qwen2 split boundaries do not match the reference.");
        }
        const std::vector<int64_t> utf8_tokens{
            108386, 3837, 82707, 84238, 241, 24360, 102021, 11319
        };
        if(
            tokenizer.encode("你好，KV 缓存是什么？") != utf8_tokens ||
            tokenizer.decode(utf8_tokens) != "你好，KV 缓存是什么？"
        ){
            throw std::runtime_error("Qwen2 UTF-8 tokenizer case does not match the reference.");
        }
        const std::string prompt_text = tinyinfer::qwen2_chat_prompt(
            "You are a concise inference systems tutor.",
            "用一句话解释 KV cache。"
        );
        const std::vector<int64_t> prompt = tokenizer.encode(prompt_text);
        if(prompt != test.prompt){
            throw std::runtime_error("Qwen2 tokenizer prompt ids do not match the reference.");
        }

        tinyinfer::Qwen2Model model = tinyinfer::load_qwen2_model(argv[1]);
        if(model.config.vocab_size != test.vocab_size){
            throw std::runtime_error("Model case vocabulary does not match the model.");
        }

        const size_t capacity = prompt.size() + test.generated.size() - 1;
        tinyinfer::Qwen2Decoder decoder(std::move(model), capacity);
        tinyinfer::Tensor logits = decoder.prefill(prompt);

        for(size_t step = 0; step < test.generated.size(); step++){
            const float* expected = test.logits.data() + step * test.vocab_size;
            check_logits(logits, expected, test.vocab_size, step);

            const int64_t token = argmax(logits);
            if(token != test.generated[step]){
                throw std::runtime_error(
                    "Greedy token mismatch at step " + std::to_string(step) +
                    ": actual=" + std::to_string(token) +
                    ", expected=" + std::to_string(test.generated[step])
                );
            }
            std::cout << "step " << step << ": greedy_token=" << token << "\n";

            if(step + 1 < test.generated.size()){
                logits = decoder.decode(token);
            }
        }

        const std::string generated_text = tokenizer.decode(test.generated, true);
        if(generated_text != "KV cache，"){
            throw std::runtime_error("Qwen2 decoded text does not match the reference.");
        }

        std::cout << "generated_text=" << generated_text << "\n"
                  << "real Qwen2 text generation matches the reference\n";
        return 0;
    }catch(const std::exception& error){
        std::cerr << error.what() << "\n";
        return 1;
    }
}

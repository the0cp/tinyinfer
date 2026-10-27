#include <tinyinfer/llm/decoder_block.h>
#include <tinyinfer/compute/ops.h>
#include <tinyinfer/llm/qwen2.h>

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

struct Layer0Case{
    size_t tokens;
    size_t hidden_size;
    std::vector<int64_t> token_ids;
    std::vector<float> expected;
};

template<typename Value>
void read_value(std::ifstream& stream, Value& value, const char* field){
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if(!stream){
        throw std::runtime_error(std::string("Cannot read layer0 case ") + field + ".");
    }
}

Layer0Case load_case(const std::filesystem::path& path){
    if constexpr(std::endian::native != std::endian::little){
        throw std::runtime_error("Layer0 fixture requires a little-endian platform.");
    }
    std::ifstream stream(path, std::ios::binary);
    if(!stream){
        throw std::runtime_error("Cannot open layer0 case: " + path.string());
    }

    char magic[8]{};
    uint32_t version = 0;
    uint32_t tokens = 0;
    uint32_t hidden = 0;
    stream.read(magic, sizeof(magic));
    read_value(stream, version, "version");
    read_value(stream, tokens, "token count");
    read_value(stream, hidden, "hidden size");
    if(std::memcmp(magic, "TILYR001", sizeof(magic)) != 0 || version != 1){
        throw std::runtime_error("Unsupported layer0 case format.");
    }
    if(tokens == 0 || hidden == 0){
        throw std::runtime_error("Layer0 case dimensions must be positive.");
    }

    if(
        static_cast<size_t>(tokens) >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(hidden) ||
        static_cast<size_t>(tokens) * hidden >
        std::numeric_limits<size_t>::max() / sizeof(float)
    ){
        throw std::runtime_error("Layer0 case dimensions are too large.");
    }
    const size_t output_count = static_cast<size_t>(tokens) * hidden;
    std::vector<int64_t> token_ids(tokens);
    std::vector<float> expected(output_count);
    stream.read(
        reinterpret_cast<char*>(token_ids.data()),
        static_cast<std::streamsize>(token_ids.size() * sizeof(int64_t))
    );
    stream.read(
        reinterpret_cast<char*>(expected.data()),
        static_cast<std::streamsize>(expected.size() * sizeof(float))
    );
    if(!stream || stream.peek() != std::ifstream::traits_type::eof()){
        throw std::runtime_error("Layer0 case size does not match its header.");
    }
    return {tokens, hidden, std::move(token_ids), std::move(expected)};
}

void check_close(const tinyinfer::Tensor& actual, const std::vector<float>& expected){
    if(actual.numel() != expected.size()){
        throw std::runtime_error("Layer0 output element count mismatch.");
    }
    constexpr float absolute_tolerance = 1e-4f;
    constexpr float relative_tolerance = 1e-4f;
    float maximum_error = 0.0f;
    for(size_t i = 0; i < expected.size(); i++){
        const float error = std::fabs(actual.data()[i] - expected[i]);
        const float allowed = absolute_tolerance + relative_tolerance * std::fabs(expected[i]);
        if(!std::isfinite(actual.data()[i]) || error > allowed){
            throw std::runtime_error(
                "Layer0 mismatch at element " + std::to_string(i) +
                ": actual=" + std::to_string(actual.data()[i]) +
                ", expected=" + std::to_string(expected[i])
            );
        }
        maximum_error = std::max(maximum_error, error);
    }
    std::cout << "maximum absolute error: " << maximum_error << "\n";
}

}

int main(int argc, char** argv){
    if(argc != 3){
        std::cerr << "usage: check_qwen_layer0 <model.tiq> <layer0_case.bin>\n";
        return 2;
    }

    try{
        const Layer0Case test = load_case(argv[2]);
        const tinyinfer::Qwen2Model model = tinyinfer::load_qwen2_model(argv[1]);
        if(test.hidden_size != model.config.block.hidden_size){
            throw std::runtime_error("Layer0 case hidden size does not match the model.");
        }
        if(test.tokens > model.config.max_position_embeddings){
            throw std::runtime_error("Layer0 case exceeds the model context length.");
        }

        const tinyinfer::Tensor hidden = tinyinfer::embedding(
            model.token_embedding,
            test.token_ids
        );
        const size_t head_dim = model.config.block.hidden_size /
            model.config.block.query_heads;
        tinyinfer::ContiguousKvCache cache({
            test.tokens,
            model.config.block.kv_heads,
            head_dim
        });
        const tinyinfer::Tensor output = tinyinfer::decoder_block_prefill(
            hidden,
            model.config.block,
            model.layers.front(),
            cache
        );
        check_close(output, test.expected);
        std::cout << "real Qwen2 layer 0 matches the reference\n";
        return 0;
    }catch(const std::exception& error){
        std::cerr << error.what() << "\n";
        return 1;
    }
}

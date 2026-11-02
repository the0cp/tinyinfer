#include <tinyinfer/llm/qwen2.h>
#include <tinyinfer/llm/qwen2_tokenizer.h>
#include <tinyinfer/runtime/model_loader.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace{

using Clock = std::chrono::steady_clock;

constexpr std::string_view system_prompt = "You are a helpful assistant.";
constexpr size_t default_max_new_tokens = 32;

size_t parse_max_new_tokens(std::string_view text){
    size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);

    if(error != std::errc{} || end != text.data() + text.size() || value == 0){
        throw std::invalid_argument("max-new-tokens must be a positive integer.");
    }
    return value;
}

int64_t argmax(const tinyinfer::Tensor& logits){
    const float* begin = logits.data();
    return static_cast<int64_t>(std::max_element(begin, begin + logits.numel()) - begin);
}

tinyinfer::Tensor load_input(
    const std::filesystem::path& path,
    const tinyinfer::Shape& shape
){
    tinyinfer::Tensor input(shape);
    const size_t bytes = input.size_bytes();
    const uintmax_t file_size = std::filesystem::file_size(path);

    if(file_size != bytes){
        throw std::runtime_error(
            "Input file has " + std::to_string(file_size) +
            " bytes; model expects " + std::to_string(bytes) + "."
        );
    }

    std::ifstream stream(path, std::ios::binary);

    if(!stream){
        throw std::runtime_error("Cannot open input file: " + path.string());
    }

    if(bytes != 0){
        stream.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(bytes));

        if(!stream){
            throw std::runtime_error("Failed to read input file: " + path.string());
        }
    }
    return input;
}

int run_graph(
    const std::filesystem::path& manifest,
    const std::filesystem::path& input_path
){
    auto session = tinyinfer::ModelLoader::load(manifest);
    const tinyinfer::Tensor input = load_input(input_path, session->metadata().input_shape);

    tinyinfer::RunProfile profile;
    const tinyinfer::Tensor output = session->run(input, profile);

    std::cout << "output_shape=";
    for(size_t axis = 0; axis < output.dim(); axis++){
        std::cout << (axis == 0 ? "" : "x") << output.shape()[axis];
    }

    std::cout << "\noutput_values=" << std::setprecision(9);
    const float* output_values = output.data();
    const size_t output_size = output.numel();

    for(size_t i = 0; i < output_size; i++){
        if(i != 0){
            std::cout << ' ';
        }
        std::cout << output_values[i];
    }

    std::cout << '\n';
    if(output_size != 0){
        std::cout << "predicted_index=" << argmax(output) << '\n';
    }
    
    std::cout << std::fixed << std::setprecision(3)
              << "inference_ms=" << static_cast<double>(profile.duration_ns) / 1'000'000.0 << '\n'
              << "peak_live_bytes=" << profile.peak_live_bytes << '\n';
    return 0;
}

double milliseconds(Clock::duration duration){
    return std::chrono::duration<double, std::milli>(duration).count();
}

int generate(
    const std::filesystem::path& model_directory,
    std::string_view user_prompt,
    size_t max_new_tokens
){
    tinyinfer::Qwen2Tokenizer tokenizer = tinyinfer::load_qwen2_tokenizer(model_directory / "tokenizer.json");
    const std::vector<int64_t> prompt = tokenizer.encode(tinyinfer::qwen2_chat_prompt(system_prompt, user_prompt));

    tinyinfer::Qwen2Model model = tinyinfer::load_qwen2_model(model_directory / "model.tiq");

    const size_t context_limit = model.config.max_position_embeddings;
    const size_t cached_gened_tokens = max_new_tokens - 1;

    if(prompt.size() > context_limit || cached_gened_tokens > context_limit - prompt.size()){
        throw std::length_error("Prompt and generated tokens exceed the model context limit.");
    }

    const size_t capacity = prompt.size() + cached_gened_tokens;
    const int64_t eos_token_id = model.config.eos_token_id;
    tinyinfer::Qwen2Decoder decoder(std::move(model), capacity);

    std::vector<int64_t> generated;
    generated.reserve(max_new_tokens);

    Clock::duration prefill_time{};
    Clock::duration decode_time{};

    size_t decode_steps = 0;

    const Clock::time_point prefill_start = Clock::now();
    tinyinfer::Tensor logits = decoder.prefill(prompt);
    prefill_time = Clock::now() - prefill_start;

    for(size_t step = 0; step < max_new_tokens; step++){
        // Get the token with the highest logit value
        const int64_t token = argmax(logits);
        generated.push_back(token);

        if(token == eos_token_id || step + 1 == max_new_tokens){
            break;
        }

        const Clock::time_point decode_start = Clock::now();
        logits = decoder.decode(token);
        decode_time += Clock::now() - decode_start;
        decode_steps++;
    }

    // Calculate performance metrics
    const double prefill_ms = milliseconds(prefill_time);
    const double decode_ms = milliseconds(decode_time);
    const double inference_ms = prefill_ms + decode_ms;
    const double decode_tokens_ps = decode_ms == 0.0 ? 0.0 : static_cast<double>(decode_steps) * 1000.0 / decode_ms;
    const double gened_tokens_ps = inference_ms == 0.0 ? 0.0 : static_cast<double>(generated.size()) * 1000.0 / inference_ms;

    std::cout << tokenizer.decode(generated, true) << "\n\n"
              << std::fixed << std::setprecision(3)
              << "prompt_tokens=" << prompt.size() << '\n'
              << "generated_tokens=" << generated.size() << '\n'
              << "prefill_ms=" << prefill_ms << '\n'
              << "decode_ms=" << decode_ms << '\n'
              << "decode_tokens_per_second=" << decode_tokens_ps << '\n'
              << "generated_tokens_per_second=" << gened_tokens_ps << '\n';
    return 0;
}

void print_usage(std::ostream& stream){
    stream << "Usage:\n" << "  tinyinfer run <model.ti> <input.f32>\n" << "  tinyinfer generate <model-directory> <prompt> [max-new-tokens]\n";
}

}

int main(int argc, char** argv){
    try{
        if(argc == 2 && std::string_view(argv[1]) == "--help"){
            print_usage(std::cout);
            return 0;
        }

        if(argc == 4 && std::string_view(argv[1]) == "run"){
            return run_graph(argv[2], argv[3]);
        }

        if(argc < 4 || argc > 5 || std::string_view(argv[1]) != "generate"){
            print_usage(std::cerr);
            return 2;
        }

        const size_t max_new_tokens = argc == 5 ? parse_max_new_tokens(argv[4]) : default_max_new_tokens;

        return generate(argv[2], argv[3], max_new_tokens);
    }catch(const std::exception& error){
        std::cerr << "tinyinfer: " << error.what() << '\n';
        return 1;
    }
}

#include <tinyinfer/compute/ops.h>
#include <tinyinfer/core/thread_pool.h>
#include <tinyinfer/llm/qwen2.h>
#include <tinyinfer/llm/qwen2_tokenizer.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace{

using namespace tinyinfer;

struct ModelCase{
    std::vector<int64_t> prompt;
    std::vector<int64_t> generated;
    size_t vocab_size;
    size_t layer_count;
    size_t hidden_size;
    std::vector<float> prefill_embedding;
    std::vector<float> prefill_layers;
    std::vector<float> prefill_final;
    std::vector<float> decode_embeddings;
    std::vector<float> decode_layers;
    std::vector<float> decode_final;
    std::vector<float> logits;
};

struct Metrics{
    float maximum_error;
    double mean_error;
    double cosine_similarity;
};

template<typename Value>
void read_value(std::ifstream& stream, Value& value, const char* field){
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if(!stream){
        throw std::runtime_error(std::string("Cannot read model case ") + field + ".");
    }
}

size_t checked_product(size_t left, size_t right, std::string_view field){
    if(left != 0 && right > std::numeric_limits<size_t>::max() / left){
        throw std::runtime_error(
            "Model case " + std::string(field) + " overflows size_t."
        );
    }
    return left * right;
}

template<typename Value>
std::vector<Value> read_vector(
    std::ifstream& stream,
    size_t count,
    const char* field
){
    const size_t byte_count = checked_product(count, sizeof(Value), field);
    if(byte_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())){
        throw std::runtime_error(std::string("Model case ") + field + " is too large.");
    }

    std::vector<Value> result(count);
    if(byte_count != 0){
        stream.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(byte_count)
        );
    }
    if(!stream){
        throw std::runtime_error(std::string("Cannot read model case ") + field + ".");
    }
    return result;
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
    uint32_t layer_count = 0;
    uint32_t hidden_size = 0;
    uint32_t steps = 0;
    stream.read(magic, sizeof(magic));
    read_value(stream, version, "version");
    read_value(stream, prompt_tokens, "prompt token count");
    read_value(stream, vocab_size, "vocabulary size");
    read_value(stream, layer_count, "layer count");
    read_value(stream, hidden_size, "hidden size");
    read_value(stream, steps, "step count");

    if(std::memcmp(magic, "TIQMC002", sizeof(magic)) != 0 || version != 2){
        throw std::runtime_error("Unsupported model case format.");
    }
    if(
        prompt_tokens == 0 || vocab_size == 0 || layer_count == 0 ||
        hidden_size == 0 || steps == 0
    ){
        throw std::runtime_error("Model case dimensions must be positive.");
    }

    const size_t prompt_hidden = checked_product(
        prompt_tokens,
        hidden_size,
        "prompt hidden states"
    );
    const size_t prefill_layers = checked_product(
        layer_count,
        prompt_hidden,
        "prefill layer outputs"
    );
    const size_t decode_steps = steps - 1;
    const size_t decode_hidden = checked_product(
        decode_steps,
        hidden_size,
        "decode hidden states"
    );
    const size_t decode_layers = checked_product(
        layer_count,
        decode_hidden,
        "decode layer outputs"
    );
    const size_t logit_count = checked_product(steps, vocab_size, "logits");

    ModelCase result{
        read_vector<int64_t>(stream, prompt_tokens, "prompt"),
        read_vector<int64_t>(stream, steps, "generated tokens"),
        vocab_size,
        layer_count,
        hidden_size,
        read_vector<float>(stream, prompt_hidden, "prefill embedding"),
        read_vector<float>(stream, prefill_layers, "prefill layer outputs"),
        read_vector<float>(stream, prompt_hidden, "prefill final hidden state"),
        read_vector<float>(stream, decode_hidden, "decode embeddings"),
        read_vector<float>(stream, decode_layers, "decode layer outputs"),
        read_vector<float>(stream, decode_hidden, "decode final hidden states"),
        read_vector<float>(stream, logit_count, "logits")
    };
    if(stream.peek() != std::ifstream::traits_type::eof()){
        throw std::runtime_error("Model case size does not match its header.");
    }
    return result;
}

Metrics compare(
    const Tensor& actual,
    const float* expected,
    size_t count,
    std::string_view stage,
    std::string_view phase
){
    if(actual.numel() != count){
        throw std::runtime_error(
            std::string(stage) + " element count mismatch at " + std::string(phase) + "."
        );
    }

    constexpr float absolute_tolerance = 2e-3f;
    constexpr float relative_tolerance = 2e-3f;
    float maximum_error = 0.0f;
    double error_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;

    for(size_t index = 0; index < count; index++){
        const float actual_value = actual.data()[index];
        const float expected_value = expected[index];
        const float error = std::fabs(actual_value - expected_value);
        const float allowed = absolute_tolerance +
            relative_tolerance * std::fabs(expected_value);
        if(
            !std::isfinite(actual_value) ||
            !std::isfinite(expected_value) ||
            error > allowed
        ){
            throw std::runtime_error(
                std::string(stage) + " mismatch at " + std::string(phase) +
                ", element " + std::to_string(index) +
                ": actual=" + std::to_string(actual_value) +
                ", expected=" + std::to_string(expected_value) +
                ", allowed=" + std::to_string(allowed)
            );
        }
        maximum_error = std::max(maximum_error, error);
        error_sum += error;
        dot += static_cast<double>(actual_value) * expected_value;
        actual_norm += static_cast<double>(actual_value) * actual_value;
        expected_norm += static_cast<double>(expected_value) * expected_value;
    }

    const double norm_product = std::sqrt(actual_norm * expected_norm);
    const double cosine = norm_product == 0.0 ? 1.0 : dot / norm_product;
    return {maximum_error, error_sum / static_cast<double>(count), cosine};
}

void print_metrics(
    std::string_view phase,
    std::string_view stage,
    const Metrics& metrics
){
    std::cout << phase << ',' << stage << ','
              << std::setprecision(12) << metrics.maximum_error << ','
              << metrics.mean_error << ',' << metrics.cosine_similarity << '\n';
}

int64_t argmax(const Tensor& logits){
    const float* begin = logits.data();
    return static_cast<int64_t>(
        std::max_element(begin, begin + logits.numel()) - begin
    );
}

Tensor final_hidden(const Tensor& hidden, const Qwen2Model& model){
    return rms_norm(
        hidden,
        model.final_norm,
        model.config.block.rms_norm_epsilon
    );
}

Tensor logits(const Tensor& normalized, const Qwen2Model& model, ThreadPool* pool){
    const size_t hidden_size = model.config.block.hidden_size;
    Tensor last_hidden({1, hidden_size});
    std::copy_n(
        normalized.data() + (normalized.shape()[0] - 1) * hidden_size,
        hidden_size,
        last_hidden.data()
    );
    return matmul_bt(last_hidden, model.token_embedding, pool);
}

void check_token(const Tensor& actual, int64_t expected, size_t step){
    const int64_t token = argmax(actual);
    if(token != expected){
        throw std::runtime_error(
            "Greedy token mismatch at step " + std::to_string(step) +
            ": actual=" + std::to_string(token) +
            ", expected=" + std::to_string(expected)
        );
    }
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
        const Qwen2Tokenizer tokenizer = load_qwen2_tokenizer(argv[2]);
        const std::string prompt_text = qwen2_chat_prompt(
            "You are a concise inference systems tutor.",
            "Explain KV cache in one short sentence."
        );
        const std::vector<int64_t> prompt = tokenizer.encode(prompt_text);
        if(prompt != test.prompt){
            throw std::runtime_error("Qwen2 tokenizer prompt ids do not match the reference.");
        }

        const Qwen2Model model = load_qwen2_model(argv[1]);
        if(
            model.config.vocab_size != test.vocab_size ||
            model.config.num_layers != test.layer_count ||
            model.config.block.hidden_size != test.hidden_size
        ){
            throw std::runtime_error("Model case dimensions do not match the model.");
        }

        const size_t cpu_threads = std::max<size_t>(
            1,
            std::thread::hardware_concurrency()
        );
        std::unique_ptr<ThreadPool> pool;
        if(cpu_threads > 1){
            pool = std::make_unique<ThreadPool>(cpu_threads);
        }

        const size_t capacity = test.prompt.size() + test.generated.size() - 1;
        const size_t head_dim = test.hidden_size / model.config.block.query_heads;
        std::vector<ContiguousKvCache> caches;
        caches.reserve(test.layer_count);
        for(size_t layer = 0; layer < test.layer_count; layer++){
            caches.emplace_back(KvCacheConfig{
                capacity,
                model.config.block.kv_heads,
                head_dim
            });
        }

        const size_t prompt_hidden = test.prompt.size() * test.hidden_size;
        std::cout << "layer_metrics_begin\n"
                  << "phase,stage,max_abs_error,mean_abs_error,cosine_similarity\n";

        Tensor hidden = embedding(model.token_embedding, test.prompt);
        print_metrics(
            "prefill",
            "Embedding",
            compare(
                hidden,
                test.prefill_embedding.data(),
                prompt_hidden,
                "Embedding",
                "prefill"
            )
        );
        for(size_t layer = 0; layer < test.layer_count; layer++){
            hidden = decoder_block_prefill(
                hidden,
                model.config.block,
                model.layers[layer],
                caches[layer],
                pool.get()
            );
            const std::string stage = "Layer " + std::to_string(layer);
            print_metrics(
                "prefill",
                stage,
                compare(
                    hidden,
                    test.prefill_layers.data() + layer * prompt_hidden,
                    prompt_hidden,
                    stage,
                    "prefill"
                )
            );
        }

        Tensor normalized = final_hidden(hidden, model);
        print_metrics(
            "prefill",
            "Final RMSNorm",
            compare(
                normalized,
                test.prefill_final.data(),
                prompt_hidden,
                "Final RMSNorm",
                "prefill"
            )
        );
        Tensor output = logits(normalized, model, pool.get());
        print_metrics(
            "prefill",
            "Logits",
            compare(
                output,
                test.logits.data(),
                test.vocab_size,
                "Logits",
                "prefill"
            )
        );
        check_token(output, test.generated.front(), 0);

        const size_t layer_stride = test.layer_count * test.hidden_size;
        for(size_t step = 1; step < test.generated.size(); step++){
            const std::string phase = "decode_" + std::to_string(step);
            const int64_t token_ids[]{test.generated[step - 1]};
            hidden = embedding(model.token_embedding, token_ids);
            print_metrics(
                phase,
                "Embedding",
                compare(
                    hidden,
                    test.decode_embeddings.data() + (step - 1) * test.hidden_size,
                    test.hidden_size,
                    "Embedding",
                    phase
                )
            );
            for(size_t layer = 0; layer < test.layer_count; layer++){
                hidden = decoder_block_decode(
                    hidden,
                    model.config.block,
                    model.layers[layer],
                    caches[layer],
                    pool.get()
                );
                const std::string stage = "Layer " + std::to_string(layer);
                const size_t offset = (step - 1) * layer_stride +
                    layer * test.hidden_size;
                print_metrics(
                    phase,
                    stage,
                    compare(
                        hidden,
                        test.decode_layers.data() + offset,
                        test.hidden_size,
                        stage,
                        phase
                    )
                );
            }

            normalized = final_hidden(hidden, model);
            print_metrics(
                phase,
                "Final RMSNorm",
                compare(
                    normalized,
                    test.decode_final.data() + (step - 1) * test.hidden_size,
                    test.hidden_size,
                    "Final RMSNorm",
                    phase
                )
            );
            output = logits(normalized, model, pool.get());
            print_metrics(
                phase,
                "Logits",
                compare(
                    output,
                    test.logits.data() + step * test.vocab_size,
                    test.vocab_size,
                    "Logits",
                    phase
                )
            );
            check_token(output, test.generated[step], step);
        }

        std::cout << "layer_metrics_end\n";
        const std::string generated_text = tokenizer.decode(test.generated, true);
        std::cout << "prompt_tokens=" << test.prompt.size() << '\n'
                  << "generated_tokens=" << test.generated.size() << '\n'
                  << "decode_steps=" << test.generated.size() - 1 << '\n'
                  << "generated_text=" << generated_text << '\n'
                  << "cpu_threads=" << cpu_threads << '\n'
                  << "real Qwen2 generation and every layer match the reference\n";
        return 0;
    }catch(const std::exception& error){
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "qwen_model.h"

#include "buffer.h"
#include "model_format.h"
#include "ops.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tinyinfer{

namespace{

constexpr size_t weight_alignment = 64;

struct ScalarValue{
    std::string text;
    size_t line;
};

struct QwenManifest{
    std::filesystem::path weights_file;
    std::string model_id;
    std::string revision;
    std::string source_sha256;
    Qwen2Config config{};
    std::vector<TensorMetadata> tensors;
};

[[noreturn]] void parse_error(
    const std::filesystem::path& path,
    size_t line,
    const std::string& message
){
    throw std::runtime_error(
        path.string() + ":" + std::to_string(line) + ": " + message
    );
}

std::vector<std::string> split_tokens(std::string line){
    const size_t comment = line.find('#');
    if(comment != std::string::npos){
        line.erase(comment);
    }

    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while(stream >> token){
        tokens.push_back(std::move(token));
    }

    return tokens;
}

uint64_t parse_u64(
    const std::string& text,
    const std::filesystem::path& path,
    size_t line,
    const char* field
){
    if(text.empty() || text.front() == '-'){
        parse_error(path, line, std::string(field) + " must be non-negative");
    }

    uint64_t value = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value
    );

    if(error != std::errc{} || end != text.data() + text.size()){
        parse_error(path, line, "invalid " + std::string(field) + ": " + text);
    }

    return value;
}

int64_t parse_i64(
    const std::string& text,
    const std::filesystem::path& path,
    size_t line,
    const char* field
){
    int64_t value = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value
    );

    if(error != std::errc{} || end != text.data() + text.size()){
        parse_error(path, line, "invalid " + std::string(field) + ": " + text);
    }

    return value;
}

float parse_float(
    const std::string& text,
    const std::filesystem::path& path,
    size_t line,
    const char* field
){
    float value = 0.0f;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value
    );

    if(error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value)){
        parse_error(path, line, "invalid " + std::string(field) + ": " + text);
    }
    return value;
}

size_t as_size(
    const ScalarValue& scalar,
    const std::filesystem::path& path,
    const char* field
){
    const uint64_t value = parse_u64(scalar.text, path, scalar.line, field);
    if(value > std::numeric_limits<size_t>::max()){
        parse_error(path, scalar.line, std::string(field) + " exceeds size_t");
    }

    return static_cast<size_t>(value);
}

ScalarValue require_scalar(
    const std::unordered_map<std::string, ScalarValue>& scalars,
    const std::string& name,
    const std::filesystem::path& path
){
    const auto entry = scalars.find(name);
    if(entry == scalars.end()){
        throw std::runtime_error(path.string() + ": missing directive: " + name);
    }

    return entry->second;
}

Shape parse_shape(
    const std::vector<std::string>& tokens,
    size_t rank,
    const std::filesystem::path& path,
    size_t line
){
    Shape shape;
    shape.reserve(rank);
    for(size_t i = 0; i < rank; i++){
        const uint64_t value = parse_u64(
            tokens.at(4 + i), path, line, "tensor dimension"
        );
        if(value == 0 || value > std::numeric_limits<size_t>::max()){
            parse_error(path, line, "tensor dimensions must be positive size_t values");
        }
        shape.push_back(static_cast<size_t>(value));
    }

    return shape;
}

uint64_t expected_bytes(
    const Shape& shape,
    DataType dtype,
    const std::filesystem::path& path,
    size_t line
){
    uint64_t elements = 1;
    for(size_t dimension : shape){
        if(elements > std::numeric_limits<uint64_t>::max() / dimension){
            parse_error(path, line, "tensor element count overflows uint64");
        }
        elements *= dimension;
    }

    const size_t item_size = element_size(dtype);
    if(elements > std::numeric_limits<uint64_t>::max() / item_size){
        parse_error(path, line, "tensor byte size overflows uint64");
    }

    return elements * item_size;
}

void validate_sha256(
    const ScalarValue& scalar,
    const std::filesystem::path& path
){
    const bool valid = scalar.text.size() == 64 && std::all_of(
        scalar.text.begin(),
        scalar.text.end(),
        [](unsigned char value){
            return (value >= '0' && value <= '9') ||
                (value >= 'a' && value <= 'f');
        }
    );

    if(!valid){
        parse_error(path, scalar.line, "source_sha256 must be 64 lowercase hex digits");
    }
}

QwenManifest parse_manifest(const std::filesystem::path& path){
    std::ifstream file(path);
    if(!file){
        throw std::runtime_error("Cannot open Qwen2 manifest: " + path.string());
    }

    const std::unordered_set<std::string> allowed_scalars{
        "weights", "model_id", "revision", "source_sha256", "model_type",
        "hidden_act", "rope_layout", "qkv_bias", "tie_word_embeddings",
        "alignment", "vocab_size", "hidden_size", "intermediate_size",
        "num_layers", "query_heads", "kv_heads", "max_position_embeddings",
        "rms_norm_epsilon", "rope_theta", "bos_token_id", "eos_token_id"
    };

    std::unordered_map<std::string, ScalarValue> scalars;
    std::unordered_set<std::string> tensor_names;
    std::vector<TensorMetadata> tensors;
    bool seen_header = false;
    bool seen_end = false;
    std::string line_text;
    size_t line = 0;

    while(std::getline(file, line_text)){
        line++;
        std::vector<std::string> tokens = split_tokens(std::move(line_text));
        if(tokens.empty()){
            continue;
        }

        if(seen_end){
            parse_error(path, line, "content found after end");
        }

        if(!seen_header){
            if(tokens.size() != 2 || tokens[0] != "TINYINFER_QWEN2" || tokens[1] != "1"){
                parse_error(path, line, "expected 'TINYINFER_QWEN2 1'");
            }
            seen_header = true;
            continue;
        }

        if(tokens[0] == "end"){
            if(tokens.size() != 1){
                parse_error(path, line, "end does not accept arguments");
            }
            seen_end = true;
            continue;
        }

        if(tokens[0] == "tensor"){
            if(tokens.size() < 7 || tokens[2] != "bf16"){
                parse_error(path, line, "invalid bfloat16 tensor declaration");
            }

            if(!tensor_names.insert(tokens[1]).second){
                parse_error(path, line, "duplicate tensor: " + tokens[1]);
            }

            const uint64_t rank64 = parse_u64(tokens[3], path, line, "tensor rank");
            if(rank64 == 0 || rank64 > std::numeric_limits<size_t>::max()){
                parse_error(path, line, "tensor rank must be a positive size_t value");
            }

            const size_t rank = static_cast<size_t>(rank64);
            if(tokens.size() != 6 + rank){
                parse_error(path, line, "tensor dimension count does not match rank");
            }

            Shape shape = parse_shape(tokens, rank, path, line);
            const uint64_t offset = parse_u64(tokens[4 + rank], path, line, "tensor offset");
            const uint64_t bytes = parse_u64(tokens[5 + rank], path, line, "tensor byte size");

            if(offset % weight_alignment != 0){
                parse_error(path, line, "tensor offset is not 64-byte aligned");
            }

            if(bytes != expected_bytes(shape, DataType::BFloat16, path, line)){
                parse_error(path, line, "tensor byte size does not match shape");
            }

            tensors.push_back(TensorMetadata{
                tokens[1], DataType::BFloat16, std::move(shape), offset, bytes
            });

            continue;
        }

        if(tokens.size() != 2 || !allowed_scalars.contains(tokens[0])){
            parse_error(path, line, "unknown or malformed directive: " + tokens[0]);
        }

        if(!scalars.emplace(tokens[0], ScalarValue{tokens[1], line}).second){
            parse_error(path, line, "duplicate directive: " + tokens[0]);
        }
    }

    if(!seen_header || !seen_end){
        throw std::runtime_error(path.string() + ": incomplete Qwen2 manifest");
    }

    for(const std::string& name : allowed_scalars){
        (void)require_scalar(scalars, name, path);
    }

    auto require_value = [&](const char* name, const char* expected){
        const ScalarValue scalar = require_scalar(scalars, name, path);
        if(scalar.text != expected){
            parse_error(
                path,
                scalar.line,
                std::string(name) + " must be " + expected
            );
        }
    };

    require_value("model_type", "qwen2");
    require_value("hidden_act", "silu");
    require_value("rope_layout", "half_split");
    require_value("qkv_bias", "1");
    require_value("tie_word_embeddings", "1");
    require_value("alignment", "64");

    const ScalarValue source_hash = require_scalar(scalars, "source_sha256", path);
    validate_sha256(source_hash, path);

    Qwen2Config config{};
    config.vocab_size = as_size(require_scalar(scalars, "vocab_size", path), path, "vocab_size");
    config.num_layers = as_size(require_scalar(scalars, "num_layers", path), path, "num_layers");
    config.max_position_embeddings = as_size(
        require_scalar(scalars, "max_position_embeddings", path),
        path,
        "max_position_embeddings"
    );

    const ScalarValue bos_token = require_scalar(scalars, "bos_token_id", path);
    const ScalarValue eos_token = require_scalar(scalars, "eos_token_id", path);

    config.bos_token_id = parse_i64(bos_token.text, path, bos_token.line, "bos_token_id");
    config.eos_token_id = parse_i64(eos_token.text, path, eos_token.line, "eos_token_id");
    config.block.hidden_size = as_size(require_scalar(scalars, "hidden_size", path), path, "hidden_size");
    config.block.intermediate_size = as_size(require_scalar(scalars, "intermediate_size", path), path, "intermediate_size");
    config.block.query_heads = as_size(require_scalar(scalars, "query_heads", path), path, "query_heads");
    config.block.kv_heads = as_size(require_scalar(scalars, "kv_heads", path), path, "kv_heads");

    const ScalarValue epsilon = require_scalar(scalars, "rms_norm_epsilon", path);
    config.block.rms_norm_epsilon = parse_float(epsilon.text, path, epsilon.line, "rms_norm_epsilon");

    const ScalarValue theta = require_scalar(scalars, "rope_theta", path);
    config.block.rope_theta = parse_float(theta.text, path, theta.line, "rope_theta");

    if(
        config.vocab_size == 0 || config.num_layers == 0 ||
        config.max_position_embeddings == 0 || config.bos_token_id < 0 ||
        config.eos_token_id < 0 ||
        static_cast<uint64_t>(config.bos_token_id) >= config.vocab_size ||
        static_cast<uint64_t>(config.eos_token_id) >= config.vocab_size
    ){
        throw std::runtime_error(path.string() + ": invalid Qwen2 scalar configuration");
    }

    const ScalarValue weights = require_scalar(scalars, "weights", path);
    const std::filesystem::path weights_file = weights.text;
    if(weights_file.is_absolute() || weights_file.has_parent_path()){
        parse_error(path, weights.line, "weights must name one file beside the manifest");
    }

    return QwenManifest{
        weights_file,
        require_scalar(scalars, "model_id", path).text,
        require_scalar(scalars, "revision", path).text,
        source_hash.text,
        config,
        std::move(tensors)
    };
}

std::string layer_name(size_t layer, const char* suffix){
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

std::vector<std::pair<std::string, Shape>> expected_tensors(const Qwen2Config& config){
    const size_t hidden = config.block.hidden_size;
    const size_t intermediate = config.block.intermediate_size;

    if(
        hidden == 0 || intermediate == 0 || config.block.query_heads == 0 ||
        config.block.kv_heads == 0 || hidden % config.block.query_heads != 0 ||
        config.block.query_heads % config.block.kv_heads != 0 ||
        (hidden / config.block.query_heads) % 2 != 0 ||
        !std::isfinite(config.block.rms_norm_epsilon) ||
        config.block.rms_norm_epsilon <= 0.0f ||
        !std::isfinite(config.block.rope_theta) || config.block.rope_theta <= 0.0f
    ){
        throw std::runtime_error("Invalid Qwen2 decoder dimensions.");
    }

    const size_t kv_size = config.block.kv_heads * (hidden / config.block.query_heads);

    if(config.num_layers > (std::numeric_limits<size_t>::max() - 2) / 12){
        throw std::runtime_error("Qwen2 tensor count overflows size_t.");
    }

    std::vector<std::pair<std::string, Shape>> expected;
    expected.reserve(2 + config.num_layers * 12);
    expected.emplace_back("model.embed_tokens.weight", Shape{config.vocab_size, hidden});

    for(size_t layer = 0; layer < config.num_layers; layer++){
        expected.emplace_back(layer_name(layer, "input_layernorm.weight"), Shape{hidden});
        expected.emplace_back(layer_name(layer, "self_attn.q_proj.weight"), Shape{hidden, hidden});
        expected.emplace_back(layer_name(layer, "self_attn.q_proj.bias"), Shape{hidden});
        expected.emplace_back(layer_name(layer, "self_attn.k_proj.weight"), Shape{hidden, kv_size});
        expected.emplace_back(layer_name(layer, "self_attn.k_proj.bias"), Shape{kv_size});
        expected.emplace_back(layer_name(layer, "self_attn.v_proj.weight"), Shape{hidden, kv_size});
        expected.emplace_back(layer_name(layer, "self_attn.v_proj.bias"), Shape{kv_size});
        expected.emplace_back(layer_name(layer, "self_attn.o_proj.weight"), Shape{hidden, hidden});
        expected.emplace_back(layer_name(layer, "post_attention_layernorm.weight"), Shape{hidden});
        expected.emplace_back(layer_name(layer, "mlp.gate_proj.weight"), Shape{hidden, intermediate});
        expected.emplace_back(layer_name(layer, "mlp.up_proj.weight"), Shape{hidden, intermediate});
        expected.emplace_back(layer_name(layer, "mlp.down_proj.weight"), Shape{intermediate, hidden});
    }

    expected.emplace_back("model.norm.weight", Shape{hidden});

    return expected;
}

std::shared_ptr<Buffer> load_weight_buffer(
    const std::filesystem::path& path,
    uint64_t expected_size
){
    std::error_code error;
    const uintmax_t size = std::filesystem::file_size(path, error);

    if(error){
        throw std::runtime_error("Cannot get Qwen2 weights file size: " + path.string());
    }

    if(size == 0 || size > std::numeric_limits<size_t>::max()){
        throw std::runtime_error("Qwen2 weights file has an unsupported size.");
    }

    if(size != expected_size){
        throw std::runtime_error("Qwen2 weights size does not match the manifest.");
    }

    if(size > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max())){
        throw std::runtime_error("Qwen2 weights file is too large for one read.");
    }

    std::shared_ptr<Buffer> buffer = make_cpu_buffer(static_cast<size_t>(size));
    std::ifstream stream(path, std::ios::binary);

    if(!stream){
        throw std::runtime_error("Cannot open Qwen2 weights file: " + path.string());
    }

    stream.read(
        reinterpret_cast<char*>(buffer->data()),
        static_cast<std::streamsize>(size)
    );

    if(!stream){
        throw std::runtime_error("Failed to read Qwen2 weights file: " + path.string());
    }

    return buffer;
}

void require_execution_weight(
    const Tensor& tensor,
    const Shape& expected,
    const char* name
){
    if(tensor.shape() != expected){
        throw std::invalid_argument(std::string(name) + " shape mismatch.");
    }

    if(
        (tensor.dtype() != DataType::Float32 &&
         tensor.dtype() != DataType::BFloat16) ||
        !tensor.is_contiguous()
    ){
        throw std::invalid_argument(
            std::string(name) + " must be contiguous float32 or bfloat16."
        );
    }
}

void validate_execution_model(const Qwen2Model& model){
    if(
        model.config.vocab_size == 0 || model.config.num_layers == 0 ||
        model.config.max_position_embeddings == 0
    ){
        throw std::invalid_argument("Qwen2 model dimensions must be positive.");
    }

    if(model.layers.size() != model.config.num_layers){
        throw std::invalid_argument("Qwen2 model layer count mismatch.");
    }

    require_execution_weight(
        model.token_embedding,
        {model.config.vocab_size, model.config.block.hidden_size},
        "Qwen2 token embedding"
    );

    require_execution_weight(
        model.final_norm,
        {model.config.block.hidden_size},
        "Qwen2 final norm"
    );

    for(const DecoderBlockWeights& layer : model.layers){
        validate_decoder_block(model.config.block, layer);
    }
}

}

Qwen2Model load_qwen2_model(const std::filesystem::path& manifest_path){
    static_assert(sizeof(float) == 4);
    static_assert(std::numeric_limits<float>::is_iec559);

    if constexpr(std::endian::native != std::endian::little){
        throw std::runtime_error("Qwen2 packages require a little-endian platform.");
    }

    QwenManifest manifest = parse_manifest(manifest_path);
    const auto expected = expected_tensors(manifest.config);
    const size_t expected_count = expected.size();
    if(manifest.tensors.size() < expected_count){
        throw std::runtime_error("Qwen2 package is missing tensor declarations.");
    }

    if(manifest.tensors.size() > expected_count){
        throw std::runtime_error("Qwen2 package contains unexpected tensors.");
    }

    std::unordered_map<std::string, const TensorMetadata*> declared;

    for(const TensorMetadata& tensor : manifest.tensors){
        declared.emplace(tensor.name, &tensor);
    }

    for(const auto& [name, shape] : expected){
        const auto entry = declared.find(name);
        if(entry == declared.end()){
            throw std::runtime_error("Qwen2 package is missing tensor: " + name);
        }
        if(entry->second->shape != shape){
            throw std::runtime_error("Qwen2 tensor shape mismatch: " + name);
        }
    }

    if(declared.size() != expected.size()){
        throw std::runtime_error("Qwen2 package contains unexpected tensors.");
    }

    std::vector<const TensorMetadata*> ordered;
    ordered.reserve(manifest.tensors.size());

    for(const TensorMetadata& tensor : manifest.tensors){
        ordered.push_back(&tensor);
    }

    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const TensorMetadata* left, const TensorMetadata* right){
            return left->offset_bytes < right->offset_bytes;
        }
    );

    uint64_t end = 0;
    for(const TensorMetadata* tensor : ordered){
        if(tensor->offset_bytes < end){
            throw std::runtime_error("Qwen2 tensor ranges overlap: " + tensor->name);
        }
        if(tensor->byte_size > std::numeric_limits<uint64_t>::max() - tensor->offset_bytes){
            throw std::runtime_error("Qwen2 tensor range overflows: " + tensor->name);
        }
        end = tensor->offset_bytes + tensor->byte_size;
    }

    const std::filesystem::path weights_path =
        manifest_path.parent_path() / manifest.weights_file;
    std::shared_ptr<Buffer> buffer = load_weight_buffer(weights_path, end);

    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(manifest.tensors.size());

    for(const TensorMetadata& metadata : manifest.tensors){
        tensors.emplace(
            metadata.name,
            Tensor(
                buffer,
                static_cast<size_t>(metadata.offset_bytes),
                metadata.shape,
                metadata.dtype
            )
        );
    }

    auto tensor = [&](const std::string& name){ return tensors.at(name); };

    std::vector<DecoderBlockWeights> layers;
    layers.reserve(manifest.config.num_layers);
    for(size_t layer = 0; layer < manifest.config.num_layers; layer++){
        DecoderBlockWeights weights{
            tensor(layer_name(layer, "input_layernorm.weight")),
            tensor(layer_name(layer, "self_attn.q_proj.weight")),
            tensor(layer_name(layer, "self_attn.q_proj.bias")),
            tensor(layer_name(layer, "self_attn.k_proj.weight")),
            tensor(layer_name(layer, "self_attn.k_proj.bias")),
            tensor(layer_name(layer, "self_attn.v_proj.weight")),
            tensor(layer_name(layer, "self_attn.v_proj.bias")),
            tensor(layer_name(layer, "self_attn.o_proj.weight")),
            tensor(layer_name(layer, "post_attention_layernorm.weight")),
            tensor(layer_name(layer, "mlp.gate_proj.weight")),
            tensor(layer_name(layer, "mlp.up_proj.weight")),
            tensor(layer_name(layer, "mlp.down_proj.weight"))
        };

        layers.push_back(std::move(weights));
    }

    return Qwen2Model{
        std::move(manifest.model_id),
        std::move(manifest.revision),
        std::move(manifest.source_sha256),
        manifest.config,
        tensor("model.embed_tokens.weight"),
        std::move(layers),
        tensor("model.norm.weight")
    };
}

Qwen2Decoder::Qwen2Decoder(Qwen2Model model, size_t max_tokens)
    : model_(std::move(model)), capacity_tokens_(max_tokens){
    validate_execution_model(model_);
    if(capacity_tokens_ == 0){
        throw std::invalid_argument("Qwen2 decoder capacity must be positive.");
    }

    if(capacity_tokens_ > model_.config.max_position_embeddings){
        throw std::invalid_argument("Qwen2 decoder capacity exceeds the model limit.");
    }

    const size_t head_dim = model_.config.block.hidden_size / model_.config.block.query_heads;
    caches_.reserve(model_.config.num_layers);

    for(size_t layer = 0; layer < model_.config.num_layers; layer++){
        caches_.emplace_back(KvCacheConfig{
            capacity_tokens_,
            model_.config.block.kv_heads,
            head_dim
        });
    }
}

Tensor Qwen2Decoder::last_token_logits(const Tensor& hidden_states) const{
    const size_t hidden_size = model_.config.block.hidden_size;
    Tensor last_hidden({1, hidden_size});

    // Copy the last hidden state from the batch of hidden states 
    // to a new tensor for normalization and projection.
    std::copy_n(
        hidden_states.data() + (hidden_states.shape()[0] - 1) * hidden_size,
        hidden_size,
        last_hidden.data()
    );

    const Tensor normalized = rms_norm(
        last_hidden,
        model_.final_norm,
        model_.config.block.rms_norm_epsilon
    );

    // Tied LM head: project the normalized hidden state to the 
    // vocabulary size using the token embedding weights
    return matmul_transposed_b(normalized, model_.token_embedding);
}

Tensor Qwen2Decoder::prefill(std::span<const int64_t> token_ids){
    if(token_ids.empty()){
        throw std::invalid_argument("Qwen2 prefill requires at least one token.");
    }

    if(token_ids.size() > capacity_tokens_){
        throw std::length_error("Qwen2 prefill exceeds decoder capacity.");
    }

    Tensor hidden_states = embedding(model_.token_embedding, token_ids);
    for(size_t layer = 0; layer < model_.layers.size(); layer++){
        hidden_states = decoder_block_prefill(
            hidden_states,
            model_.config.block,
            model_.layers[layer],
            caches_[layer]
        );
    }

    return last_token_logits(hidden_states);
}

Tensor Qwen2Decoder::decode(int64_t token_id){
    const size_t position = size_tokens();
    if(position == 0){
        throw std::logic_error("Qwen2 decode requires prefill first.");
    }

    if(position == capacity_tokens_){
        throw std::length_error("Qwen2 decode exceeds decoder capacity.");
    }

    const int64_t token_ids[]{token_id};
    Tensor hidden_state = embedding(model_.token_embedding, token_ids);

    // Process the single token through each decoder block, updating the KV caches
    for(size_t layer = 0; layer < model_.layers.size(); layer++){
        hidden_state = decoder_block_decode(
            hidden_state,
            model_.config.block,
            model_.layers[layer],
            caches_[layer]
        );
    }

    return last_token_logits(hidden_state);
}

void Qwen2Decoder::reset() noexcept{
    for(ContiguousKvCache& cache : caches_){
        cache.clear();
    }
}

size_t Qwen2Decoder::size_tokens() const noexcept{
    return caches_.front().size_tokens();
}

size_t Qwen2Decoder::capacity_tokens() const noexcept{
    return capacity_tokens_;
}

}

#include "model_loader.h"

#include "operator_registry.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <system_error>

namespace tinyinfer{

namespace{

[[noreturn]] void throw_parse_error(
    const std::filesystem::path& path,
    size_t line_number,
    const std::string& message
){
    throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + message);
}

std::vector<std::string> split_tokens(std::string line){
    const size_t comment = line.find('#');

    if(comment != std::string::npos){
        line.erase(comment);
    }

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;

    while(iss >> token){
        tokens.push_back(std::move(token));
    }

    return tokens;
}

uint64_t parse_u64(
    const std::string& token,
    const std::filesystem::path& path,
    size_t line_number,
    const std::string& field
){
    if(token.empty() || token.front() == '-'){
        throw_parse_error(path, line_number, field + " must be a non-neg int");
    }

    uint64_t value = 0;

    const char* begin = token.data();
    const char* end = begin + token.size();

    auto [ptr, errc] = std::from_chars(begin, end, value);

    if(errc != std::errc{} || ptr != end){
        throw_parse_error(path, line_number, "invalid " + field + ": " + token);
    }

    return value;
}

uint64_t checked_numel(
    const Shape& shape,
    const std::filesystem::path& path,
    size_t line_number
){
    uint64_t total = 1;

    for(size_t dim : shape){
        if(dim != 0 && total > std::numeric_limits<uint64_t>::max() / dim){
            throw_parse_error(path, line_number, "Tensor element count overflows");
        }

        total *= dim;
    }

    return total;
}

Shape parse_shape(
    const std::vector<std::string>& tokens,
    size_t first_dim,
    size_t rank,
    const std::filesystem::path& path,
    size_t line_number
){
    Shape shape;
    shape.reserve(rank);

    for(size_t i = 0; i < rank; i++){
        const uint64_t dim = parse_u64(
            tokens[first_dim + i],
            path,
            line_number,
            "shape dimension"
        );

        if(dim > std::numeric_limits<size_t>::max()){
            throw_parse_error(
                path,
                line_number,
                "shape dimension exceeds size_t"
            );
        }

        shape.push_back(static_cast<size_t>(dim));
    }

    return shape;
}

Tensor load_tensor(
    std::ifstream& weights,
    uint64_t weights_size,
    const TensorMetadata& metadata,
    const std::filesystem::path& weights_path
){
    if(metadata.offset_bytes > weights_size || // Check if offset is out of bounds
       metadata.byte_size > weights_size - metadata.offset_bytes){ 
        // Check if byte size is out of bounds
        throw std::runtime_error(
            "Tensor '" + metadata.name + "' is out of bounds in weights file: " +
            weights_path.string()
        );
    }

    const uint64_t numel64 = metadata.byte_size / sizeof(float);
    
    if(numel64 > std::numeric_limits<size_t>::max()){
        throw std::runtime_error("Tensor '" + metadata.name + "' is too large for this platform");
    }

    if(metadata.offset_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())){
        throw std::runtime_error("Tensor '" + metadata.name + "' offset is too large for this platform");
    }

    std::vector<float> data(static_cast<size_t>(numel64));

    weights.clear(); // Clear any error flags
    weights.seekg(static_cast<std::streamoff>(metadata.offset_bytes), std::ios::beg);

    if(!weights){
        throw std::runtime_error(
            "Failed to seek to tensor '" + metadata.name + "' in weights file: " +
            weights_path.string()
        );
    }

    if(metadata.byte_size != 0){
        weights.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(metadata.byte_size));

        if(!weights){
            throw std::runtime_error(
                "Failed to read tensor '" + metadata.name + "' from weights file: " +
                weights_path.string()
            );
        }
    }

    return Tensor(metadata.shape, std::move(data));
}

void require_non_empty_name(const std::string& name, const std::string& field){
    if(name.empty()){
        throw std::runtime_error(field + " name cannot be empty");
    }
}

uint64_t tensor_byte_size(const Tensor& tensor){
    const uint64_t numel = static_cast<uint64_t>(tensor.numel());

    if(numel > std::numeric_limits<uint64_t>::max() / sizeof(float)){
        throw std::runtime_error("Tensor byte size overflows uint64");
    }

    return numel * sizeof(float);
}

void write_tensor_data(std::ofstream& file, const NamedTensor& named_tensor){
    const Tensor& tensor = named_tensor.tensor;
    const uint64_t byte_size = tensor_byte_size(tensor);

    if(byte_size == 0){
        return;
    }

    file.write(
        reinterpret_cast<const char*>(tensor.data()),
        static_cast<std::streamsize>(byte_size)
    );

    if(!file){
        throw std::runtime_error("Failed to write tensor '" + named_tensor.name + "' data");
    }
}

}

ModelMetadata ModelLoader::parse_manifest(const std::filesystem::path& manifest_path){
    std::ifstream file(manifest_path);

    if(!file){
        throw std::runtime_error("Cannot open model manifest: " + manifest_path.string());
    }

    ModelMetadata metadata;

    bool seen_header = false;
    bool seen_weights = false;
    bool seen_input = false;
    bool seen_output = false;
    bool seen_end = false;

    std::unordered_set<std::string> tensor_names;
    std::unordered_set<std::string> node_names;

    std::string line;
    size_t line_number = 0;

    while(std::getline(file, line)){
        line_number++;

        std::vector<std::string> tokens = split_tokens(std::move(line));

        if(tokens.empty()){
            continue;
        }

        if(seen_end){
            throw_parse_error(
                manifest_path,
                line_number,
                "content found after end"
            );
        }

        if(!seen_header){
            if(tokens.size() != 2 ||
               tokens[0] != "TINYINFER_MODEL"){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "expected 'TINYINFER_MODEL <version>'"
                );
            }

            const uint64_t version = parse_u64(
                tokens[1],
                manifest_path,
                line_number,
                "model version"
            );

            if(version != 1){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "unsupported model version: " + std::to_string(version)
                );
            }

            metadata.version = static_cast<uint32_t>(version);

            seen_header = true;
            continue;
        }

        const std::string& directive = tokens[0];

        if(directive == "weights"){
            if(seen_weights){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "duplicate weights directive"
                );
            }

            if(tokens.size() != 2){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "weights expects one file path"
                );
            }

            metadata.weights_file = tokens[1];
            seen_weights = true;
            continue;
        }

        if(directive == "input"){
            if(seen_input){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "duplicate input directive"
                );
            }

            if(tokens.size() < 3){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "input expects name, rank and dimensions"
                );
            }

            const uint64_t rank_value = parse_u64(
                tokens[2],
                manifest_path,
                line_number,
                "input rank"
            );

            if(rank_value >
               std::numeric_limits<size_t>::max()){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "input rank exceeds size_t"
                );
            }

            const size_t rank = static_cast<size_t>(rank_value);

            if(tokens.size() != 3 + rank){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "input dimension count does not match rank"
                );
            }

            metadata.input_name = tokens[1];
            metadata.input_shape = parse_shape(
                tokens,
                3,
                rank,
                manifest_path,
                line_number
            );

            seen_input = true;
            continue;
        }

        if(directive == "output"){
            if(seen_output){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "duplicate output directive"
                );
            }

            if(tokens.size() != 2){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "output expects one tensor name"
                );
            }

            metadata.output_name = tokens[1];
            seen_output = true;
            continue;
        }

        if(directive == "tensor"){
            if(tokens.size() < 6){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "invalid tensor declaration"
                );
            }

            const std::string& name = tokens[1];

            if(!tensor_names.insert(name).second){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "duplicate tensor name: " + name
                );
            }

            if(tokens[2] != "f32"){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "unsupported tensor dtype: " +
                    tokens[2]
                );
            }

            const uint64_t rank_value = parse_u64(
                tokens[3],
                manifest_path,
                line_number,
                "tensor rank"
            );

            if(rank_value >
               std::numeric_limits<size_t>::max()){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "tensor rank exceeds size_t"
                );
            }

            const size_t rank = static_cast<size_t>(rank_value);

            if(tokens.size() != 6 + rank){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "tensor dimension count does not match rank"
                );
            }

            Shape shape = parse_shape(
                tokens,
                4,
                rank,
                manifest_path,
                line_number
            );

            const uint64_t offset = parse_u64(
                tokens[4 + rank],
                manifest_path,
                line_number,
                "tensor offset"
            );

            const uint64_t byte_size = parse_u64(
                tokens[5 + rank],
                manifest_path,
                line_number,
                "tensor byte size"
            );

            const uint64_t numel = checked_numel(
                shape,
                manifest_path,
                line_number
            );

            if(numel >
               std::numeric_limits<uint64_t>::max() /
               sizeof(float)){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "tensor byte size overflows uint64"
                );
            }

            const uint64_t expected_bytes = numel * sizeof(float);

            if(byte_size != expected_bytes){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "tensor byte size does not match shape"
                );
            }

            if(offset % sizeof(float) != 0){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "tensor offset is not float aligned"
                );
            }

            metadata.tensors.push_back(
                TensorMetadata{
                    name,
                    TensorDataType::Float32,
                    std::move(shape),
                    offset,
                    byte_size
                }
            );

            continue;
        }

        if(directive == "node"){
            if(tokens.size() < 5){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "invalid node declaration"
                );
            }

            const std::string& name = tokens[1];

            if(!node_names.insert(name).second){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "duplicate node name: " + name
                );
            }

            const uint64_t input_count_value =
                parse_u64(
                    tokens[3],
                    manifest_path,
                    line_number,
                    "node input count"
                );

            if(input_count_value >
               std::numeric_limits<size_t>::max()){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "node input count exceeds size_t"
                );
            }

            const size_t input_count = static_cast<size_t>(input_count_value);

            if(tokens.size() !=
               5 + input_count){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "node input count does not match declaration"
                );
            }

            std::vector<std::string> inputs;
            inputs.reserve(input_count);

            for(size_t i = 0; i < input_count; i++){
                inputs.push_back(tokens[4 + i]);
            }

            metadata.nodes.push_back(
                NodeMetadata{
                    name,
                    tokens[2],
                    std::move(inputs),
                    tokens[4 + input_count]
                }
            );

            continue;
        }

        if(directive == "end"){
            if(tokens.size() != 1){
                throw_parse_error(
                    manifest_path,
                    line_number,
                    "end does not accept arguments"
                );
            }

            seen_end = true;
            continue;
        }

        throw_parse_error(
            manifest_path,
            line_number,
            "unknown directive: " + directive
        );
    }

    if(!seen_header){
        throw std::runtime_error("Model manifest has no header");
    }

    if(!seen_weights){
        throw std::runtime_error("Model manifest has no weights directive");
    }

    if(!seen_input){
        throw std::runtime_error("Model manifest has no input directive");
    }

    if(!seen_output){
        throw std::runtime_error("Model manifest has no output directive");
    }

    if(!seen_end){
        throw std::runtime_error("Model manifest has no end directive");
    }

    return metadata;
}

Tensor LoadedModel::run(const Tensor& input) const{
    ExecutionContext context;
    return graph_.run(plan_, context, input);
}

const Graph& LoadedModel::graph() const noexcept{
    return graph_;
}

const ExecutionPlan& LoadedModel::plan() const noexcept{
    return plan_;
}

const ModelMetadata& LoadedModel::metadata() const noexcept{
    return metadata_;
}

std::unique_ptr<LoadedModel> ModelLoader::load(const std::filesystem::path& manifest_path){
    static_assert(sizeof(float) == 4, "tinyinfer model format requires 32-bit float");

    static_assert(std::numeric_limits<float>::is_iec559, "tinyinfer model format requires IEEE-754 float");

    if constexpr(std::endian::native != std::endian::little){
        throw std::runtime_error("tinyinfer v1 weights require a little-endian platform");
    }

    auto model = std::unique_ptr<LoadedModel>(new LoadedModel());

    model->metadata_ = parse_manifest(manifest_path);

    const std::filesystem::path weights_path = manifest_path.parent_path() / model->metadata_.weights_file;

    std::error_code ec;
    const uint64_t weights_size = std::filesystem::file_size(weights_path, ec);

    if(ec){
        throw std::runtime_error("Cannot get weights file size: " + weights_path.string());
    }

    std::ifstream weights(weights_path, std::ios::binary);

    if(!weights){
        throw std::runtime_error(
            "Cannot open weights file: " +
            weights_path.string()
        );
    }

    for(const TensorMetadata& tensor :
        model->metadata_.tensors){
        model->graph_.set_tensor(
            tensor.name,
            load_tensor(
                weights,
                weights_size,
                tensor,
                weights_path
            )
        );
    }

    OperatorRegistry registry;

    for(const NodeMetadata& node :
        model->metadata_.nodes){
        const OpType type = registry.type_from_name(node.op_name);

        model->graph_.add_node(
            node.name,
            type,
            node.inputs,
            node.output
        );
    }

    model->plan_ = model->graph_.compile(
        model->metadata_.input_name,
        model->metadata_.input_shape,
        model->metadata_.output_name
    );

    return model;
}

void ModelWriter::save(
    const ModelPackage& package, 
    const std::filesystem::path& manifest_path, 
    const std::string& weights_filename
){
    require_non_empty_name(package.input_name, "input");
    require_non_empty_name(package.output_name, "output");

    if(weights_filename.empty()){
        throw std::runtime_error("weights filename cannot be empty");
    }

    const std::filesystem::path parent = manifest_path.parent_path();

    if(!parent.empty()){
        std::filesystem::create_directories(parent);
    }

    const std::filesystem::path weights_path = parent / weights_filename;

    std::unordered_set<std::string> tensor_names;
    std::unordered_set<std::string> node_names;

    for(const NamedTensor& tensor : package.tensors){ // Validate tensor metadata
        require_non_empty_name(tensor.name, "tensor");

        if(!tensor_names.insert(tensor.name).second){
            throw std::runtime_error("duplicate tensor name: " + tensor.name);
        }

        if(tensor.name == package.input_name){
            throw std::runtime_error("tensor name collides with input name: " + tensor.name);
        }
    }

    for(const NodeMetadata& node : package.nodes){ // Validate node metadata
        require_non_empty_name(node.name, "node");
        require_non_empty_name(node.op_name, "node op");
        require_non_empty_name(node.output, "node output");

        if(!node_names.insert(node.name).second){
            throw std::runtime_error("duplicate node name: " + node.name);
        }

        for(const std::string& input : node.inputs){
            require_non_empty_name(input, "node input");
        }
    }

    std::ofstream weights(weights_path, std::ios::binary);

    if(!weights){
        throw std::runtime_error("Cannot open weights file for writing: " + weights_path.string());
    }

    struct TensorWriteInfo{
        const NamedTensor* tensor;
        uint64_t offset_bytes;
        uint64_t byte_size;
    };

    std::vector<TensorWriteInfo> tensor_infos;
    tensor_infos.reserve(package.tensors.size());

    uint64_t offset = 0;

    for(const NamedTensor& tensor : package.tensors){
        const uint64_t byte_size = tensor_byte_size(tensor.tensor);

        tensor_infos.push_back(
            TensorWriteInfo{
                &tensor,
                offset,
                byte_size
            }
        );

        write_tensor_data(weights, tensor);

        if(byte_size > std::numeric_limits<uint64_t>::max() - offset){
            throw std::runtime_error("Tensor data exceeds uint64 max size");
        }

        offset += byte_size;
    }

    weights.close();

    if(!weights){
        throw std::runtime_error("Failed to write weights file: " + weights_path.string());
    }

    std::ofstream manifest(manifest_path);

    if(!manifest){
        throw std::runtime_error("Cannot open manifest file for writing: " + manifest_path.string());
    }

    manifest << "TINYINFER_MODEL 1\n";
    manifest << "weights " << weights_filename << "\n";
    manifest << "\n";

    manifest << "input " << package.input_name << " " << package.input_shape.size();

    for(size_t dim : package.input_shape){
        manifest << " " << dim;
    }

    manifest << "\n";
    manifest << "output " << package.output_name << "\n";
    manifest << "\n";

    for(const TensorWriteInfo& info : tensor_infos){
        const NamedTensor& tensor = *info.tensor;
        const Shape& shape = tensor.tensor.shape();

        manifest << "tensor " << tensor.name << " f32 " << shape.size();

        for(size_t dim : shape){
            manifest << " " << dim;
        }

        manifest << " " << info.offset_bytes << " " << info.byte_size << "\n";
    }

    manifest << "\n";

    for(const NodeMetadata& node : package.nodes){
        manifest << "node " << node.name << " " << node.op_name << " " << node.inputs.size();

        for(const std::string& input : node.inputs){
            manifest << " " << input;
        }

        manifest << " " << node.output << "\n";
    }

    manifest << "\n";
    manifest << "end\n";

    if(!manifest){
        throw std::runtime_error("failed to write model manifest: " + manifest_path.string());
    }
}

}
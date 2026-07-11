#pragma once

#include "graph.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tinyinfer{

enum class TensorDataType{
    Float32
};

struct TensorMetadata{
    std::string name;
    TensorDataType dtype;
    Shape shape;
    uint64_t offset_bytes;
    uint64_t byte_size;
};

struct NodeMetadata{
    std::string name;
    std::string op_name;
    std::vector<std::string> inputs;
    std::string output;
};

struct ModelMetadata{
    uint32_t version = 0;
    std::filesystem::path weights_file;
    
    std::string input_name;
    Shape input_shape;

    std::string output_name;

    std::vector<TensorMetadata> tensors;
    std::vector<NodeMetadata> nodes;
};

struct NamedTensor{
    std::string name;
    Tensor tensor;
};

struct ModelPackage{
    std::string input_name;
    Shape input_shape;
    std::string output_name;

    std::vector<NamedTensor> tensors;
    std::vector<NodeMetadata> nodes;
};

class LoadedModel{
public:
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&) = delete;
    LoadedModel& operator=(LoadedModel&&) = delete;

    Tensor run(const Tensor& input) const;

    const Graph& graph() const noexcept;
    const ExecutionPlan& plan() const noexcept;
    const ModelMetadata& metadata() const noexcept;

private:
    friend class ModelLoader;

    LoadedModel() = default;

    ModelMetadata metadata_;
    Graph graph_;
    ExecutionPlan plan_;
};

class ModelLoader{
public:
    static ModelMetadata parse_manifest(const std::filesystem::path& manifest_path);
    static std::unique_ptr<LoadedModel> load(const std::filesystem::path& manifest_path);
};

class ModelWriter{
public:
    static void save(
        const ModelPackage& package, 
        const std::filesystem::path& manifest_path, 
        const std::string& weights_filename = "weights.bin"
    );
};

}
#pragma once

#include "graph.h"

#include <cstdint>
#include <filesystem>
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
    uint64_t offset_bytes = 0;
    uint64_t byte_size = 0;
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

struct LoadedGraph{
    ModelMetadata metadata;
    Graph graph;
};

}

#pragma once

#include "tensor.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tinyinfer{

enum class OpType{
    Linear,
    ReLU,
    Softmax,
    Add
};

struct OpTypeHash{
    size_t operator()(OpType type) const noexcept{
        return static_cast<size_t>(type);
    }
};

using ShapeInputs = std::vector<const Shape*>;
using ShapeInferenceFunction = Shape (*)(const ShapeInputs&, std::string_view node_name);

struct OperatorSchema{
    std::string name;
    size_t input_count = 0;
    size_t output_count = 1;
    ShapeInferenceFunction infer_shape = nullptr;
};

class OperatorRegistry{
public:
    OperatorRegistry();

    void register_schema(OpType type, OperatorSchema schema);

    const OperatorSchema& get(OpType type) const;
    OpType type_from_name(std::string_view name) const;

private:
    std::unordered_map<OpType, OperatorSchema, OpTypeHash> schemas_;
    std::unordered_map<std::string, OpType> types_by_name_;
};

}

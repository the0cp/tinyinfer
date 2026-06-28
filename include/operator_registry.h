#pragma once

#include "tensor.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyinfer{

enum class OpType{
    Linear,
    ReLU,
    Softmax,
    Add
};

using TensorInputs = std::vector<const Tensor*>;

using ShapeInputs = std::vector<const Shape*>;

using ExecuteKernel = Tensor (*)(const TensorInputs& inputs);

using ShapeInferenceKernel = Shape (*)(
    const ShapeInputs& inputs,
    const std::string& node_name
);

struct OperatorDefinition{
    std::string name;
    size_t input_count;

    ExecuteKernel execute;
    ShapeInferenceKernel infer_shape;
};

struct OpTypeHash{
    size_t operator()(OpType type) const noexcept{
        return static_cast<size_t>(type);
    }
};

class OperatorRegistry{
public:
    OperatorRegistry();

    void register_operator(
        OpType type,
        OperatorDefinition definition
    );

    const OperatorDefinition& get(OpType type) const;

private:
    std::unordered_map<
        OpType,
        OperatorDefinition,
        OpTypeHash
    > definitions_;
};

}
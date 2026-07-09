#include "operator_registry.h"

#include "ops.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinyinfer{

namespace{

std::string shape_to_string(const Shape& shape){
    std::ostringstream oss;

    oss << "[";

    for(size_t i = 0; i < shape.size(); i++){
        if(i > 0){
            oss << ", ";
        }

        oss << shape[i];
    }

    oss << "]";
    return oss.str();
}

Tensor execute_linear(const TensorInputs& inputs){
    return linear(
        *inputs[0],
        *inputs[1],
        *inputs[2]
    );
}

Tensor execute_relu(const TensorInputs& inputs){
    return relu(*inputs[0]);
}

Tensor execute_softmax(const TensorInputs& inputs){
    return softmax(*inputs[0]);
}

Tensor execute_add(const TensorInputs& inputs){
    return add(
        *inputs[0],
        *inputs[1]
    );
}

Shape infer_linear_shape(
    const ShapeInputs& inputs,
    const std::string& node_name
){
    const Shape& x = *inputs[0];
    const Shape& weight = *inputs[1];
    const Shape& bias = *inputs[2];

    if(x.size() != 2){
        throw std::runtime_error(
            "Linear node '" + node_name +
            "' expects a 2D input, got " +
            shape_to_string(x)
        );
    }

    if(weight.size() != 2){
        throw std::runtime_error(
            "Linear node '" + node_name +
            "' expects a 2D weight, got " +
            shape_to_string(weight)
        );
    }

    if(bias.size() != 1){
        throw std::runtime_error(
            "Linear node '" + node_name +
            "' expects a 1D bias, got " +
            shape_to_string(bias)
        );
    }

    if(x[1] != weight[0]){
        throw std::runtime_error(
            "Linear node '" + node_name +
            "' cannot multiply input " +
            shape_to_string(x) +
            " by weight " +
            shape_to_string(weight)
        );
    }

    if(weight[1] != bias[0]){
        throw std::runtime_error(
            "Linear node '" + node_name +
            "' has weight output size " +
            std::to_string(weight[1]) +
            " but bias size " +
            std::to_string(bias[0])
        );
    }

    return {x[0], weight[1]};
}

Shape infer_relu_shape(
    const ShapeInputs& inputs,
    const std::string&
){
    return *inputs[0];
}

Shape infer_softmax_shape(
    const ShapeInputs& inputs,
    const std::string& node_name
){
    const Shape& x = *inputs[0];

    if(x.size() != 2){
        throw std::runtime_error(
            "Softmax node '" + node_name +
            "' expects a 2D input, got " +
            shape_to_string(x)
        );
    }

    if(x[1] == 0){
        throw std::runtime_error(
            "Softmax node '" + node_name +
            "' has an empty feature dimension"
        );
    }

    return x;
}

Shape infer_add_shape(
    const ShapeInputs& inputs,
    const std::string& node_name
){
    const Shape& a = *inputs[0];
    const Shape& b = *inputs[1];

    if(a != b){
        throw std::runtime_error(
            "Add node '" + node_name +
            "' has incompatible input shapes " +
            shape_to_string(a) +
            " and " +
            shape_to_string(b)
        );
    }

    return a;
}

}

OperatorRegistry::OperatorRegistry(){
    register_operator(
        OpType::Linear,
        OperatorDefinition{
            "Linear",
            3,
            execute_linear,
            infer_linear_shape
        }
    );

    register_operator(
        OpType::ReLU,
        OperatorDefinition{
            "ReLU",
            1,
            execute_relu,
            infer_relu_shape
        }
    );

    register_operator(
        OpType::Softmax,
        OperatorDefinition{
            "Softmax",
            1,
            execute_softmax,
            infer_softmax_shape
        }
    );

    register_operator(
        OpType::Add,
        OperatorDefinition{
            "Add",
            2,
            execute_add,
            infer_add_shape
        }
    );
}

void OperatorRegistry::register_operator(
    OpType type,
    OperatorDefinition definition
){
    if(definition.name.empty()){
        throw std::runtime_error(
            "Cannot register an operator with an empty name."
        );
    }

    if(definition.execute == nullptr){
        throw std::runtime_error(
            "Operator '" + definition.name +
            "' has no execution kernel"
        );
    }

    if(definition.infer_shape == nullptr){
        throw std::runtime_error(
            "Operator '" + definition.name +
            "' has no shape inference kernel"
        );
    }

    if(definitions_.contains(type)){
        throw std::runtime_error(
            "Operator type is already registered."
        );
    }

    if(types_by_name_.contains(definition.name)){
        throw std::runtime_error(
            "Operator name is already registered: " +
            definition.name
        );
    }

    std::string name = definition.name;

    definitions_.emplace(
        type,
        std::move(definition)
    );

    types_by_name_.emplace(
        std::move(name),
        type
    );
}

OpType OperatorRegistry::type_from_name(const std::string& name) const{
    auto it = types_by_name_.find(name);

    if(it == types_by_name_.end()){
        throw std::runtime_error("Unknown operator: " + name);
    }

    return it->second;
}

const OperatorDefinition& OperatorRegistry::get(OpType type) const{
    auto it = definitions_.find(type);

    if(it == definitions_.end()){
        throw std::runtime_error(
            "Operator type is not registered"
        );
    }

    return it->second;
}

}
#include "operator_registry.h"

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

Shape infer_linear_shape(const ShapeInputs& inputs, std::string_view node_name){
    const Shape& x = *inputs.at(0);
    const Shape& weight = *inputs.at(1);
    const Shape& bias = *inputs.at(2);

    if(x.size() != 2){
        throw std::runtime_error(
            "Linear node '" + std::string(node_name) + "' expects a 2D input, got " + shape_to_string(x)
        );
    }

    if(weight.size() != 2){
        throw std::runtime_error(
            "Linear node '" + std::string(node_name) + "' expects a 2D weight, got " + shape_to_string(weight)
        );
    }

    if(bias.size() != 1){
        throw std::runtime_error(
            "Linear node '" + std::string(node_name) + "' expects a 1D bias, got " + shape_to_string(bias)
        );
    }

    if(x[1] != weight[0]){
        throw std::runtime_error(
            "Linear node '" + std::string(node_name) + "' cannot multiply input " +
            shape_to_string(x) + " by weight " + shape_to_string(weight)
        );
    }

    if(weight[1] != bias[0]){
        throw std::runtime_error(
            "Linear node '" + std::string(node_name) + "' has weight output size " +
            std::to_string(weight[1]) + " but bias size " + std::to_string(bias[0])
        );
    }

    return {x[0], weight[1]};
}

Shape infer_relu_shape(const ShapeInputs& inputs, std::string_view){
    return *inputs.at(0);
}

Shape infer_softmax_shape(const ShapeInputs& inputs, std::string_view node_name){
    const Shape& x = *inputs.at(0);

    if(x.size() != 2){
        throw std::runtime_error(
            "Softmax node '" + std::string(node_name) + "' expects a 2D input, got " + shape_to_string(x)
        );
    }

    if(x[1] == 0){
        throw std::runtime_error(
            "Softmax node '" + std::string(node_name) + "' has an empty feature dimension"
        );
    }

    return x;
}

Shape infer_add_shape(const ShapeInputs& inputs, std::string_view node_name){
    const Shape& a = *inputs.at(0);
    const Shape& b = *inputs.at(1);

    if(a != b){
        throw std::runtime_error(
            "Add node '" + std::string(node_name) + "' has incompatible input shapes " +
            shape_to_string(a) + " and " + shape_to_string(b)
        );
    }

    return a;
}

}

OperatorRegistry::OperatorRegistry(){
    register_schema(OpType::Linear, OperatorSchema{"Linear", 3, 1, infer_linear_shape});
    register_schema(OpType::ReLU, OperatorSchema{"ReLU", 1, 1, infer_relu_shape});
    register_schema(OpType::Softmax, OperatorSchema{"Softmax", 1, 1, infer_softmax_shape});
    register_schema(OpType::Add, OperatorSchema{"Add", 2, 1, infer_add_shape});
}

void OperatorRegistry::register_schema(OpType type, OperatorSchema schema){
    if(schema.name.empty()){
        throw std::invalid_argument("Cannot register an operator with an empty name.");
    }

    if(schema.output_count != 1){
        throw std::invalid_argument("tinyinfer currently supports exactly one output per operator.");
    }

    if(schema.infer_shape == nullptr){
        throw std::invalid_argument("Operator '" + schema.name + "' has no shape inference function.");
    }

    if(schemas_.contains(type)){
        throw std::runtime_error("Operator type is already registered.");
    }

    if(types_by_name_.contains(schema.name)){
        throw std::runtime_error("Operator name is already registered: " + schema.name);
    }

    std::string name = schema.name;
    schemas_.emplace(type, std::move(schema));
    types_by_name_.emplace(std::move(name), type);
}

const OperatorSchema& OperatorRegistry::get(OpType type) const{
    auto it = schemas_.find(type);

    if(it == schemas_.end()){
        throw std::runtime_error("Operator type is not registered.");
    }

    return it->second;
}

OpType OperatorRegistry::type_from_name(std::string_view name) const{
    auto it = types_by_name_.find(std::string(name));

    if(it == types_by_name_.end()){
        throw std::runtime_error("Unknown operator: " + std::string(name));
    }

    return it->second;
}

}

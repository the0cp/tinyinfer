#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace tinyinfer{

enum class DataType : uint8_t{
    Float32
};

inline constexpr size_t element_size(DataType dtype){
    switch(dtype){
    case DataType::Float32:
        return sizeof(float);
    }

    throw std::invalid_argument("Unsupported tensor data type.");
}

inline constexpr size_t element_alignment(DataType dtype){
    switch(dtype){
        case DataType::Float32: return alignof(float);
    }

    throw std::invalid_argument("Unsupported tensor data type.");
}

inline constexpr std::string_view data_type_name(DataType dtype) noexcept{
    switch(dtype){
        case DataType::Float32: return "float32";
    }

    return "unknown";
}

}

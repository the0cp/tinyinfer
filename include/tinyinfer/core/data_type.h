#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace tinyinfer{

struct BFloat16{
    uint16_t bits = 0;

    static constexpr BFloat16 from_float(float value) noexcept{
        const uint32_t value_bits = std::bit_cast<uint32_t>(value);
        if((value_bits & 0x7fffffffU) > 0x7f800000U){
            return BFloat16{static_cast<uint16_t>((value_bits >> 16) | 0x0040U)};
        }
        const uint32_t rounding_bias = 0x7fffU + ((value_bits >> 16) & 1U);
        return BFloat16{static_cast<uint16_t>((value_bits + rounding_bias) >> 16)};
    }

    constexpr float to_float() const noexcept{
        return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }
};

static_assert(sizeof(BFloat16) == 2);

enum class DataType : uint8_t{
    Float32,
    BFloat16
};

inline constexpr size_t element_size(DataType dtype){
    switch(dtype){
        case DataType::Float32: return sizeof(float);
        case DataType::BFloat16: return sizeof(BFloat16);
    }

    throw std::invalid_argument("Unsupported tensor data type.");
}

inline constexpr size_t element_alignment(DataType dtype){
    switch(dtype){
        case DataType::Float32: return alignof(float);
        case DataType::BFloat16: return alignof(BFloat16);
    }

    throw std::invalid_argument("Unsupported tensor data type.");
}

inline constexpr std::string_view data_type_name(DataType dtype) noexcept{
    switch(dtype){
        case DataType::Float32: return "float32";
        case DataType::BFloat16: return "bfloat16";
    }

    return "unknown";
}

}

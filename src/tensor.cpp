#include "tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer{

namespace{

size_t checked_add(size_t a, size_t b, const char* message){
    if(b > std::numeric_limits<size_t>::max() - a){
        throw std::overflow_error(message);
    }

    return a + b;
}

size_t checked_mul(size_t a, size_t b, const char* message){
    if(a != 0 && b > std::numeric_limits<size_t>::max() / a){
        throw std::overflow_error(message);
    }

    return a * b;
}

bool has_zero_dimension(const Shape& shape) noexcept{
    return std::find(shape.begin(), shape.end(), 0) != shape.end();
}

}

size_t tensor_numel(const Shape& shape){
    if(has_zero_dimension(shape)){
        return 0;
    }

    size_t total = 1;

    for(size_t dimension : shape){
        total = checked_mul(total, dimension, "Tensor element count overflows size_t.");
    }

    return total;
}

size_t tensor_bytes(const Shape& shape, DataType dtype){
    return checked_mul(
        tensor_numel(shape),
        element_size(dtype),
        "Tensor byte size overflows size_t."
    );
}

Strides contiguous_strides(const Shape& shape){
    if(has_zero_dimension(shape)){
        // No element is addressable
        return Strides(shape.size(), 0);
    }

    Strides strides(shape.size());
    size_t stride = 1;

    for(size_t i = shape.size(); i-- > 0;){
        strides[i] = stride;
        stride = checked_mul(stride, shape[i], "Tensor stride overflows size_t.");
    }

    return strides;
}

Tensor::Tensor(Shape shape)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)),
      dtype_(DataType::Float32), buffer_(make_cpu_buffer(tensor_bytes(shape_, dtype_))){
    fill(0.0f);
}

Tensor::Tensor(Shape shape, std::vector<float> data)
    : shape_(std::move(shape)),
      strides_(contiguous_strides(shape_)),
      dtype_(DataType::Float32){
    const size_t expected_elements = tensor_numel(shape_);

    if(data.size() != expected_elements){
        throw std::invalid_argument("Tensor data size does not match the shape.");
    }

    buffer_ = make_cpu_buffer(tensor_bytes(shape_, dtype_));

    if(!data.empty()){
        std::memcpy(buffer_->data(), data.data(), size_bytes());
    }
}

Tensor::Tensor(std::shared_ptr<Buffer> buffer, size_t byte_offset, Shape shape, DataType dtype) 
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)),
      dtype_(dtype), buffer_(std::move(buffer)), byte_offset_(byte_offset){
    validate_layout();
}

Tensor::Tensor(std::shared_ptr<Buffer> buffer, size_t byte_offset, Shape shape, Strides strides, DataType dtype) 
    : shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype),
      buffer_(std::move(buffer)), byte_offset_(byte_offset){
    validate_layout();
}

void Tensor::validate_layout() const{
    if(!buffer_){
        throw std::invalid_argument("Tensor requires a non-null buffer.");
    }

    if(shape_.size() != strides_.size()){
        throw std::invalid_argument("Tensor shape and stride ranks must match.");
    }

    const size_t alignment = element_alignment(dtype_);

    if(byte_offset_ % alignment != 0){
        throw std::invalid_argument("Tensor byte offset is not aligned for its data type.");
    }

    const size_t required = required_span_bytes();

    if(byte_offset_ > buffer_->size_bytes() || required > buffer_->size_bytes() - byte_offset_){
        throw std::out_of_range("Tensor view exceeds its backing buffer.");
    }

    if(required != 0){
        const Buffer& const_buffer = *buffer_;
        const auto address = reinterpret_cast<std::uintptr_t>(const_buffer.data());

        if(address == 0 || address % alignment != 0){
            throw std::invalid_argument("Tensor buffer is not aligned for its data type.");
        }
    }
}

size_t Tensor::required_span_bytes() const{
    if(tensor_numel(shape_) == 0){
        return 0;
    }

    size_t max_element_offset = 0;

    for(size_t axis = 0; axis < shape_.size(); axis++){
        const size_t axis_span = checked_mul(
            shape_[axis] - 1,
            strides_[axis],
            "Tensor strided span overflows size_t."
        );
        max_element_offset = checked_add(
            max_element_offset,
            axis_span,
            "Tensor strided span overflows size_t."
        );
    }

    const size_t elements_in_span = checked_add(
        max_element_offset,
        1,
        "Tensor strided span overflows size_t."
    );

    return checked_mul(
        elements_in_span,
        element_size(dtype_),
        "Tensor strided byte span overflows size_t."
    );
}

size_t Tensor::compute_offset(std::initializer_list<size_t> indices) const{
    if(indices.size() != shape_.size()){
        throw std::invalid_argument("Tensor index dim mismatch.");
    }

    size_t offset = 0;
    size_t axis = 0;

    for(size_t index : indices){
        if(index >= shape_[axis]){
            throw std::out_of_range("Tensor index is out of range.");
        }

        offset += index * strides_[axis];
        axis++;
    }

    return offset;
}

size_t Tensor::linear_offset(size_t linear_index) const{
    if(linear_index >= numel()){
        throw std::out_of_range("Tensor linear index is out of range.");
    }

    size_t remaining = linear_index;
    size_t element_offset = 0;

    for(size_t axis = shape_.size(); axis-- > 0;){
        const size_t index = remaining % shape_[axis];
        remaining /= shape_[axis];
        element_offset += index * strides_[axis];
    }

    return element_offset;
}

std::byte* Tensor::byte_ptr(size_t element_offset){
    auto* bytes = static_cast<std::byte*>(buffer_->data());
    return bytes + byte_offset_ + element_offset * element_size(dtype_);
}

const std::byte* Tensor::byte_ptr(size_t element_offset) const{
    const Buffer& const_buffer = *buffer_;
    const auto* bytes = static_cast<const std::byte*>(const_buffer.data());
    return bytes + byte_offset_ + element_offset * element_size(dtype_);
}

float* Tensor::element_ptr(size_t element_offset){
    if(dtype_ != DataType::Float32){
        throw std::logic_error("Float access requires a float32 tensor.");
    }
    return reinterpret_cast<float*>(byte_ptr(element_offset));
}

const float* Tensor::element_ptr(size_t element_offset) const{
    if(dtype_ != DataType::Float32){
        throw std::logic_error("Float access requires a float32 tensor.");
    }
    return reinterpret_cast<const float*>(byte_ptr(element_offset));
}

float& Tensor::at(std::initializer_list<size_t> indices){
    return *element_ptr(compute_offset(indices));
}

const float& Tensor::at(std::initializer_list<size_t> indices) const{
    return *element_ptr(compute_offset(indices));
}

float* Tensor::data(){
    if(dtype_ != DataType::Float32){
        throw std::logic_error("Tensor::data() requires a float32 tensor.");
    }
    if(numel() == 0){
        return nullptr;
    }

    if(!is_contiguous()){
        throw std::logic_error("Tensor::data() requires a contiguous tensor.");
    }

    return element_ptr(0);
}

const float* Tensor::data() const{
    if(dtype_ != DataType::Float32){
        throw std::logic_error("Tensor::data() requires a float32 tensor.");
    }
    if(numel() == 0){
        return nullptr;
    }

    if(!is_contiguous()){
        throw std::logic_error("Tensor::data() requires a contiguous tensor.");
    }

    return element_ptr(0);
}

BFloat16* Tensor::bfloat16_data(){
    if(dtype_ != DataType::BFloat16){
        throw std::logic_error("Tensor::bfloat16_data() requires a bfloat16 tensor.");
    }
    if(numel() == 0){
        return nullptr;
    }
    if(!is_contiguous()){
        throw std::logic_error("Tensor::bfloat16_data() requires a contiguous tensor.");
    }
    return reinterpret_cast<BFloat16*>(byte_ptr(0));
}

const BFloat16* Tensor::bfloat16_data() const{
    if(dtype_ != DataType::BFloat16){
        throw std::logic_error("Tensor::bfloat16_data() requires a bfloat16 tensor.");
    }
    if(numel() == 0){
        return nullptr;
    }
    if(!is_contiguous()){
        throw std::logic_error("Tensor::bfloat16_data() requires a contiguous tensor.");
    }
    return reinterpret_cast<const BFloat16*>(byte_ptr(0));
}

const Shape& Tensor::shape() const noexcept{
    return shape_;
}

const Strides& Tensor::strides() const noexcept{
    return strides_;
}

DataType Tensor::dtype() const noexcept{
    return dtype_;
}

size_t Tensor::dim() const noexcept{
    return shape_.size();
}

size_t Tensor::numel() const{
    return tensor_numel(shape_);
}

size_t Tensor::size_bytes() const{
    return tensor_bytes(shape_, dtype_);
}

size_t Tensor::byte_offset() const noexcept{
    return byte_offset_;
}

bool Tensor::is_contiguous() const{
    if(numel() == 0){
        return true;
    }

    size_t expected_stride = 1;

    for(size_t axis = shape_.size(); axis-- > 0;){
        if(shape_[axis] > 1 && strides_[axis] != expected_stride){
            return false;
        }

        expected_stride = checked_mul(
            expected_stride,
            shape_[axis],
            "Tensor contiguous stride check overflows size_t."
        );
    }

    return true;
}

const std::shared_ptr<Buffer>& Tensor::buffer() const noexcept{
    return buffer_;
}

Tensor Tensor::reshape(Shape shape) const{
    if(!is_contiguous()){
        throw std::logic_error("Tensor::reshape() requires a contiguous tensor.");
    }

    if(tensor_numel(shape) != numel()){
        throw std::invalid_argument("Tensor reshape changes the element count.");
    }

    return Tensor(buffer_, byte_offset_, std::move(shape), dtype_);
}

Tensor Tensor::clone() const{
    Tensor copy(make_cpu_buffer(size_bytes()), 0, shape_, dtype_);
    const size_t count = numel();

    if(count == 0){
        return copy;
    }

    if(is_contiguous()){
        std::memcpy(copy.byte_ptr(0), byte_ptr(0), size_bytes());
        return copy;
    }

    const size_t item_size = element_size(dtype_);
    for(size_t linear = 0; linear < count; linear++){
        std::memcpy(
            copy.byte_ptr(linear),
            byte_ptr(linear_offset(linear)),
            item_size
        );
    }

    return copy;
}

void Tensor::fill(float value){
    const size_t count = numel();

    if(count == 0){
        return;
    }

    if(dtype_ == DataType::Float32){
        if(is_contiguous()){
            std::fill(data(), data() + count, value);
            return;
        }
        for(size_t linear = 0; linear < count; linear++){
            *element_ptr(linear_offset(linear)) = value;
        }
        return;
    }

    const BFloat16 converted = BFloat16::from_float(value);
    if(is_contiguous()){
        std::fill(bfloat16_data(), bfloat16_data() + count, converted);
        return;
    }
    for(size_t linear = 0; linear < count; linear++){
        std::memcpy(
            byte_ptr(linear_offset(linear)),
            &converted,
            sizeof(converted)
        );
    }
}

}

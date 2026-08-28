#pragma once

#include "buffer.h"
#include "data_type.h"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

namespace tinyinfer{

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

size_t tensor_numel(const Shape& shape);
size_t tensor_bytes(const Shape& shape, DataType dtype = DataType::Float32);
Strides contiguous_strides(const Shape& shape);

class Tensor{
public:
    Tensor() = delete;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    explicit Tensor(Shape shape);
    Tensor(Shape shape, std::vector<float> data);

    Tensor(
        std::shared_ptr<Buffer> buffer,
        size_t byte_offset,
        Shape shape,
        DataType dtype = DataType::Float32
    );

    Tensor(
        std::shared_ptr<Buffer> buffer,
        size_t byte_offset,
        Shape shape,
        Strides strides,
        DataType dtype = DataType::Float32
    );

    float& at(std::initializer_list<size_t> indices);
    const float& at(std::initializer_list<size_t> indices) const;

    float* data();
    const float* data() const;

    const Shape& shape() const noexcept;
    const Strides& strides() const noexcept;
    DataType dtype() const noexcept;

    size_t dim() const noexcept;
    size_t numel() const;
    size_t size_bytes() const;
    size_t byte_offset() const noexcept;

    bool is_contiguous() const;
    const std::shared_ptr<Buffer>& buffer() const noexcept;

    Tensor reshape(Shape shape) const;
    Tensor clone() const;
    void fill(float value);

private:
    Shape shape_;
    Strides strides_;
    DataType dtype_ = DataType::Float32;
    std::shared_ptr<Buffer> buffer_;
    size_t byte_offset_ = 0;

    size_t compute_offset(std::initializer_list<size_t> indices) const;
    size_t linear_offset(size_t linear_index) const;
    size_t required_span_bytes() const;
    void validate_layout() const;
    float* element_ptr(size_t element_offset);
    const float* element_ptr(size_t element_offset) const;
};

}

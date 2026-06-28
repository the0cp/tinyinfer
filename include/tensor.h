#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace tinyinfer{

using Shape = std::vector<size_t>;

class Tensor{
public:
    Tensor() = delete;
    explicit Tensor(Shape shape);
    Tensor(Shape shape, std::vector<float> data);

    float& at(std::initializer_list<size_t> indices);
    const float& at(std::initializer_list<size_t> indices) const;

    float* data();
    const float* data() const;

    const Shape& shape() const;
    const std::vector<size_t>& strides() const;

    size_t dim() const;
    size_t numel() const;

    void fill(float value);

private:
    Shape shape_;
    std::vector<size_t> strides_;
    std::vector<float> data_;

    void compute_strides();
    size_t compute_offset(std::initializer_list<size_t> indices) const;
};

}
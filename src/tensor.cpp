#include "tensor.h"

#include <algorithm>
#include <limits>

namespace tinyinfer{

namespace{

size_t checked_numel(const Shape& shape){
    size_t total = 1;

    for(size_t dim : shape){
        if(dim != 0 && total > std::numeric_limits<size_t>::max() / dim){
            throw std::runtime_error("Tensor element count overflows size_t.");
        }

        total *= dim;
    }

    return total;
}

}

Tensor::Tensor(Shape shape)
    : shape_(std::move(shape)){
    compute_strides();
    data_.resize(numel(), 0.0f);
}

Tensor::Tensor(Shape shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)){
    compute_strides();

    if(data_.size() != numel()){
        throw std::runtime_error("Tensor data size does not match the shape.");
    }
}

void Tensor::compute_strides(){
    strides_.resize(shape_.size());
    if(shape_.empty()){
        return;
    }

    size_t stride = 1;
    for(size_t i = shape_.size(); i-- > 0;){
        strides_[i] = stride;

        const size_t dim = shape_[i];

        if(dim != 0 && stride > std::numeric_limits<size_t>::max() / dim){
            throw std::runtime_error("Tensor stride overflows size_t.");
        }

        stride *= dim;
    }
}

size_t Tensor::compute_offset(std::initializer_list<size_t> indices) const{
    if(indices.size() != shape_.size()){
        throw std::runtime_error("Index dim mismatch.");
    }

    size_t offset = 0;
    
    size_t i = 0;
    for(size_t index : indices){
        if(index >= shape_[i]){
            throw std::runtime_error("Tensor index out of range.");
        }

        offset += index * strides_[i];
        i++;
    }
    
    return offset;
}

float& Tensor::at(std::initializer_list<size_t> indices){
    return data_[compute_offset(indices)];
}

const float& Tensor::at(std::initializer_list<size_t> indices) const{
    return data_[compute_offset(indices)];
}

const Shape& Tensor::shape() const{
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const{
    return strides_;
}

float* Tensor::data(){
    return data_.data();
}

const float* Tensor::data() const{
    return data_.data();
}

size_t Tensor::dim() const{
    return shape_.size();
}

size_t Tensor::numel() const{
    return checked_numel(shape_);
}

void Tensor::fill(float value){
    std::fill(data_.begin(), data_.end(), value);
}

}
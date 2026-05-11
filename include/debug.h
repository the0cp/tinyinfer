#pragma once

#include "tensor.h"

#include <iostream>
#include <stdexcept>

namespace tinyinfer{

inline void print_tensor_2d(const Tensor& t){
    if(t.dim() != 2){
        throw std::runtime_error("print_tensor_2d excepts a 2D tensor.");
    }

    const size_t rows = t.shape()[0];
    const size_t cols = t.shape()[1];
    
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            std::cout << t.at({i, j}) << " ";
        }
        std::cout << "\n";
    }
}    

}
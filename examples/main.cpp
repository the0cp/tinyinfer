#include "ops.h"

#include <iostream>

int main(){
    tinyinfer::Tensor input({1, 3}, {-1.0f, 2.0f, 3.0f});
    tinyinfer::Tensor output = tinyinfer::relu(input);

    std::cout << "ReLU output:";
    for(size_t i = 0; i < output.numel(); i++){
        std::cout << " " << output.data()[i];
    }
    std::cout << "\n";
    return 0;
}

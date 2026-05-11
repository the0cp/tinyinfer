#include <iostream>

#include "ops.h"

int main(){
    tinyinfer::Tensor a({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    tinyinfer::Tensor b({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });

    tinyinfer::Tensor c = tinyinfer::naive_matmul(a, b);
    tinyinfer::Tensor d = tinyinfer::relu(c);

    std::cout << d.at({0, 0}) << " " << d.at({0, 1}) << std::endl;
    std::cout << d.at({1, 0}) << " " << d.at({1, 1}) << std::endl;

    return 0;
}
#include "module.h"

#include <iostream>

using namespace tinyinfer;

int main() {
    Tensor x({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor w1({3, 4}, {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    });

    Tensor b1({4}, {
        0.1f, 0.1f, 0.1f, 0.1f
    });

    Tensor w2({4, 2}, {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f,
        0.7f, 0.8f
    });

    Tensor b2({2}, {
        0.5f, 0.5f
    });

    Sequential model;

    model.add(std::make_unique<Linear>(std::move(w1), std::move(b1)));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(std::move(w2), std::move(b2)));
    model.add(std::make_unique<Softmax>());

    Tensor y = model.forward(x);

    std::cout << y.at({0, 0}) << " " << y.at({0, 1}) << "\n";
    std::cout << y.at({1, 0}) << " " << y.at({1, 1}) << "\n";

    return 0;
}
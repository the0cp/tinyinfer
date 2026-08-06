#include "module.h"

#include <iostream>
#include <memory>

int main(){
    using namespace tinyinfer;

    Sequential model;
    model.add(std::make_unique<Linear>(
        Tensor({2, 2}, {1.0f, -1.0f, 1.0f, 1.0f}),
        Tensor({2}, {0.0f, 0.0f})
    ));
    model.add(std::make_unique<ReLU>());

    Tensor output = model.forward(Tensor({1, 2}, {1.0f, 2.0f}));
    std::cout << "Legacy eager MLP output: "
              << output.data()[0] << ", " << output.data()[1] << "\n";
    return 0;
}

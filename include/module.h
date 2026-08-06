#pragma once

#include "ops.h"
#include "tensor.h"

#include <memory>
#include <utility>
#include <vector>

namespace tinyinfer{

// Legacy eager teaching API. It is intentionally not part of the compiled runtime path.
class Module{
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
};

class Linear : public Module{
public:
    Linear(Tensor weight, Tensor bias)
        : weight_(std::move(weight)), bias_(std::move(bias)){}

    Tensor forward(const Tensor& input) override{
        return linear(input, weight_, bias_);
    }

private:
    Tensor weight_;
    Tensor bias_;
};

class ReLU : public Module{
public:
    Tensor forward(const Tensor& input) override{
        return relu(input);
    }
};

class Softmax : public Module{
public:
    Tensor forward(const Tensor& input) override{
        return softmax(input);
    }
};

class Sequential{
public:
    void add(std::unique_ptr<Module> module){
        modules_.push_back(std::move(module));
    }

    Tensor forward(const Tensor& input){
        Tensor value = input;

        for(auto& module : modules_){
            value = module->forward(value);
        }

        return value;
    }

    size_t size() const noexcept{
        return modules_.size();
    }

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

}

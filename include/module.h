#pragma once

#include "ops.h"
#include "tensor.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tinyinfer{

class Module{
public:
    virtual ~Module() = default;
    
    virtual Tensor forward(const Tensor& input) = 0;
};

class Linear : public Module{
public:
    Linear(Tensor weight, Tensor bias)
        : weight_(std::move(weight)), bias_(std::move(bias)) {}

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
        Tensor x = input;

        for(auto& module : modules_){
            x = module->forward(x);
        }

        return x;
    }

    size_t size() const{
        return modules_.size();
    }

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

}
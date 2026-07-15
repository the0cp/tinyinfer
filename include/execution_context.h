#pragma once

#include "tensor.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace tinyinfer{

class Graph;

class ExecutionContext{
public:
    ExecutionContext() = default;
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    ExecutionContext(ExecutionContext&&) = delete;
    ExecutionContext& operator=(ExecutionContext&&) = delete;

    void clear();

    bool has_tensor(const std::string& name) const;
    const Tensor& tensor(const std::string& name) const;

    std::string dump_tensors() const;

private:
    friend class Graph;

    void set_tensor(std::string name, Tensor tensor);
    void erase_tensor(const std::string& name);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Tensor> workspace_;
};

}

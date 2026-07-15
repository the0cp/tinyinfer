#include "execution_context.h"

#include <mutex>
#include <sstream>
#include <stdexcept>

namespace tinyinfer{

namespace{

std::string shape_to_string(const Shape& shape){
    std::ostringstream oss;

    oss << "[";

    for(size_t i = 0; i < shape.size(); i++){
        if(i > 0){
            oss << ", ";
        }

        oss << shape[i];
    }

    oss << "]";

    return oss.str();
}

}

void ExecutionContext::clear(){
    std::lock_guard<std::mutex> lock(mutex_);
    workspace_.clear();
}

bool ExecutionContext::has_tensor(const std::string& name) const{
    std::lock_guard<std::mutex> lock(mutex_);
    return workspace_.contains(name);
}

const Tensor& ExecutionContext::tensor(const std::string& name) const{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = workspace_.find(name);

    if(it == workspace_.end()){
        throw std::runtime_error("ExecutionContext tensor not found: " + name);
    }

    return it->second;
}

std::string ExecutionContext::dump_tensors() const{
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss << "ExecutionContext tensors:\n";

    if(workspace_.empty()){
        oss << "  <empty>\n";
        return oss.str();
    }

    for(const auto& [name, tensor] : workspace_){
        oss << "  " << name
            << ": shape=" << shape_to_string(tensor.shape())
            << ", numel=" << tensor.numel() << "\n";
    }

    return oss.str();
}

void ExecutionContext::set_tensor(std::string name, Tensor tensor){
    std::lock_guard<std::mutex> lock(mutex_);
    workspace_.insert_or_assign(std::move(name), std::move(tensor));
}

void ExecutionContext::erase_tensor(const std::string& name){
    std::lock_guard<std::mutex> lock(mutex_);
    workspace_.erase(name);
}

}

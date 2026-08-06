#include "value_index.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace tinyinfer{

ValueIndex ValueNameIndexMap::add(std::string name){
    if(name.empty()){
        throw std::invalid_argument("Cannot index an empty value name.");
    }

    if(name_to_index_.contains(name)){
        throw std::runtime_error("Value name is already indexed: " + name);
    }

    if(index_to_name_.size() >= static_cast<size_t>(invalid_value_index)){
        throw std::overflow_error("Value index exceeds uint32_t.");
    }

    const ValueIndex index = static_cast<ValueIndex>(index_to_name_.size());
    index_to_name_.push_back(std::move(name));
    name_to_index_.emplace(index_to_name_.back(), index);
    return index;
}

bool ValueNameIndexMap::contains(std::string_view name) const{
    return name_to_index_.find(std::string(name)) != name_to_index_.end();
}

ValueIndex ValueNameIndexMap::get(std::string_view name) const{
    auto it = name_to_index_.find(std::string(name));

    if(it == name_to_index_.end()){
        throw std::runtime_error("Unknown value name: " + std::string(name));
    }

    return it->second;
}

std::string_view ValueNameIndexMap::name(ValueIndex index) const{
    if(static_cast<size_t>(index) >= index_to_name_.size()){
        throw std::out_of_range("ValueIndex is out of range.");
    }

    return index_to_name_[index];
}

size_t ValueNameIndexMap::size() const noexcept{
    return index_to_name_.size();
}

}

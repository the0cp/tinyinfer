#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tinyinfer{

using ValueIndex = uint32_t;
using NodeIndex = uint32_t;

inline constexpr ValueIndex invalid_value_index = std::numeric_limits<ValueIndex>::max();
inline constexpr NodeIndex invalid_node_index = std::numeric_limits<NodeIndex>::max();

class ValueNameIndexMap{  // Maps value names to their corresponding indices in the execution plan
public:
    ValueIndex add(std::string name);

    bool contains(std::string_view name) const;
    ValueIndex get(std::string_view name) const;
    std::string_view name(ValueIndex index) const;
    size_t size() const noexcept;

private:
    std::unordered_map<std::string, ValueIndex> name_to_index_;
    std::vector<std::string> index_to_name_;
};

}

#pragma once

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tinyinfer{

enum class KernelThreading{
    Serial,
    IntraOp
};

struct KernelSelectionContext{
    std::vector<Shape> input_shapes;
    Shape output_shape;
    size_t effective_intra_op_threads = 1;
};

struct KernelMatch{
    bool supported = false;
    uint64_t estimated_cost = 0;
    std::string rejection;

    static KernelMatch accept(uint64_t estimated_cost) noexcept{
        return KernelMatch{true, estimated_cost, {}};
    }

    static KernelMatch reject(std::string reason){
        return KernelMatch{false, 0, std::move(reason)};
    }
};

struct KernelSelectionRecord{
    std::string kernel_name;
    KernelThreading threading = KernelThreading::Serial;
    uint64_t estimated_cost = 0;
    std::string reason;
};

inline constexpr const char* kernel_threading_name(KernelThreading threading) noexcept{
    switch(threading){
        case KernelThreading::Serial: return "serial";
        case KernelThreading::IntraOp: return "intra_op";
    }
    return "unknown";
}

}

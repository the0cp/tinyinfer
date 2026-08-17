#pragma once

#include "session_state.h"

#include <optional>
#include <string>
#include <vector>

namespace tinyinfer{

class ExecutionFrame{
public:
    explicit ExecutionFrame(const SessionState& session_state);

    ExecutionFrame(const ExecutionFrame&) = delete;
    ExecutionFrame& operator=(const ExecutionFrame&) = delete;
    ExecutionFrame(ExecutionFrame&&) = delete;
    ExecutionFrame& operator=(ExecutionFrame&&) = delete;

    void bind_input(ValueIndex index, const Tensor& tensor);  
    // Bind an input tensor to the execution frame.

    bool has_value(ValueIndex index) const;
    const Tensor& value(ValueIndex index) const;
    void set_value(ValueIndex index, Tensor tensor);
    void release(ValueIndex index);

    Tensor fetch(ValueIndex index) const;
    std::string dump_values() const;

private:
    struct ValueSlot{
        const Tensor* borrowed = nullptr;   // Pointer to a tensor that is borrowed from outside the frame
        std::optional<Tensor> owned;  // Optional tensor that is owned by the execution frame

        bool has_value() const noexcept{
            return borrowed != nullptr || owned.has_value();
        }
    };

    const SessionState& session_state_;
    std::vector<ValueSlot> values_;
};

}

#pragma once

#include "execution_frame.h"
#include "session_state.h"
#include "thread_pool.h"

namespace tinyinfer{

class IExecutor{
public:
    virtual ~IExecutor() = default;

    virtual void execute(
        const SessionState& session_state,
        ExecutionFrame& frame
    ) const = 0;
};

class SequentialExecutor final : public IExecutor{
public:
    void execute(
        const SessionState& session_state,
        ExecutionFrame& frame
    ) const override;
};

class ParallelExecutor final : public IExecutor{
public:
    explicit ParallelExecutor(ThreadPool& pool);

    void execute(
        const SessionState& session_state,
        ExecutionFrame& frame
    ) const override;

private:
    ThreadPool& pool_;
};

}

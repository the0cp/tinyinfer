#pragma once

#include "execution_frame.h"

namespace tinyinfer{

using ExecutionContext [[deprecated(
    "ExecutionContext was renamed to ExecutionFrame; runtime execution now goes through InferenceSession."
)]] = ExecutionFrame;

}

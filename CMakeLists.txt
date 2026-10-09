cmake_minimum_required(VERSION 3.16)

project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TINYINFER_ENABLE_ASAN_UBSAN "Build tinyinfer with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(TINYINFER_ENABLE_TSAN "Build tinyinfer with ThreadSanitizer" OFF)

if(TINYINFER_ENABLE_ASAN_UBSAN AND TINYINFER_ENABLE_TSAN)
    message(FATAL_ERROR "ASan/UBSan and TSan must be built separately.")
endif()

if(TINYINFER_ENABLE_ASAN_UBSAN)
    if(MSVC)
        message(FATAL_ERROR "This ASan/UBSan configuration is not supported on MSVC.")
    endif()
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

if(TINYINFER_ENABLE_TSAN)
    if(MSVC)
        message(FATAL_ERROR "ThreadSanitizer is not supported by this CMake configuration on MSVC.")
    endif()
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=thread)
endif()

find_package(Threads REQUIRED)

add_library(tinyinfer
    src/buffer.cpp
    src/tensor.cpp
    src/kv_cache.cpp
    src/attention.cpp
    src/ops.cpp
    src/thread_pool.cpp
    src/threading_policy.cpp
    src/value_index.cpp
    src/operator_registry.cpp
    src/graph.cpp
    src/graph_optimizer.cpp
    src/execution_plan.cpp
    src/memory_plan.cpp
    src/arena.cpp
    src/session_state.cpp
    src/profiler.cpp
    src/execution_frame.cpp
    src/op_kernel.cpp
    src/kernel_registry.cpp
    src/builtin_kernels.cpp
    src/sequential_executor.cpp
    src/parallel_executor.cpp
    src/inference_session.cpp
    src/model_loader.cpp
)

target_include_directories(tinyinfer PUBLIC include)
target_link_libraries(tinyinfer PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(tinyinfer PRIVATE /W4)
else()
    target_compile_options(tinyinfer PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE tinyinfer)

add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE tinyinfer)

add_executable(test_tensor_storage tests/test_tensor_storage.cpp)
target_link_libraries(test_tensor_storage PRIVATE tinyinfer)

add_executable(test_memory_plan tests/test_memory_plan.cpp)
target_link_libraries(test_memory_plan PRIVATE tinyinfer)

add_executable(test_threading_policy tests/test_threading_policy.cpp)
target_link_libraries(test_threading_policy PRIVATE tinyinfer)

add_executable(test_kernel_selection tests/test_kernel_selection.cpp)
target_link_libraries(test_kernel_selection PRIVATE tinyinfer)

add_executable(test_attention tests/test_attention.cpp)
target_link_libraries(test_attention PRIVATE tinyinfer)

enable_testing()
add_test(NAME test_ops COMMAND test_ops)
add_test(NAME test_runtime COMMAND test_runtime)
add_test(NAME test_tensor_storage COMMAND test_tensor_storage)
add_test(NAME test_memory_plan COMMAND test_memory_plan)
add_test(NAME test_threading_policy COMMAND test_threading_policy)
add_test(NAME test_kernel_selection COMMAND test_kernel_selection)
add_test(NAME test_attention COMMAND test_attention)

add_executable(demo examples/mlp.cpp)
target_link_libraries(demo PRIVATE tinyinfer)

add_executable(ops_demo examples/main.cpp)
target_link_libraries(ops_demo PRIVATE tinyinfer)

add_executable(graph_demo examples/graph_demo.cpp)
target_link_libraries(graph_demo PRIVATE tinyinfer)

add_executable(residual_demo examples/residual_demo.cpp)
target_link_libraries(residual_demo PRIVATE tinyinfer)

add_executable(model_demo examples/model.cpp)
target_link_libraries(model_demo PRIVATE tinyinfer)

add_executable(runtime_demo examples/runtime_demo.cpp)
target_link_libraries(runtime_demo PRIVATE tinyinfer)

add_executable(bench_matmul benchmark/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE tinyinfer)

add_executable(bench_graph_scheduler benchmark/bench_graph_scheduler.cpp)
target_link_libraries(bench_graph_scheduler PRIVATE tinyinfer)

add_executable(bench_session benchmark/bench_session.cpp)
target_link_libraries(bench_session PRIVATE tinyinfer)

add_executable(bench_profiler benchmark/bench_profiler.cpp)
target_link_libraries(bench_profiler PRIVATE tinyinfer)

add_executable(bench_tensor_storage benchmark/bench_tensor_storage.cpp)
target_link_libraries(bench_tensor_storage PRIVATE tinyinfer)

add_executable(bench_memory_plan benchmark/bench_memory_plan.cpp)
target_link_libraries(bench_memory_plan PRIVATE tinyinfer)

add_executable(bench_kernel_selection benchmark/bench_kernel_selection.cpp)
target_link_libraries(bench_kernel_selection PRIVATE tinyinfer)

add_executable(bench_kv_cache benchmark/bench_kv_cache.cpp)
target_link_libraries(bench_kv_cache PRIVATE tinyinfer)

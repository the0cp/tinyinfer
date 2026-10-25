cmake_minimum_required(VERSION 3.16)

project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    src/paged_kv_cache.cpp
    src/attention.cpp
    src/decoder_block.cpp
    src/qwen_model.cpp
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

enable_testing()
foreach(test_name IN ITEMS
    ops
    runtime
    tensor_storage
    memory_plan
    threading_policy
    kernel_selection
    attention
    paged_kv_cache
    decoder_block
    qwen_model
)
    add_executable(test_${test_name} tests/test_${test_name}.cpp)
    target_link_libraries(test_${test_name} PRIVATE tinyinfer)
    add_test(NAME test_${test_name} COMMAND test_${test_name})
endforeach()

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

add_executable(check_qwen_layer0 tools/check_qwen_layer0.cpp)
target_link_libraries(check_qwen_layer0 PRIVATE tinyinfer)

foreach(benchmark_name IN ITEMS
    matmul
    graph_scheduler
    session
    profiler
    tensor_storage
    memory_plan
    kernel_selection
    kv_cache
    paged_kv_cache
)
    add_executable(
        bench_${benchmark_name}
        benchmark/bench_${benchmark_name}.cpp
    )
    target_link_libraries(bench_${benchmark_name} PRIVATE tinyinfer)
endforeach()

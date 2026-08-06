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
    src/tensor.cpp
    src/ops.cpp
    src/thread_pool.cpp
    src/value_index.cpp
    src/operator_registry.cpp
    src/graph.cpp
    src/graph_optimizer.cpp
    src/execution_plan.cpp
    src/session_state.cpp
    src/execution_frame.cpp
    src/op_kernel.cpp
    src/kernel_registry.cpp
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

enable_testing()
add_test(NAME test_ops COMMAND test_ops)
add_test(NAME test_runtime COMMAND test_runtime)

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

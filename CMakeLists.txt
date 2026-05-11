cmake_minimum_required(VERSION 3.16)

project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tinyinfer
    src/ops.cpp
    src/tensor.cpp
    src/thread_pool.cpp
)

target_include_directories(tinyinfer PUBLIC include)

if(MSVC)
    target_compile_options(tinyinfer PRIVATE /W4)
else()
    target_compile_options(tinyinfer PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(demo
    examples/mlp.cpp
)

target_link_libraries(demo PRIVATE tinyinfer)

add_executable(test_ops
    tests/test_ops.cpp
)

target_link_libraries(test_ops PRIVATE tinyinfer)

add_executable(bench_matmul
    benchmark/bench_matmul.cpp
)

target_link_libraries(bench_matmul PRIVATE tinyinfer)
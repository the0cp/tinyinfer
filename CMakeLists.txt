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

add_library(tinyinfer)
add_subdirectory(src)

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

add_executable(legacy_mlp_demo examples/legacy_mlp_demo.cpp)
target_link_libraries(legacy_mlp_demo PRIVATE tinyinfer)

add_executable(ops_demo examples/ops_demo.cpp)
target_link_libraries(ops_demo PRIVATE tinyinfer)

add_executable(graph_demo examples/graph_demo.cpp)
target_link_libraries(graph_demo PRIVATE tinyinfer)

add_executable(residual_demo examples/residual_demo.cpp)
target_link_libraries(residual_demo PRIVATE tinyinfer)

add_executable(graph_model_demo examples/graph_model_demo.cpp)
target_link_libraries(graph_model_demo PRIVATE tinyinfer)

add_executable(runtime_demo examples/runtime_demo.cpp)
target_link_libraries(runtime_demo PRIVATE tinyinfer)

add_executable(check_qwen_layer0 tools/check_qwen_layer0.cpp)
target_link_libraries(check_qwen_layer0 PRIVATE tinyinfer)

add_executable(check_qwen_model tools/check_qwen_model.cpp)
target_link_libraries(check_qwen_model PRIVATE tinyinfer)

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

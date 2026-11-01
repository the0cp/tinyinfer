cmake_minimum_required(VERSION 3.19)

project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_library(tinyinfer_sanitizers INTERFACE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        tinyinfer_sanitizers INTERFACE
        -fsanitize=address,undefined -fno-omit-frame-pointer -g
    )
    target_link_options(tinyinfer_sanitizers INTERFACE -fsanitize=address,undefined)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "ThreadSanitizer")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "The ThreadSanitizer build requires GCC or Clang.")
    endif()
    target_compile_options(
        tinyinfer_sanitizers INTERFACE
        -fsanitize=thread -fno-omit-frame-pointer -g -O1
    )
    target_link_options(tinyinfer_sanitizers INTERFACE -fsanitize=thread)
endif()

find_package(Threads REQUIRED)

add_library(tinyinfer)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "tokenizers-cpp is missing. Run: git submodule update --init --recursive"
    )
endif()
find_program(
    CARGO_EXECUTABLE cargo
    HINTS
        "$ENV{USERPROFILE}/.cargo/bin"
        "$ENV{HOME}/.cargo/bin"
)
if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR "Cargo is required. Install Rust with rustup and reconfigure.")
endif()
add_subdirectory(third_party/tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)
add_library(tinyinfer_tokenizer src/llm/qwen2_tokenizer.cpp)
add_subdirectory(src)

target_include_directories(tinyinfer PUBLIC include)
target_link_libraries(tinyinfer PUBLIC Threads::Threads tinyinfer_sanitizers)

if(MSVC)
    target_compile_options(tinyinfer PRIVATE /W4)
else()
    target_compile_options(tinyinfer PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

target_include_directories(tinyinfer_tokenizer PUBLIC include)
target_link_libraries(tinyinfer_tokenizer PUBLIC tinyinfer PRIVATE tokenizers_cpp)
if(MSVC)
    target_compile_options(tinyinfer_tokenizer PRIVATE /W4)
else()
    target_compile_options(
        tinyinfer_tokenizer PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
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

add_executable(test_qwen_tokenizer tests/test_qwen_tokenizer.cpp)
target_compile_definitions(
    test_qwen_tokenizer PRIVATE
    TINYINFER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
)
target_link_libraries(test_qwen_tokenizer PRIVATE tinyinfer_tokenizer)
add_test(NAME test_qwen_tokenizer COMMAND test_qwen_tokenizer)

add_executable(tinyinfer_cli apps/tinyinfer.cpp)
set_target_properties(tinyinfer_cli PROPERTIES OUTPUT_NAME tinyinfer)
target_link_libraries(tinyinfer_cli PRIVATE tinyinfer_tokenizer)
add_test(NAME test_cli_help COMMAND tinyinfer_cli --help)
if(MSVC)
    target_compile_options(tinyinfer_cli PRIVATE /W4)
else()
    target_compile_options(
        tinyinfer_cli PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()

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
target_link_libraries(check_qwen_model PRIVATE tinyinfer_tokenizer)

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

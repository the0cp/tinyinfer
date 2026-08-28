#include "buffer.h"
#include "data_type.h"
#include "inference_session.h"
#include "model_loader.h"
#include "ops.h"
#include "tensor.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace{

using namespace tinyinfer;

void expect(bool condition, const std::string& message){
    if(!condition){
        throw std::runtime_error("EXPECT failed: " + message);
    }
}

void expect_close(float actual, float expected, float tolerance = 1e-5f){
    if(std::fabs(actual - expected) > tolerance){
        throw std::runtime_error(
            "EXPECT_CLOSE failed: actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected)
        );
    }
}

template<typename Function>
void expect_throw(Function&& function, const std::string& needle){
    try{
        function();
    }catch(const std::exception& error){
        expect(
            std::string(error.what()).find(needle) != std::string::npos,
            "exception did not contain '" + needle + "': " + error.what()
        );
        return;
    }

    throw std::runtime_error("Expected exception containing: " + needle);
}

std::shared_ptr<Buffer> float_buffer(std::initializer_list<float> values){
    auto buffer = make_cpu_buffer(values.size() * sizeof(float));
    auto* data = static_cast<float*>(buffer->data());
    size_t index = 0;

    for(float value : values){
        data[index++] = value;
    }

    return buffer;
}

Tensor make_shared_view(){
    auto buffer = float_buffer({1.0f, 2.0f, 3.0f, 4.0f});
    return Tensor(buffer, 0, {2, 2});
}

void test_dtype_and_checked_byte_helpers(){
    expect(element_size(DataType::Float32) == sizeof(float), "float32 element size");
    expect(element_alignment(DataType::Float32) == alignof(float), "float32 alignment");
    expect(data_type_name(DataType::Float32) == "float32", "float32 name");
    expect(tensor_numel({2, 3, 4}) == 24, "checked numel");
    expect(tensor_bytes({2, 3}) == 6 * sizeof(float), "checked bytes");

    expect_throw(
        [](){ (void)tensor_numel({std::numeric_limits<size_t>::max(), 2}); },
        "overflows"
    );

    expect(
        tensor_numel({0, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()}) == 0,
        "an empty shape must not overflow on unreachable dimensions"
    );
}

void test_cpu_buffer_contract(){
    CpuBuffer empty(0);
    expect(empty.size_bytes() == 0, "zero buffer size");
    expect(empty.data() == nullptr, "zero buffer data is null");

    CpuBuffer buffer(64);
    expect(buffer.size_bytes() == 64, "buffer byte size");
    expect(buffer.data() != nullptr, "non-empty buffer data");
}

void test_tensor_owning_constructors_keep_source_api(){
    Tensor zeros({2, 3});
    expect(zeros.is_contiguous(), "owning tensor is contiguous");
    expect(zeros.dtype() == DataType::Float32, "owning tensor dtype");
    expect(zeros.byte_offset() == 0, "owning tensor offset");
    expect(zeros.size_bytes() == 6 * sizeof(float), "owning tensor bytes");
    expect(zeros.buffer()->size_bytes() == zeros.size_bytes(), "owning buffer size");

    for(size_t i = 0; i < zeros.numel(); i++){
        expect_close(zeros.data()[i], 0.0f);
    }

    Tensor values({2, 2}, {1, 2, 3, 4});
    expect_close(values.at({1, 0}), 3.0f);

    expect_throw(
        [](){ Tensor ignored({1024}, {1.0f}); (void)ignored; },
        "does not match"
    );
}

void test_shared_storage_lifetime_aliasing_and_clone(){
    Tensor view = make_shared_view();
    expect_close(view.at({1, 1}), 4.0f);

    Tensor alias = view;
    expect(alias.buffer().get() == view.buffer().get(), "Tensor copy shares storage handle");

    alias.at({0, 1}) = 9.0f;
    expect_close(view.at({0, 1}), 9.0f);

    Tensor independent = view.clone();
    expect(independent.buffer().get() != view.buffer().get(), "clone owns independent storage");
    independent.at({0, 1}) = -7.0f;
    expect_close(view.at({0, 1}), 9.0f);
}

void test_offset_and_strided_indexing(){
    auto buffer = float_buffer({10, 11, 12, 13, 14, 15, 16, 17});

    Tensor offset_view(buffer, 2 * sizeof(float), {2, 3});
    expect_close(offset_view.at({0, 0}), 12.0f);
    expect_close(offset_view.at({1, 2}), 17.0f);
    expect(offset_view.data() == static_cast<float*>(buffer->data()) + 2, "offset data pointer");

    Tensor padded(buffer, 0, {2, 2}, Strides{3, 1});
    expect(!padded.is_contiguous(), "padded view is non-contiguous");
    expect_close(padded.at({0, 0}), 10.0f);
    expect_close(padded.at({0, 1}), 11.0f);
    expect_close(padded.at({1, 0}), 13.0f);
    expect_close(padded.at({1, 1}), 14.0f);
    expect_throw([&](){ (void)padded.data(); }, "contiguous");

    Tensor cloned = padded.clone();
    expect(cloned.is_contiguous(), "a strided clone is materialized contiguously");
    expect_close(cloned.data()[0], 10.0f);
    expect_close(cloned.data()[2], 13.0f);

    padded.fill(-3.0f);
    expect_close(padded.at({0, 0}), -3.0f);
    expect_close(padded.at({1, 1}), -3.0f);
    expect_close(static_cast<float*>(buffer->data())[2], 12.0f);
}

void test_reshape_is_metadata_only(){
    Tensor tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    const Buffer* storage = tensor.buffer().get();
    const long original_use_count = tensor.buffer().use_count();

    Tensor reshaped = tensor.reshape({3, 2});

    expect(reshaped.buffer().get() == storage, "reshape reuses backing buffer");
    expect(reshaped.byte_offset() == tensor.byte_offset(), "reshape preserves byte offset");
    expect(reshaped.buffer().use_count() == original_use_count + 1, "reshape shares ownership");
    expect(reshaped.strides() == Strides({2, 1}), "reshape recomputes contiguous strides");
    expect_close(reshaped.at({2, 1}), 6.0f);

    reshaped.at({0, 0}) = 42.0f;
    expect_close(tensor.at({0, 0}), 42.0f);

    expect_throw([&](){ (void)tensor.reshape({4, 2}); }, "element count");

    auto buffer = float_buffer({1, 2, 3, 4, 5, 6});
    Tensor padded(buffer, 0, {2, 2}, Strides{3, 1});
    expect_throw([&](){ (void)padded.reshape({4}); }, "contiguous");

    Tensor singleton_stride(buffer, 0, {1, 3}, Strides{99, 1});
    expect(singleton_stride.is_contiguous(), "size-one dimensions do not break contiguity");
    expect_close(singleton_stride.reshape({3}).at({2}), 3.0f);
}

void test_layout_validation(){
    auto buffer = make_cpu_buffer(4 * sizeof(float));

    expect_throw(
        [&](){ Tensor ignored(buffer, 1, {1}); (void)ignored; },
        "aligned"
    );

    expect_throw(
        [&](){ Tensor ignored(buffer, 0, {2, 2}, Strides{2}); (void)ignored; },
        "ranks"
    );

    expect_throw(
        [&](){ Tensor ignored(buffer, 3 * sizeof(float), {2}); (void)ignored; },
        "exceeds"
    );

    expect_throw(
        [&](){
            Tensor ignored(
                buffer,
                0,
                {2, 2},
                Strides{std::numeric_limits<size_t>::max(), 1}
            );
            (void)ignored;
        },
        "overflows"
    );

    expect_throw(
        [](){ Tensor ignored(std::shared_ptr<Buffer>{}, 0, {1}); (void)ignored; },
        "non-null"
    );
}

void test_zero_sized_tensor_has_safe_null_data(){
    Tensor zero({2, 0, 3});
    expect(zero.numel() == 0, "zero tensor numel");
    expect(zero.size_bytes() == 0, "zero tensor bytes");
    expect(zero.data() == nullptr, "zero tensor data pointer");
    expect(zero.is_contiguous(), "zero tensor uses a canonical contiguous layout");

    Tensor reshaped = zero.reshape({0});
    expect(reshaped.buffer().get() == zero.buffer().get(), "empty reshape still shares storage");
}

void test_layout_policy_is_explicit_at_kernel_boundary(){
    auto buffer = float_buffer({1, 2, 99, 3, 4, 99});
    Tensor strided(buffer, 0, {2, 2}, Strides{3, 1});
    Tensor identity({2, 2}, {1, 0, 0, 1});

    Tensor reference = naive_matmul(strided, identity);
    expect_close(reference.at({0, 0}), 1.0f);
    expect_close(reference.at({1, 1}), 4.0f);

    expect_throw([&](){ (void)fast_matmul(strided, identity); }, "contiguous");
}

void test_session_accepts_externally_backed_input(){
    Graph graph;
    graph.add_node("relu", OpType::ReLU, {"input"}, "output");

    InferenceSession session;
    session.load(std::move(graph), "input", {1, 2}, "output");
    session.initialize();

    const ValueInfo& input_info = session.plan().value_info(session.plan().input_index());
    expect(input_info.dtype == DataType::Float32, "execution plan preserves value dtype");
    expect(input_info.byte_size == tensor_bytes({1, 2}), "execution plan uses dtype-aware bytes");

    auto buffer = float_buffer({-2.0f, 5.0f});
    Tensor input(buffer, 0, {1, 2});
    Tensor output = session.run(input);

    expect_close(output.at({0, 0}), 0.0f);
    expect_close(output.at({0, 1}), 5.0f);
}

void test_model_writer_materializes_strided_tensor(){
    auto buffer = float_buffer({1, 2, 99, 3, 4, 99});
    Tensor strided(buffer, 0, {2, 2}, Strides{3, 1});

    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1};
    package.output_name = "weight";
    package.tensors.push_back(NamedTensor{"weight", strided});

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "tinyinfer_strided_model_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::filesystem::path manifest = directory / "model.ti";

    ModelWriter::save(package, manifest);
    const LoadedGraph loaded = ModelLoader::load_graph(manifest);
    const Tensor& restored = loaded.graph.constant("weight");

    expect(restored.is_contiguous(), "serialized strided tensor is materialized contiguously");
    expect_close(restored.data()[0], 1.0f);
    expect_close(restored.data()[1], 2.0f);
    expect_close(restored.data()[2], 3.0f);
    expect_close(restored.data()[3], 4.0f);

    std::filesystem::remove_all(directory);
}

struct TestCase{
    const char* name;
    void (*run)();
};

}

int main(){
    const TestCase tests[]{
        {"dtype_and_checked_byte_helpers", test_dtype_and_checked_byte_helpers},
        {"cpu_buffer_contract", test_cpu_buffer_contract},
        {"tensor_owning_constructors_keep_source_api", test_tensor_owning_constructors_keep_source_api},
        {"shared_storage_lifetime_aliasing_and_clone", test_shared_storage_lifetime_aliasing_and_clone},
        {"offset_and_strided_indexing", test_offset_and_strided_indexing},
        {"reshape_is_metadata_only", test_reshape_is_metadata_only},
        {"layout_validation", test_layout_validation},
        {"zero_sized_tensor_has_safe_null_data", test_zero_sized_tensor_has_safe_null_data},
        {"layout_policy_is_explicit_at_kernel_boundary", test_layout_policy_is_explicit_at_kernel_boundary},
        {"session_accepts_externally_backed_input", test_session_accepts_externally_backed_input},
        {"model_writer_materializes_strided_tensor", test_model_writer_materializes_strided_tensor}
    };

    size_t passed = 0;

    for(const TestCase& test : tests){
        try{
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
            passed++;
        }catch(const std::exception& error){
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << "\n";
            return 1;
        }
    }

    std::cout << "Passed " << passed << " tensor storage tests.\n";
    return 0;
}

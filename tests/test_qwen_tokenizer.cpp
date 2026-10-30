#include <tinyinfer/llm/qwen2_tokenizer.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace{

using namespace tinyinfer;

void expect(bool condition, const std::string& message){
    if(!condition){
        throw std::runtime_error("EXPECT failed: " + message);
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

void test_hugging_face_json(){
    const std::filesystem::path path =
        std::filesystem::path(TINYINFER_TEST_DATA_DIR) / "tiny_tokenizer.json";
    const Qwen2Tokenizer tokenizer = load_qwen2_tokenizer(path);

    expect(tokenizer.encode("ab") == std::vector<int64_t>({2}), "BPE merge");
    const std::vector<int64_t> ordinary{2};
    expect(tokenizer.decode(ordinary) == "ab", "ordinary-token decode");

    const std::vector<int64_t> special = tokenizer.encode(
        "<|im_start|>ab<|im_end|>"
    );
    expect(special == std::vector<int64_t>({4, 2, 5}), "added special tokens");
    expect(
        tokenizer.decode(special) == "<|im_start|>ab<|im_end|>",
        "decode with special tokens"
    );
    expect(tokenizer.decode(special, true) == "ab", "skip special tokens");
}

void test_chat_prompt(){
    expect(
        qwen2_chat_prompt("Be concise.", "What is KV cache?") ==
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nWhat is KV cache?<|im_end|>\n"
            "<|im_start|>assistant\n",
        "fixed Qwen2 chat format"
    );
}

void test_file_boundary_errors(){
    const std::filesystem::path missing =
        std::filesystem::temp_directory_path() / "tinyinfer_missing_tokenizer.json";
    std::filesystem::remove(missing);
    expect_throw([&](){ (void)load_qwen2_tokenizer(missing); }, "Cannot open");

    const std::filesystem::path empty =
        std::filesystem::temp_directory_path() / "tinyinfer_empty_tokenizer.json";
    std::ofstream(empty, std::ios::trunc);
    expect_throw([&](){ (void)load_qwen2_tokenizer(empty); }, "empty");
    std::filesystem::remove(empty);
}

}

int main(){
    try{
        test_hugging_face_json();
        test_chat_prompt();
        test_file_boundary_errors();
        std::cout << "qwen tokenizer tests passed\n";
        return 0;
    }catch(const std::exception& error){
        std::cerr << error.what() << '\n';
        return 1;
    }
}

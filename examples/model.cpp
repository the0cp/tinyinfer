#include "model_loader.h"

#include <filesystem>
#include <iostream>

int main(){
    using namespace tinyinfer;

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "tinyinfer_model_demo";
    std::filesystem::remove_all(directory);

    ModelPackage package;
    package.input_name = "input";
    package.input_shape = {1, 2};
    package.output_name = "output";
    package.tensors.push_back(NamedTensor{
        "weight", Tensor({2, 1}, {2.0f, 3.0f})
    });
    package.tensors.push_back(NamedTensor{"bias", Tensor({1}, {1.0f})});
    package.nodes.push_back(NodeMetadata{
        "linear", "Linear", {"input", "weight", "bias"}, "output"
    });

    const std::filesystem::path manifest = directory / "model.ti";
    ModelWriter::save(package, manifest);
    std::unique_ptr<InferenceSession> session = ModelLoader::load(manifest);
    Tensor output = session->run(Tensor({1, 2}, {4.0f, 5.0f}));
    std::cout << "loaded model output = " << output.data()[0] << "\n";
    std::filesystem::remove_all(directory);
    return 0;
}

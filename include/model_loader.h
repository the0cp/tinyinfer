#pragma once

#include "inference_session.h"
#include "model_format.h"
#include "session_options.h"

#include <filesystem>
#include <memory>
#include <string>

namespace tinyinfer{

class ModelLoader{
public:
    static ModelMetadata parse_manifest(const std::filesystem::path& manifest_path);
    static LoadedGraph load_graph(const std::filesystem::path& manifest_path);

    static std::unique_ptr<InferenceSession> load(
        const std::filesystem::path& manifest_path,
        SessionOptions options = {}
    );
};

class ModelWriter{
public:
    static void save(
        const ModelPackage& package,
        const std::filesystem::path& manifest_path,
        const std::string& weights_filename = "weights.bin"
    );
};

}

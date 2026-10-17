// Asset-backed checks of optional flow preprocessing. No Kitten dependencies.
#include "engine/framework/codecs/s3gen_runtime.h"
#include "engine/framework/core/module.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace s3 = engine::codecs::s3gen;
namespace core = engine::core;
namespace assets = engine::assets;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: s3gen_input_transform_probe S3GEN cpu|cuda OUTPUT_DIR");
        core::BackendConfig backend;
        backend.threads = 8;
        const std::string backend_name = argv[2];
        require(backend_name == "cpu" || backend_name == "cuda", "unsupported backend");
        backend.type = backend_name == "cuda" ? core::BackendType::Cuda : core::BackendType::Cpu;
        core::ExecutionContext execution(backend);
        auto source = assets::open_tensor_source(argv[1]);
        std::filesystem::create_directories(argv[3]);
        s3::S3GenConditioning ref;
        ref.prompt_tokens = {4299, 17, 306, 4299};
        ref.prompt_token_count = ref.prompt_tokens.size();
        ref.prompt_feat_frames = 2 * ref.prompt_token_count;
        ref.prompt_feat_dims = 80;
        ref.prompt_feat.resize(ref.prompt_feat_frames * 80, 0.1f);
        ref.embedding.resize(192, 0.05f);
        ref.embedding_size = 192;
        const std::vector<int32_t> tokens = {123, 456, 789, 4299, 4299, 4299};
        const std::vector<float> noise(80 * 2 * (ref.prompt_tokens.size() + tokens.size()), 0.125f);

        for (bool meanflow : {false, true}) {
            s3::S3GenConfig config;
            config.decoder.variant = meanflow ? s3::S3FlowVariant::MeanFlow : s3::S3FlowVariant::Conditional;
            const std::string label = meanflow ? "meanflow" : "conditional";
            auto run = [&](const s3::S3GenRuntimeOptions & options) {
                s3::S3GenRuntime runtime(source, execution, config, options);
                return runtime.token_to_mel(ref, tokens, tokens.size(), 2, meanflow ? 0 : 0.7f, !meanflow, noise).mel;
            };
            s3::S3GenRuntimeOptions options{assets::TensorStorageType::F32};
            const auto original = run(options);
            require(!original.empty(), "empty baseline mel");
            for (float value : original) require(std::isfinite(value), "nonfinite baseline mel");
            std::ofstream output(std::filesystem::path(argv[3]) / (label + ".f32"), std::ios::binary);
            output.write(reinterpret_cast<const char *>(original.data()), original.size() * sizeof(float));

            std::set<std::string> seen;
            options.flow_input_transform = [&](core::ModuleBuildContext &, const core::TensorValue & input, std::string_view name) {
                const auto shape = source->require_metadata(name).shape;
                require(shape.size() == 2 || shape.size() == 3, "hook received a non-projection weight");
                require(input.shape.last_dim() == shape[1], "hook feature axis does not match input channels");
                seen.emplace(name);
                return input;
            };
            const auto identity = run(options);
            require(original == identity, label + " identity preprocessing changed mel");
            for (const std::string name : {"flow.encoder_proj.weight", "flow.spk_embed_affine_layer.weight",
                     "flow.encoder.pre_lookahead_layer.conv1.weight", "flow.decoder.estimator.final_proj.weight"})
                require(seen.count(name) != 0, "hook missed " + name);
            require(seen.count("flow.decoder.estimator.time_embed_mixer.weight") == (meanflow ? 1 : 0),
                    "time mixer hook does not match decoder variant");

            options.flow_input_transform = [](core::ModuleBuildContext & ctx, const core::TensorValue & input, std::string_view name) {
                if (name == "flow.encoder_proj.weight")
                    return core::wrap_tensor(ggml_scale(ctx.ggml, input.tensor, 0.0f), input.shape);
                return input;
            };
            require(original != run(options), label + " selected projection hook had no effect");
            options.flow_input_transform = {};
            config.decoder.use_flash_attention = false;
            const auto unfused = run(options);
            require(unfused.size() == original.size(), "unfused attention shape mismatch");
            double error_squared = 0, reference_squared = 0;
            for (size_t i = 0; i < original.size(); ++i) {
                require(std::isfinite(unfused[i]), "nonfinite unfused attention mel");
                const double error = unfused[i] - original[i];
                error_squared += error * error;
                reference_squared += static_cast<double>(original[i]) * original[i];
            }
            const double relative_error = std::sqrt(error_squared / reference_squared);
            require(relative_error < 0.005, label + " unfused attention relative error exceeds 0.5%");
            std::cout << label << ": unfused attention relative L2 " << relative_error << '\n';
            std::cout << label << ": identity exact; selected transform active; " << seen.size() << " projection names checked\n";
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

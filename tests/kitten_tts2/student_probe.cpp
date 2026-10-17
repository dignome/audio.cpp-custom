#include "engine/community_models/kitten_tts2/student_decoder.h"
#include "engine/framework/io/json.h"
#include <ggml-alloc.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace core = engine::core;
namespace assets = engine::assets;
namespace kitten = engine::community_models::kitten_tts2;
namespace s3 = engine::codecs::s3gen;
namespace json = engine::io::json;

void write_f32(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    if (!out) throw std::runtime_error("cannot write probe output");
}

int main(int argc, char ** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: kitten_tts2_student_probe S3GEN VOICES CODES VALIDATION_DIR OUTPUT_DIR cpu|cuda");
        core::BackendConfig backend;
        backend.threads = 8;
        const std::string mode = argv[6];
        if (mode != "cpu" && mode != "cuda") throw std::runtime_error("invalid backend");
        backend.type = mode == "cuda" ? core::BackendType::Cuda : core::BackendType::Cpu;
        core::ExecutionContext execution(backend);
        const auto outdir = std::filesystem::path(argv[5]);
        std::filesystem::create_directories(outdir);
        {
            std::ifstream input(std::filesystem::path(argv[4]) / "activation-input.f32", std::ios::binary | std::ios::ate);
            if (!input || input.tellg() != 7 * 368 * sizeof(float)) throw std::runtime_error("invalid activation fixture");
            input.seekg(0);
            std::vector<float> values(7 * 368);
            input.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float));
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
            core::ModuleBuildContext build{context.get(), "kitten.student.quantization", execution.backend_type()};
            auto x = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({7, 368}));
            ggml_set_input(x.tensor);
            auto y = kitten::quantize_student_activations(build, x);
            ggml_set_output(y.tensor);
            auto * graph = ggml_new_graph_custom(context.get(), 128, false);
            ggml_build_forward_expand(graph, y.tensor);
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
            if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) throw std::runtime_error("activation allocation failed");
            ggml_backend_tensor_set(x.tensor, values.data(), 0, values.size() * sizeof(float));
            if (core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("activation inference failed");
            write_f32(outdir / "activation.f32", core::read_tensor_f32(y.tensor));
            core::release_backend_graph_resources(execution.backend(), graph);
        }
        auto packed = assets::open_tensor_source(argv[1]);
        auto source = kitten::make_student_tensor_source(packed);
        for (const std::string name : {"encoder.embed.out.0.weight", "encoder.pre_lookahead_layer.conv1.weight"})
            write_f32(outdir / (name + ".f32"), source->require_f32("flow." + name));
        std::vector<std::string> packed_names;
        for (const auto & tensor : packed->tensors()) {
            if (tensor.name.size() >= 8 && tensor.name.compare(tensor.name.size()-8,8,".packed4") == 0)
                packed_names.push_back(tensor.name.substr(0,tensor.name.size()-8));
        }
        std::sort(packed_names.begin(),packed_names.end());
        std::ofstream all_weights(outdir / "packed-weights.f32",std::ios::binary);
        for (const auto & name : packed_names) {
            const auto values = source->require_f32(name);
            all_weights.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
        }
        if (!all_weights) throw std::runtime_error("cannot write unpacked student weights");
        s3::S3GenRuntime runtime(source, execution, kitten::student_decoder_config(), kitten::student_decoder_options(packed, execution.backend_type()));
        const auto voices = json::parse_file(argv[2]);
        const auto & voice = voices.require("Bruno");
        s3::S3GenConditioning ref;
        for (const auto & value : voice.require("prompt_token").as_array()[0].as_array()) ref.prompt_tokens.push_back(value.as_i64());
        for (const auto & row : voice.require("prompt_feat").as_array()[0].as_array())
            for (const auto & value : row.as_array()) ref.prompt_feat.push_back(value.as_f32());
        for (const auto & value : voice.require("embedding").as_array()[0].as_array()) ref.embedding.push_back(value.as_f32());
        ref.prompt_token_count = ref.prompt_tokens.size();
        ref.prompt_feat_frames = ref.prompt_feat.size() / 80;
        ref.prompt_feat_dims = 80;
        ref.embedding_size = 192;
        const auto codes = json::parse_file(argv[3]);
        std::vector<int32_t> tokens;
        for (const auto & value : codes.as_array()) tokens.push_back(value.as_i64());
        tokens.insert(tokens.end(), 3, 4299);
        auto all_tokens = ref.prompt_tokens;
        all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
        auto table = source->require_f32("flow.input_embedding.weight", {6561, 368});
        std::vector<float> embedded;
        for (auto token : all_tokens) embedded.insert(embedded.end(), table.begin() + token * 368, table.begin() + (token + 1) * 368);
        const auto encoded = runtime.encode(embedded, all_tokens.size(), all_tokens.size(), 368);
        write_f32(outdir / "encoder.f32", encoded.hidden);
        const int64_t frames = 2 * all_tokens.size();
        const auto mel = runtime.token_to_mel(ref, tokens, tokens.size(), 1, 0, false, std::vector<float>(80 * frames, 0), 0);
        write_f32(outdir / "mel.f32", mel.mel);
        const auto audio = runtime.decode_waveform(mel.mel, mel.frames);
        write_f32(outdir / "waveform.f32", audio.waveform);
        std::cout << "student mel_frames=" << mel.frames << " audio_samples=" << audio.waveform.size() << '\n';
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}

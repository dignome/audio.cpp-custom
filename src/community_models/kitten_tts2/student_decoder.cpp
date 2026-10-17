#include "engine/community_models/kitten_tts2/student_decoder.h"
#include "engine/community_models/kitten_tts2/student_int8.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace engine::community_models::kitten_tts2 {
namespace {
bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}
class StudentTensorSource final : public assets::TensorSource {
    struct Packed {
        assets::TensorMetadata metadata;
        size_t count;
        bool wide;
    };
    std::shared_ptr<const assets::TensorSource> source_;
    std::unordered_map<std::string, Packed> packed_;
public:
    explicit StudentTensorSource(std::shared_ptr<const assets::TensorSource> source) : source_(std::move(source)) {
        size_t narrow_count = 0, wide_count = 0;
        for (const auto & tensor : source_->tensors()) {
            const bool wide = ends_with(tensor.name, ".packed8");
            const std::string_view suffix = wide ? ".packed8" : ".packed4";
            if (!ends_with(tensor.name, suffix)) continue;
            if (wide) ++wide_count; else ++narrow_count;
            const auto name = tensor.name.substr(0, tensor.name.size() - suffix.size());
            const auto shape_data = source_->require_tensor_data(name + ".shape");
            if (assets::ggml_type_for_tensor_dtype(shape_data.metadata.dtype) != GGML_TYPE_I64 ||
                shape_data.metadata.shape.size() != 1 ||
                (shape_data.metadata.shape[0] != 2 && shape_data.metadata.shape[0] != 3) ||
                shape_data.bytes.size() != static_cast<size_t>(shape_data.metadata.shape[0]) * sizeof(int64_t))
                throw std::runtime_error("invalid student packed shape: " + name);
            std::vector<int64_t> shape(shape_data.metadata.shape[0]);
            std::memcpy(shape.data(), shape_data.bytes.data(), shape_data.bytes.size());
            size_t count = 1;
            for (auto dim : shape) {
                if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / sizeof(float) / count)
                    throw std::runtime_error("invalid student packed dimensions: " + name);
                count *= static_cast<size_t>(dim);
            }
            if (assets::ggml_type_for_tensor_dtype(tensor.dtype) != GGML_TYPE_I8 ||
                tensor.shape != std::vector<int64_t>{static_cast<int64_t>(wide ? count : (count + 1) / 2)} || source_->has_tensor(name))
                throw std::runtime_error("invalid student packed payload: " + name);
            assets::require_tensor_shape(*source_, name + ".scale", {shape[0]});
            packed_.emplace(name, Packed{{name, "F32", std::move(shape)}, count, wide});
        }
        if (!((narrow_count == 474 && wide_count == 0) ||
              (wide_count == 457 && narrow_count == 0 && source_->has_tensor("flow.decoder.estimator.time_embed_mixer.weight.folded"))))
            throw std::runtime_error("unsupported student projection layout");
    }
    const std::filesystem::path & source_path() const noexcept override { return source_->source_path(); }
    bool has_tensor(std::string_view name) const noexcept override {
        return packed_.count(std::string(name)) || source_->has_tensor(name);
    }
    assets::TensorMetadata require_metadata(std::string_view name) const override {
        const auto it = packed_.find(std::string(name));
        return it == packed_.end() ? source_->require_metadata(name) : it->second.metadata;
    }
    std::vector<assets::TensorMetadata> tensors() const override {
        auto result = source_->tensors();
        for (const auto & [name, packed] : packed_) result.push_back(packed.metadata);
        return result;
    }
    void release_storage() const override { source_->release_storage(); }
    std::vector<float> require_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & expected = std::nullopt) const override {
        const auto it = packed_.find(std::string(name));
        if (it == packed_.end()) return source_->require_f32(name, expected);
        const auto & info = it->second;
        if (expected && *expected != info.metadata.shape) throw std::runtime_error("student tensor shape mismatch");
        const auto raw = source_->require_tensor_data(std::string(name) + (info.wide ? ".packed8" : ".packed4"));
        if (raw.bytes.size() != (info.wide ? info.count : (info.count + 1) / 2)) throw std::runtime_error("truncated student packed tensor");
        const auto scales = source_->require_f32(std::string(name) + ".scale", {info.metadata.shape[0]});
        for (float scale : scales) if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("invalid student scale");
        const size_t row_size = info.count / scales.size();
        std::vector<float> values(info.count);
        for (size_t i = 0; i < info.count; ++i) {
            const auto byte = std::to_integer<unsigned>(raw.bytes[info.wide ? i : i / 2]);
            const int code = info.wide ? (byte < 128 ? static_cast<int>(byte) : static_cast<int>(byte)-256)
                : static_cast<int>((byte >> (4 * (1 - i % 2))) & 15) - 8;
            values[i] = static_cast<float>(code) / scales[i / row_size];
        }
        return values;
    }
    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        if (!packed_.count(std::string(name))) return source_->require_tensor_data(name);
        auto values = require_f32(name);
        assets::RawTensorData output{require_metadata(name), std::vector<std::byte>(values.size() * sizeof(float))};
        std::memcpy(output.bytes.data(), values.data(), output.bytes.size());
        return output;
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & expected = std::nullopt) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected);
    }
    int64_t require_i64_scalar(std::string_view name) const override { return source_->require_i64_scalar(name); }
};
}

std::shared_ptr<const assets::TensorSource> make_student_tensor_source(std::shared_ptr<const assets::TensorSource> source) {
    return std::make_shared<StudentTensorSource>(std::move(source));
}

core::TensorValue quantize_student_activations(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const auto features = input.shape.last_dim();
    const auto rows = input.shape.prefix_elements();
    auto * x = ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, input.tensor), features, rows);
    // A row reduction expressed with existing CPU/CUDA max-pooling operators.
    auto * maximum = ggml_pool_2d(ctx.ggml, ggml_abs(ctx.ggml, x), GGML_OP_POOL_MAX,
        static_cast<int>(features), 1, static_cast<int>(features), 1, 0, 0);
    maximum = ggml_clamp(ctx.ggml, maximum, 1e-5f, std::numeric_limits<float>::max());
    auto * one = ggml_scale_bias(ctx.ggml, maximum, 0.0f, 127.0f);
    auto * scale = ggml_div(ctx.ggml, one, maximum);
    auto * scaled = ggml_mul(ctx.ggml, x, scale);
    // torch.round uses ties-to-even. ggml_round uses ties-away-from-zero.
    // For finite values in [-128,128], the two F32 additions below implement
    // ties-to-even with the default IEEE rounding mode on both CPU and CUDA.
    auto * rounded = ggml_scale_bias(ctx.ggml, scaled, 1.0f, 12582912.0f);
    rounded = ggml_scale_bias(ctx.ggml, rounded, 1.0f, -12582912.0f);
    rounded = ggml_clamp(ctx.ggml, rounded, -128.0f, 127.0f);
    auto result = core::wrap_tensor(ggml_div(ctx.ggml, rounded, scale), core::TensorShape::from_dims({rows, features}));
    return core::reshape_tensor(ctx, result, input.shape);
}

codecs::s3gen::S3GenConfig student_decoder_config() {
    codecs::s3gen::S3GenConfig config;
    config.encoder.hidden_size = 368;
    config.encoder.feed_forward_size = 1472;
    config.decoder.variant = codecs::s3gen::S3FlowVariant::MeanFlow;
    config.decoder.hidden_size = 176;
    config.decoder.head_dim = 44;
    config.decoder.use_flash_attention = false;
    config.decoder.feed_forward_size = 704;
    config.decoder.time_hidden_size = 704;
    return config;
}

codecs::s3gen::S3GenRuntimeOptions student_decoder_options(std::shared_ptr<const assets::TensorSource> source, core::BackendType backend) {
    std::unordered_set<std::string> names;
    for (const auto & tensor : source->tensors()) if (ends_with(tensor.name, ".packed4") || ends_with(tensor.name, ".packed8"))
        names.emplace(tensor.name.substr(0, tensor.name.size() - 8));
    codecs::s3gen::S3GenRuntimeOptions options{assets::TensorStorageType::F32};
    options.flow_embedding_storage_type = assets::TensorStorageType::F32;
    const char * mode = std::getenv("KITTEN_TTS2_STUDENT_LINEAR");
    if (mode && std::string_view(mode) != "f32" && std::string_view(mode) != "int8")
        throw std::runtime_error("KITTEN_TTS2_STUDENT_LINEAR must be f32 or int8");
    // CPU packages use integer linears by default. Keep the F32 reference path
    // available for component parity tests, without affecting other S3 users.
    if (backend == core::BackendType::Cpu && (!mode || std::string_view(mode) == "int8"))
        options.flow_linear_factory = make_student_int8_factory(source);
    if (source->has_tensor("flow.decoder.estimator.time_embed_mixer.weight.folded")) {
        if (backend != core::BackendType::Cpu) throw std::runtime_error("frozen student_w8 requires CPU");
        // The shipped one-step export folds time conditioning into block biases.
        // These projections contribute zero; biases retain the exported constants.
        auto integer_factory = options.flow_linear_factory;
        options.flow_linear_factory = [source, integer_factory](std::string_view name, int64_t input, int64_t output) -> codecs::s3gen::S3FlowLinearOperation {
            if (!source->has_tensor(std::string(name) + ".folded"))
                return integer_factory ? integer_factory(name, input, output) : codecs::s3gen::S3FlowLinearOperation{};
            const auto raw = source->require_tensor_data(std::string(name) + ".folded");
            int64_t shape[2]{};
            if (assets::ggml_type_for_tensor_dtype(raw.metadata.dtype) != GGML_TYPE_I64 || raw.bytes.size() != sizeof(shape))
                throw std::runtime_error("invalid folded student projection");
            std::memcpy(shape, raw.bytes.data(), sizeof(shape));
            if (shape[0] != output || shape[1] != input) throw std::runtime_error("folded student shape mismatch");
            return [output](core::ModuleBuildContext & ctx, const core::TensorValue & x) {
                auto * in = x.tensor;
                auto zero = [](ggml_tensor * dst, int, int, void *) {
                    std::memset(dst->data, 0, ggml_nbytes(dst));
                };
                auto * y = ggml_custom_4d(ctx.ggml, GGML_TYPE_F32, output, x.shape.prefix_elements(), 1, 1,
                    &in, 1, zero, 1, nullptr);
                return core::wrap_tensor(y, x.shape.with_last_dim(output));
            };
        };
    }
    options.flow_input_transform = [names = std::move(names)](core::ModuleBuildContext & ctx,
        const core::TensorValue & input, std::string_view name) {
        return names.count(std::string(name)) ? quantize_student_activations(ctx, input) : input;
    };
    return options;
}
}

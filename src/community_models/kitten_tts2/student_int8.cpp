#include "engine/community_models/kitten_tts2/student_int8.h"
#include "engine/framework/debug/trace.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace engine::community_models::kitten_tts2 {
bool student_int8_has_avx2() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int bits[4]; __cpuid(bits, 1);
    if (!(bits[2] & (1 << 27)) || !(bits[2] & (1 << 28)) || (_xgetbv(0) & 6) != 6) return false;
    __cpuidex(bits, 7, 0); return (bits[1] & (1 << 5)) != 0;
#elif defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
std::shared_ptr<StudentInt8Weights> load_student_int8_weights(
    const assets::TensorSource & source, std::string_view name, int64_t input, int64_t output) {
    const std::string prefix(name);
    const auto shape = source.require_tensor_data(prefix + ".shape");
    int64_t dims[2]{};
    if (assets::ggml_type_for_tensor_dtype(shape.metadata.dtype) != GGML_TYPE_I64 || shape.bytes.size() != sizeof(dims))
        throw std::runtime_error("invalid student INT8 shape: " + prefix);
    std::memcpy(dims, shape.bytes.data(), sizeof(dims));
    if (dims[0] != output || dims[1] != input || input <= 0 || input > 65536 || output <= 0 || output > 65536)
        throw std::runtime_error("invalid student INT8 dimensions: " + prefix);
    const bool wide = source.has_tensor(prefix + ".packed8");
    if (wide && source.has_tensor(prefix + ".packed4")) throw std::runtime_error("mixed student codes: " + prefix);
    const auto raw = source.require_tensor_data(prefix + (wide ? ".packed8" : ".packed4"));
    if (assets::ggml_type_for_tensor_dtype(raw.metadata.dtype) != GGML_TYPE_I8 ||
        raw.bytes.size() != static_cast<size_t>(wide ? input * output : (input * output + 1) / 2))
        throw std::runtime_error("invalid student INT8 packed bytes: " + prefix);
    auto w = std::make_shared<StudentInt8Weights>();
    w->full_range = wide;
    w->input = input; w->output = output; w->stride = (input + 31) / 32 * 32;
    w->scales = source.require_f32(prefix + ".scale", {output});
    for (float s : w->scales) if (!std::isfinite(s) || s <= 0) throw std::runtime_error("invalid student INT8 scale");
    w->codes.resize(static_cast<size_t>(output * w->stride), 0);
    w->sums.resize(output, 0); w->avx2 = student_int8_has_avx2();
    for (int64_t o = 0; o < output; ++o) for (int64_t k = 0; k < input; ++k) {
        const auto i = o * input + k;
        const auto byte = std::to_integer<unsigned>(raw.bytes[wide ? i : i / 2]);
        const int8_t q = static_cast<int8_t>(wide ? (byte < 128 ? static_cast<int>(byte) : static_cast<int>(byte)-256)
            : static_cast<int>((byte >> (4 * (1 - i % 2))) & 15) - 8);
        w->codes[o * w->stride + k] = q; w->sums[o] += q;
    }
    return w;
}
namespace {
float quantize(const float * x, uint8_t * q, int64_t count, bool avx2) {
    if (avx2) return student_int8_quantize_avx2(x, q, count);
    float maximum = 1e-5f;
    for (int64_t k = 0; k < count; ++k) maximum = std::max(maximum, std::abs(x[k]));
    const float scale = 127.0f / maximum;
    for (int64_t k = 0; k < count; ++k)
        q[k] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::nearbyint(x[k] * scale)), -128, 127) + 128);
    return scale;
}
void quantize_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & w = *static_cast<const StudentInt8Weights *>(userdata);
    const auto * x = static_cast<const float *>(dst->src[0]->data);
    for (int64_t row = ith; row < dst->ne[1]; row += nth) {
        auto * q = static_cast<uint8_t *>(dst->data) + row * dst->nb[1];
        const float scale = quantize(x + row * w.input, q, w.input, w.avx2);
        std::fill(q + w.input, q + w.stride, uint8_t{128});
        std::memcpy(q + w.stride, &scale, sizeof(scale));
    }
}
void linear_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & w = *static_cast<const StudentInt8Weights *>(userdata);
    const auto * q = static_cast<const uint8_t *>(dst->src[0]->data);
    auto * y = static_cast<float *>(dst->data);
    const int64_t rows = dst->ne[1], out_tiles = (w.output + 3) / 4;
    const int64_t tiles = ((rows + 1) / 2) * out_tiles;
    // A 2-row x 4-output tile reuses both operands and distributes small batches.
    const int64_t begin = tiles * ith / nth, end = tiles * (ith + 1) / nth;
    for (int64_t tile = begin; tile < end; ++tile) {
        const int64_t row = (tile / out_tiles) * 2, first = (tile % out_tiles) * 4;
        const int nout = static_cast<int>(std::min<int64_t>(4, w.output - first));
        const auto * x0 = q + row * (w.stride + 4);
        const auto * x1 = row + 1 < rows ? x0 + w.stride + 4 : x0;
        float sx[2]; std::memcpy(&sx[0], x0 + w.stride, 4); std::memcpy(&sx[1], x1 + w.stride, 4);
        int32_t sums[8]{};
        if (w.avx2) student_int8_dot_tile_avx2(w, x0, x1, first, nout, sums);
        else for (int r = 0; r < 2; ++r) for (int o = 0; o < nout; ++o) {
            const auto * codes = w.codes.data() + (first + o) * w.stride;
            const auto * x = r ? x1 : x0;
            for (int64_t k = 0; k < w.stride; ++k) sums[r * 4 + o] += (static_cast<int>(x[k]) - 128) * codes[k];
        }
        for (int r = 0; r < 2 && row + r < rows; ++r) for (int o = 0; o < nout; ++o)
            y[(row + r) * w.output + first + o] = (static_cast<float>(sums[r * 4 + o]) / sx[r]) / w.scales[first + o];
    }
}
}
core::TensorValue build_student_int8_linear(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const StudentInt8Weights & w) {
    if (ctx.backend_type != core::BackendType::Cpu || input.type != GGML_TYPE_F32 || input.shape.last_dim() != w.input)
        throw std::runtime_error("student INT8 requires CPU F32 input with matching features");
    auto * flat = ggml_cont(ctx.ggml, input.tensor);
    auto * userdata = const_cast<StudentInt8Weights *>(&w);
    const auto rows = input.shape.prefix_elements();
    auto * q = ggml_custom_4d(ctx.ggml, GGML_TYPE_I8, w.stride + 4, rows, 1, 1,
        &flat, 1, quantize_op, GGML_N_TASKS_MAX, userdata);
    ggml_set_name(q, "kitten.student.int8.activation");
    auto * y = ggml_custom_4d(ctx.ggml, GGML_TYPE_F32, w.output, rows, 1, 1,
        &q, 1, linear_op, GGML_N_TASKS_MAX, userdata);
    ggml_set_name(y, "kitten.student.int8.linear");
    return core::reshape_tensor(ctx, core::wrap_tensor(y, core::TensorShape::from_dims({rows, w.output})),
        input.shape.with_last_dim(w.output));
}
codecs::s3gen::S3FlowLinearFactory make_student_int8_factory(std::shared_ptr<const assets::TensorSource> source) {
    struct State { size_t count = 0, codes = 0, original_f32 = 0; };
    auto state = std::make_shared<State>();
    return [source = std::move(source), state](std::string_view name, int64_t input, int64_t output) -> codecs::s3gen::S3FlowLinearOperation {
        if (!source->has_tensor(std::string(name) + ".packed4") && !source->has_tensor(std::string(name) + ".packed8")) return {};
        auto w = load_student_int8_weights(*source, name, input, output);
        ++state->count; state->codes += w->codes.size(); state->original_f32 += static_cast<size_t>(input * output) * 4;
        if (state->count == (source->has_tensor("flow.decoder.estimator.time_embed_mixer.weight.folded") ? 409 : 426)) {
            debug::timing_log_scalar("kitten_tts2.student.int8.linears", static_cast<double>(state->count));
            debug::timing_log_scalar("kitten_tts2.student.int8.codes_bytes", static_cast<double>(state->codes));
            debug::timing_log_scalar("kitten_tts2.student.int8.replaced_f32_bytes", static_cast<double>(state->original_f32));
            debug::timing_log_scalar("kitten_tts2.student.int8.avx2", w->avx2 ? 1 : 0);
        }
        return [w = std::move(w)](core::ModuleBuildContext & ctx, const core::TensorValue & x) {
            return build_student_int8_linear(ctx, x, *w);
        };
    };
}
}

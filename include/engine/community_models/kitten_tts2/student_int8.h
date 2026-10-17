#pragma once
#include "engine/framework/codecs/s3gen_runtime.h"
#include <cstdint>
#include <vector>

namespace engine::community_models::kitten_tts2 {
struct StudentInt8Weights {
    int64_t input, output, stride;
    std::vector<int8_t> codes;
    std::vector<int32_t> sums;
    std::vector<float> scales;
    bool avx2;
    bool full_range = false;
};
bool student_int8_has_avx2();
std::shared_ptr<StudentInt8Weights> load_student_int8_weights(
    const assets::TensorSource &, std::string_view name, int64_t input, int64_t output);
core::TensorValue build_student_int8_linear(core::ModuleBuildContext &,
    const core::TensorValue &, const StudentInt8Weights &);
codecs::s3gen::S3FlowLinearFactory make_student_int8_factory(
    std::shared_ptr<const assets::TensorSource>);
// AVX2 entry points are called only after a CPU/OS capability check.
float student_int8_quantize_avx2(const float * input, uint8_t * output, int64_t count);
void student_int8_dot_tile_avx2(const StudentInt8Weights &, const uint8_t * x0,
    const uint8_t * x1, int64_t first_output, int outputs, int32_t * result);
}

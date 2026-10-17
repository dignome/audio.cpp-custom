#pragma once
#include "engine/framework/codecs/s3gen_runtime.h"

namespace engine::community_models::kitten_tts2 {
std::shared_ptr<const assets::TensorSource> make_student_tensor_source(
    std::shared_ptr<const assets::TensorSource> source);
codecs::s3gen::S3GenConfig student_decoder_config();
codecs::s3gen::S3GenRuntimeOptions student_decoder_options(
    std::shared_ptr<const assets::TensorSource> packed_source, core::BackendType backend = core::BackendType::Cpu);
core::TensorValue quantize_student_activations(
    core::ModuleBuildContext & ctx, const core::TensorValue & input);
}

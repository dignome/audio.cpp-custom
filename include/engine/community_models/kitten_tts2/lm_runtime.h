#pragma once

#include "engine/framework/modules/transformers/causal_decoder_runtime.h"

namespace engine::community_models::kitten_tts2 {

// Kitten's attention window follows the live context, independently of reserved
// capacity. The shared decoder layers and KV-cache operations remain unchanged.
class LanguageModelRuntime {
public:
    LanguageModelRuntime(core::ExecutionContext & execution,
        modules::CausalDecoderConfig config, modules::CausalDecoderRuntimeWeights weights);
    ~LanguageModelRuntime();
    LanguageModelRuntime(const LanguageModelRuntime &) = delete;
    LanguageModelRuntime & operator=(const LanguageModelRuntime &) = delete;

    modules::CausalDecoderStepResult prefill_embeddings_into_cache(
        const std::vector<float> & embeddings, int64_t steps, int64_t cache_steps,
        int64_t chunk_steps, int64_t keep_prefix_steps = 0);
    // The caller guarantees that the retained embedding rows are unchanged.
    int64_t retainable_prefix_steps(int64_t steps, int64_t cache_steps,
        int64_t chunk_steps, int64_t keep_prefix_steps) const;
    modules::CausalDecoderStepResult decode_token(int32_t token);
    int64_t decode_cache_steps() const noexcept;
    int64_t decode_current_end() const noexcept;
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::kitten_tts2

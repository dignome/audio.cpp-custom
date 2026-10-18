#include "engine/community_models/kitten_tts2/lm_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/lookup_modules.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::community_models::kitten_tts2 {
namespace {
constexpr const char * trace_name = "kitten_tts2.lm";
constexpr size_t graph_arena_bytes = 16 * 1024 * 1024;

struct ContextDeleter {
    void operator()(ggml_context * p) const { if (p) ggml_free(p); }
};
struct BufferDeleter {
    void operator()(ggml_backend_buffer * p) const { if (p) ggml_backend_buffer_free(p); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr * p) const { if (p) ggml_gallocr_free(p); }
};

int64_t aligned_steps(int64_t steps) {
    if (steps <= 0 || steps > std::numeric_limits<int32_t>::max() - 255) {
        throw std::runtime_error("Kitten LM cache capacity is out of range");
    }
    return ((steps + 255) / 256) * 256;
}

int64_t padded_capacity(int64_t steps, int64_t capacity, int64_t keep, int64_t chunk) {
    return aligned_steps(std::max(capacity, keep + ((steps - keep + chunk - 1) / chunk) * chunk));
}
}  // namespace

class LanguageModelRuntime::Impl {
public:
    Impl(core::ExecutionContext & execution, modules::CausalDecoderConfig config,
        modules::CausalDecoderRuntimeWeights weights)
        : backend_(execution.backend()), backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)), config_(std::move(config)), weights_(std::move(weights)) {
        if (!backend_ || !weights_.lm_head || config_.stack.hidden_size <= 0 || config_.logits_size <= 0 ||
            config_.stack.layers <= 0 || static_cast<int64_t>(weights_.stack.layers.size()) != config_.stack.layers ||
            config_.logits_mode != modules::CausalDecoderLogitsMode::LastStep ||
            config_.stack.runtime.static_cache.update_mode != modules::DecoderStaticCacheUpdateMode::DirectSetRows) {
            throw std::runtime_error("Kitten LM requires last-token logits and a direct static cache");
        }
    }
    ~Impl() { release_runtime_graphs(); }

    int64_t retainable_prefix_steps(int64_t steps, int64_t capacity, int64_t chunk_steps, int64_t keep_steps) const {
        if (steps <= 0 || capacity < steps || chunk_steps <= 0 || keep_steps < 0) {
            throw std::runtime_error("Kitten LM prefill has invalid dimensions");
        }
        const auto keep = std::min(keep_steps, steps - 1);
        if (keep <= 0) return 0;
        const auto chunk = std::min(steps - keep, chunk_steps);
        if (cache_.cache_steps() < padded_capacity(steps, capacity, keep, chunk) || cache_.valid_steps() < keep) return 0;
        return keep;
    }

    modules::CausalDecoderStepResult prefill_embeddings_into_cache(const std::vector<float> & embeddings,
        int64_t steps, int64_t capacity, int64_t chunk_steps, int64_t keep_steps) {
        const auto keep = retainable_prefix_steps(steps, capacity, chunk_steps, keep_steps);
        const auto width = config_.stack.hidden_size;
        if (embeddings.size() != static_cast<size_t>(steps * width)) {
            throw std::runtime_error("Kitten LM prefill embedding size mismatch");
        }
        const auto chunk = std::min(steps - keep, chunk_steps);
        ensure_cache(padded_capacity(steps, capacity, keep, chunk));
        if (keep == 0) cache_.clear_on_backend();
        else cache_.retain_prefix(keep);
        debug::trace_log_scalar("kitten_tts2.lm.block_prefill.kept_steps", keep);
        std::vector<float> input(static_cast<size_t>(chunk * width));
        for (int64_t offset = keep; offset < steps; offset += chunk) {
            const auto count = std::min(chunk, steps - offset);
            ensure_graph(chunk, aligned_steps(offset + chunk), false);
            std::fill(input.begin(), input.end(), 0.f);
            std::copy_n(embeddings.data() + offset * width, count * width, input.data());
            const auto positions = modules::decoder_position_ids(chunk, offset);
            write_mask(offset, chunk);
            const int32_t last = static_cast<int32_t>(count - 1);
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
            ggml_backend_tensor_set(last_, &last, 0, sizeof(last));
            compute();
            // Padding writes future slots, which stay masked until overwritten.
            cache_.advance_after_direct_append(count);
        }
        return read_logits();
    }

    modules::CausalDecoderStepResult decode_token(int32_t token) {
        if (cache_.valid_steps() >= cache_.cache_steps()) {
            throw std::runtime_error("Kitten LM decode cache exhausted");
        }
        if (token < 0 || token >= weights_.token_embedding.shape.dims[0]) {
            throw std::runtime_error("Kitten LM token outside vocabulary");
        }
        const int32_t position = static_cast<int32_t>(cache_.valid_steps());
        ensure_graph(1, aligned_steps(position + 1), true);
        const int32_t last = 0;
        ggml_backend_tensor_set(input_, &token, 0, sizeof(token));
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(position));
        ggml_backend_tensor_set(last_, &last, 0, sizeof(last));
        write_mask(position, 1);
        compute();
        cache_.advance_after_direct_append(1);
        return read_logits();
    }

    int64_t decode_cache_steps() const noexcept { return cache_.cache_steps(); }
    int64_t decode_current_end() const noexcept { return cache_.current_end(); }

    void release_runtime_graphs() {
        release_graph();
        cache_ = runtime::TransformerKVCache();
        cache_buffer_.reset();
        cache_context_.reset();
    }

private:
    void ensure_cache(int64_t capacity) {
        if (cache_.cache_steps() >= capacity) return;
        release_runtime_graphs();
        const auto & stack = config_.stack;
        cache_context_.reset(ggml_init({static_cast<size_t>(2 * stack.layers) * ggml_tensor_overhead(), nullptr, true}));
        if (!cache_context_) throw std::runtime_error("Kitten LM cache context allocation failed");
        core::ModuleBuildContext ctx{cache_context_.get(), trace_name, backend_type_};
        std::vector<core::TensorValue> keys, values;
        const auto shape = core::TensorShape::from_dims({1, capacity, stack.num_key_value_heads, stack.head_dim});
        for (int64_t i = 0; i < stack.layers; ++i) {
            keys.push_back(core::make_tensor(ctx, config_.static_cache_type, shape));
            values.push_back(core::make_tensor(ctx, config_.static_cache_type, shape));
        }
        cache_buffer_.reset(ggml_backend_alloc_ctx_tensors(cache_context_.get(), backend_));
        if (!cache_buffer_) throw std::runtime_error("Kitten LM cache buffer allocation failed");
        runtime::TransformerKVCacheOptions options;
        options.allow_f16_storage = config_.static_cache_type == GGML_TYPE_F16;
        options.allow_bf16_storage = config_.static_cache_type == GGML_TYPE_BF16;
        options.lazy_import_scratch = true;
        cache_ = runtime::TransformerKVCache(capacity, stack.num_key_value_heads * stack.head_dim,
            std::move(keys), std::move(values), options);
        debug::trace_log_scalar("kitten_tts2.lm.decode.cache_steps", capacity);
    }

    void ensure_graph(int64_t chunk, int64_t attention_steps, bool token_input) {
        if (graph_ && chunk_ == chunk && attention_steps_ == attention_steps && token_input_ == token_input) return;
        release_graph();
        graph_context_.reset(ggml_init({graph_arena_bytes, nullptr, true}));
        if (!graph_context_) throw std::runtime_error("Kitten LM graph context allocation failed");
        core::ModuleBuildContext ctx{graph_context_.get(), trace_name, backend_type_};
        const auto & stack = config_.stack;
        core::TensorValue hidden;
        if (token_input) {
            auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({chunk}));
            input_ = ids.tensor;
            hidden = modules::EmbeddingModule({weights_.token_embedding.shape.dims[0], stack.hidden_size})
                .build(ctx, ids, weights_.token_embedding);
            hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({1, chunk, stack.hidden_size}));
        } else {
            hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, chunk, stack.hidden_size}));
            input_ = hidden.tensor;
        }
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({chunk}));
        auto mask = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, chunk, attention_steps}));
        positions_ = positions.tensor;
        mask_ = mask.tensor;
        last_ = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, 1);
        for (auto * tensor : {input_, positions_, mask_, last_}) ggml_set_input(tensor);
        graph_ = ggml_new_graph_custom(ctx.ggml, 65536, false);
        const modules::DecoderLayerModule layer(modules::decoder_layer_config_from_stack(stack));
        for (size_t i = 0; i < weights_.stack.layers.size(); ++i) {
            auto key = cache_.key_tensor(i);
            auto value = cache_.value_tensor(i);
            // 256-slot alignment keeps CUDA's optimized attention kernels available.
            // The view depends on live rows, never on max_tokens or earlier requests.
            if (attention_steps < cache_.cache_steps()) {
                key = runtime::view_transformer_kv_cache_steps(ctx, key, 0, attention_steps,
                    stack.num_key_value_heads, stack.head_dim, "Kitten active key", key.type);
                value = runtime::view_transformer_kv_cache_steps(ctx, value, 0, attention_steps,
                    stack.num_key_value_heads, stack.head_dim, "Kitten active value", value.type);
            }
            hidden = layer.build_with_static_cache_block(ctx, graph_, hidden, positions,
                weights_.stack.layers[i], key, value, positions, mask).output;
        }
        hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({chunk, stack.hidden_size}));
        // Select the last real row when the final prefill block contains padding.
        hidden = core::wrap_tensor(ggml_get_rows(ctx.ggml, hidden.tensor, last_),
            core::TensorShape::from_dims({1, 1, stack.hidden_size}), GGML_TYPE_F32);
        hidden = modules::RMSNormModule({stack.hidden_size, stack.rms_norm_eps, true, false})
            .build(ctx, hidden, weights_.final_norm);
        if (config_.lm_head_input_type) {
            hidden = core::wrap_tensor(ggml_cast(ctx.ggml, hidden.tensor, *config_.lm_head_input_type),
                hidden.shape, *config_.lm_head_input_type);
        }
        logits_ = modules::LinearModule({stack.hidden_size, config_.logits_size,
            config_.use_lm_head_bias, config_.lm_head_precision}).build(ctx, hidden, *weights_.lm_head).tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("Kitten LM graph allocation failed");
        }
        chunk_ = chunk;
        attention_steps_ = attention_steps;
        token_input_ = token_input;
        debug::trace_log_scalar("kitten_tts2.lm.attention.steps", attention_steps);
    }

    void write_mask(int64_t offset, int64_t chunk) {
        mask_values_.assign(static_cast<size_t>(chunk * attention_steps_),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t q = 0; q < chunk; ++q) {
            std::fill_n(mask_values_.begin() + q * attention_steps_, offset + q + 1, ggml_fp32_to_fp16(0.f));
        }
        ggml_backend_tensor_set(mask_, mask_values_.data(), 0, mask_values_.size() * sizeof(ggml_fp16_t));
    }

    void compute() {
        core::set_backend_threads(backend_, threads_);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Kitten LM graph compute failed");
        }
    }

    modules::CausalDecoderStepResult read_logits() {
        ggml_backend_synchronize(backend_);
        return {core::read_tensor_f32(logits_), {}};
    }

    void release_graph() {
        if (graph_) core::release_backend_graph_resources(backend_, graph_, true);
        allocator_.reset();
        graph_context_.reset();
        graph_ = nullptr;
        input_ = positions_ = mask_ = last_ = logits_ = nullptr;
        chunk_ = attention_steps_ = 0;
    }

    ggml_backend_t backend_;
    core::BackendType backend_type_;
    int threads_;
    modules::CausalDecoderConfig config_;
    modules::CausalDecoderRuntimeWeights weights_;
    std::unique_ptr<ggml_context, ContextDeleter> cache_context_, graph_context_;
    std::unique_ptr<ggml_backend_buffer, BufferDeleter> cache_buffer_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    runtime::TransformerKVCache cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * last_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    int64_t chunk_ = 0, attention_steps_ = 0;
    bool token_input_ = false;
    std::vector<ggml_fp16_t> mask_values_;
};

LanguageModelRuntime::LanguageModelRuntime(core::ExecutionContext & execution,
    modules::CausalDecoderConfig config, modules::CausalDecoderRuntimeWeights weights)
    : impl_(std::make_unique<Impl>(execution, std::move(config), std::move(weights))) {}
LanguageModelRuntime::~LanguageModelRuntime() = default;
modules::CausalDecoderStepResult LanguageModelRuntime::prefill_embeddings_into_cache(
    const std::vector<float> & embeddings, int64_t steps, int64_t capacity, int64_t chunk, int64_t keep) {
    return impl_->prefill_embeddings_into_cache(embeddings, steps, capacity, chunk, keep);
}
int64_t LanguageModelRuntime::retainable_prefix_steps(int64_t steps, int64_t capacity, int64_t chunk, int64_t keep) const {
    return impl_->retainable_prefix_steps(steps, capacity, chunk, keep);
}
modules::CausalDecoderStepResult LanguageModelRuntime::decode_token(int32_t token) { return impl_->decode_token(token); }
int64_t LanguageModelRuntime::decode_cache_steps() const noexcept { return impl_->decode_cache_steps(); }
int64_t LanguageModelRuntime::decode_current_end() const noexcept { return impl_->decode_current_end(); }
void LanguageModelRuntime::release_runtime_graphs() { impl_->release_runtime_graphs(); }

}  // namespace engine::community_models::kitten_tts2

#include "engine/community_models/kitten_tts2/lm_runtime.h"
#include "engine/framework/debug/trace.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
float largest_error = 0.f;
std::vector<float> pattern(size_t size, float scale) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) { result[i] = scale * std::sin(static_cast<float>(i) * .17f + .3f); }
    return result;
}
void close(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) { throw std::runtime_error("logit size mismatch"); }
    float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) { throw std::runtime_error("nonfinite logits"); }
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    engine::debug::trace_log_scalar("kitten_cache_test.max_error", maximum);
    largest_error = std::max(largest_error, maximum);
    if (maximum > .003f) { throw std::runtime_error("chunked prefill logits differ from reference"); }
}
}

int main(int argc, char ** argv) {
    try {
        engine::core::BackendConfig backend;
        backend.threads = 12;
        engine::debug::LoggingConfig logging;
        bool extended = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--backend" && i + 1 < argc) {
                const std::string value(argv[++i]);
                if (value == "cuda") { backend.type = engine::core::BackendType::Cuda; }
                else if (value == "cpu") { backend.type = engine::core::BackendType::Cpu; }
                else { throw std::runtime_error("unsupported test backend"); }
            } else if (arg == "--extended") {
                extended = true;
            } else if (arg == "--log" && i + 1 < argc) {
                logging.enabled = true;
                logging.file_path = argv[++i];
            } else { throw std::runtime_error("unknown test argument"); }
        }
        engine::debug::configure_logging(logging);
        engine::core::ExecutionContext execution(backend);
        auto * context = ggml_init({2 * 1024 * 1024, nullptr, true});
        if (!context) { throw std::runtime_error("weight context allocation failed"); }
        struct ContextGuard { ggml_context * p; ~ContextGuard() { ggml_free(p); } } context_guard{context};
        engine::core::ModuleBuildContext ctx{context, "kitten_cache_test", backend.type};
        std::vector<engine::core::TensorValue> tensors;
        auto tensor = [&](std::initializer_list<int64_t> shape) {
            auto value = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(shape));
            tensors.push_back(value);
            return value;
        };
        engine::modules::CausalDecoderRuntimeWeights weights;
        weights.token_embedding = tensor({97, 64});
        weights.final_norm = {tensor({64}), std::nullopt};
        weights.lm_head = engine::modules::LinearWeights{weights.token_embedding, std::nullopt};
        for (int layer = 0; layer < 2; ++layer) {
            engine::modules::DecoderLayerWeights w;
            w.input_norm = {tensor({64}), std::nullopt};
            w.post_norm = {tensor({64}), std::nullopt};
            w.q_norm = {tensor({64}), std::nullopt};
            w.k_norm = {tensor({64}), std::nullopt};
            w.self_attention.q_weight = tensor({128, 64});
            w.self_attention.k_weight = tensor({64, 64});
            w.self_attention.v_weight = tensor({64, 64});
            w.self_attention.out_weight = tensor({64, 128});
            w.mlp.gate_proj = {tensor({128, 64}), std::nullopt};
            w.mlp.up_proj = {tensor({128, 64}), std::nullopt};
            w.mlp.down_proj = {tensor({64, 128}), std::nullopt};
            weights.stack.layers.push_back(w);
        }
        auto * buffer = ggml_backend_alloc_ctx_tensors(context, execution.backend());
        if (!buffer) { throw std::runtime_error("weight buffer allocation failed"); }
        struct BufferGuard { ggml_backend_buffer_t p; ~BufferGuard() { ggml_backend_buffer_free(p); } } buffer_guard{buffer};
        for (const auto & value : tensors) {
            auto values = pattern(ggml_nelements(value.tensor), .04f);
            if (value.shape.rank == 1) { std::fill(values.begin(), values.end(), 1.f); }
            engine::core::write_tensor_f32(value, values);
        }
        engine::modules::CausalDecoderRuntimeConfig config;
        auto & stack = config.decoder.stack;
        stack.hidden_size = 64;
        stack.num_attention_heads = 2;
        stack.num_key_value_heads = 1;
        stack.head_dim = 64;
        stack.intermediate_size = 128;
        stack.layers = 2;
        stack.runtime.static_cache.update_mode = engine::modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.attention.prefill_mode = engine::modules::DecoderAttentionMode::FlashGrouped;
        stack.runtime.attention.static_mode = engine::modules::DecoderAttentionMode::FlashGrouped;
        config.decoder.logits_size = 97;
        config.decoder.static_cache_type = GGML_TYPE_F32;
        config.prefill_graph_arena_bytes = 4 * 1024 * 1024;
        config.decode_graph_arena_bytes = 4 * 1024 * 1024;
        config.evict_cuda_graph_cache_on_release = true;
        engine::modules::CausalDecoderRuntime reference(execution, config, weights);
        // Reserved capacity must not alter live attention or sampled-token logits.
        engine::community_models::kitten_tts2::LanguageModelRuntime active_small(execution, config.decoder, weights);
        engine::community_models::kitten_tts2::LanguageModelRuntime active_large(execution, config.decoder, weights);
        auto exact = [&](const std::vector<float> & a, const std::vector<float> & b) {
            close(a, b);
            if (a != b) { throw std::runtime_error("unused cache capacity changed active attention logits"); }
        };
        for (int steps : {250, 506, 250}) {
            const auto embeddings = pattern(steps * 64, .3f);
            const int capacity = steps < 256 ? 512 : 768;
            const auto expected = reference.prefill_embeddings(embeddings, steps);
            auto a = active_small.prefill_embeddings_into_cache(embeddings, steps, capacity, 64);
            auto b = active_large.prefill_embeddings_into_cache(embeddings, steps, capacity + 512, 64);
            exact(a.logits, b.logits);
            close(a.logits, expected.logits);
            reference.start_decode_tokens(expected.state, capacity);
            for (int step = 0; step < 20; ++step) {
                const int token = step + 7;
                a = active_small.decode_token(token);
                b = active_large.decode_token(token);
                exact(a.logits, b.logits);
                close(a.logits, reference.decode_token(token).logits);
            }
            if (active_small.decode_current_end() != steps + 20 || active_large.decode_current_end() != steps + 20) {
                throw std::runtime_error("active attention boundary changed cache position");
            }
            // Retained prefix after decode must match a fresh prompt computation.
            a = active_small.prefill_embeddings_into_cache(embeddings, steps, capacity, 64, 192);
            b = active_large.prefill_embeddings_into_cache(embeddings, steps, capacity + 512, 64, 192);
            exact(a.logits, b.logits);
            close(a.logits, expected.logits);
        }
        // Match Kitten's reference-first prefill, including a prefix not aligned
        // to the block size and the reported 1024/1280 reserved-cache boundary.
        active_small.release_runtime_graphs();
        active_large.release_runtime_graphs();
        const auto embeddings = pattern(306 * 64, .3f);
        const std::vector<float> prefix(embeddings.begin(), embeddings.begin() + 286 * 64);
        for (auto * lm : {&active_small, &active_large}) {
            const int capacity = lm == &active_small ? 1024 : 1280;
            lm->prefill_embeddings_into_cache(prefix, 286, capacity, 128);
            if (lm->retainable_prefix_steps(306, capacity, 128, 286) != 286) {
                throw std::runtime_error("reference prefix not retained");
            }
        }
        auto a = active_small.prefill_embeddings_into_cache(embeddings, 306, 1024, 128, 286);
        auto b = active_large.prefill_embeddings_into_cache(embeddings, 306, 1280, 128, 286);
        exact(a.logits, b.logits);
        const auto expected = reference.prefill_embeddings(embeddings, 306);
        close(a.logits, expected.logits);
        for (int i = 0; i < 12; ++i) exact(active_small.decode_token(i).logits, active_large.decode_token(i).logits);
        // Repeating after decode and after growth must match the cold result.
        auto warm = active_small.prefill_embeddings_into_cache(embeddings, 306, 1024, 128, 286);
        exact(warm.logits, a.logits);
        if (active_small.retainable_prefix_steps(306, 2048, 128, 286) != 0) {
            throw std::runtime_error("prefix retained across cache reallocation");
        }
        active_small.prefill_embeddings_into_cache(prefix, 286, 2048, 128);
        auto grown = active_small.prefill_embeddings_into_cache(embeddings, 306, 2048, 128, 286);
        exact(grown.logits, a.logits);
        auto reject = [](auto action) {
            try { action(); } catch (const std::runtime_error &) { return; }
            throw std::runtime_error("invalid cache operation accepted");
        };
        reject([&] { active_small.decode_token(-1); });
        reject([&] { active_small.decode_token(97); });
        reject([&] { active_small.prefill_embeddings_into_cache({}, 0, 256, 128); });
        reject([&] { active_small.prefill_embeddings_into_cache({}, 10, 256, 0); });
        reject([&] { active_small.prefill_embeddings_into_cache({}, 10, 256, 128); });
        active_small.release_runtime_graphs();
        if (active_small.decode_cache_steps() != 0 || active_small.decode_current_end() != 0) {
            throw std::runtime_error("cache release failed");
        }
        reject([&] { active_small.decode_token(1); });
        active_small.prefill_embeddings_into_cache(pattern(256 * 64, .3f), 256, 256, 128);
        reject([&] { active_small.decode_token(1); });
        if (extended) {
            size_t cases = 0;
            // Exercise both production blocks and blocks that do not divide the
            // attention window. Compare identical schedules across allocations;
            // compare different schedules to full prefill with a tolerance.
            for (int steps : {1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257,
                              383, 384, 385, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025}) {
                const auto input = pattern(steps * 64, .3f);
                const auto full = reference.prefill_embeddings(input, steps);
                for (int chunk : {31, 128}) {
                    active_small.release_runtime_graphs();
                    active_large.release_runtime_graphs();
                    auto small = active_small.prefill_embeddings_into_cache(input, steps, steps + 3, chunk);
                    auto large = active_large.prefill_embeddings_into_cache(input, steps, steps + 516, chunk);
                    exact(small.logits, large.logits);
                    close(small.logits, full.logits);
                    reference.start_decode_tokens(full.state, steps + 516);
                    for (int token : {0, 96, 7}) {
                        small = active_small.decode_token(token);
                        large = active_large.decode_token(token);
                        exact(small.logits, large.logits);
                        close(small.logits, reference.decode_token(token).logits);
                    }
                    if (active_small.decode_current_end() != steps + 3) {
                        throw std::runtime_error("boundary prefill/decode position mismatch");
                    }
                    ++cases;
                }
            }
            std::cout << "PASS " << cases << " prompt/block boundary cases" << std::endl;

            // A one-row embedding graph has the same dimensions as a token
            // graph, but its input type and operations must still be rebuilt.
            for (int chunk : {1, 3, 64, 127, 129, 256}) {
                const auto input = pattern(129 * 64, .3f);
                const auto full = reference.prefill_embeddings(input, 129);
                auto result = active_small.prefill_embeddings_into_cache(input, 129, 512, chunk);
                close(result.logits, full.logits);
                active_small.decode_token(96);
                result = active_small.prefill_embeddings_into_cache(input, 129, 512, chunk);
                close(result.logits, full.logits);
                ++cases;
            }
            for (int capacity : {255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 8191, 8192, 8193}) {
                active_small.release_runtime_graphs();
                active_large.release_runtime_graphs();
                const auto input = pattern(3 * 64, .3f);
                auto small = active_small.prefill_embeddings_into_cache(input, 3, capacity, 128);
                auto large = active_large.prefill_embeddings_into_cache(input, 3, capacity + 512, 128);
                exact(small.logits, large.logits);
                if (active_small.decode_cache_steps() != ((capacity + 255) / 256) * 256) {
                    throw std::runtime_error("reserved capacity rounding mismatch");
                }
                exact(active_small.decode_token(96).logits, active_large.decode_token(96).logits);
                ++cases;
            }

            const auto original = pattern(513 * 64, .3f);
            for (int keep : {0, 1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 514}) {
                active_small.release_runtime_graphs();
                active_large.release_runtime_graphs();
                active_small.prefill_embeddings_into_cache(original, 513, 1024, 128);
                active_large.prefill_embeddings_into_cache(original, 513, 1536, 128);
                const int retained = std::min(keep, 512);
                if (active_small.retainable_prefix_steps(513, 1024, 128, keep) != retained) {
                    throw std::runtime_error("retained prefix clamping mismatch");
                }
                auto changed = original;
                for (size_t i = retained * 64; i < changed.size(); ++i) changed[i] *= -.7f;
                const auto full = reference.prefill_embeddings(changed, 513);
                auto small = active_small.prefill_embeddings_into_cache(changed, 513, 1024, 128, keep);
                auto large = active_large.prefill_embeddings_into_cache(changed, 513, 1536, 128, keep);
                exact(small.logits, large.logits);
                close(small.logits, full.logits);
                reference.start_decode_tokens(full.state, 1536);
                auto decoded = active_small.decode_token(0);
                exact(decoded.logits, active_large.decode_token(0).logits);
                close(decoded.logits, reference.decode_token(0).logits);
                decoded = active_small.decode_token(96);
                exact(decoded.logits, active_large.decode_token(96).logits);
                close(decoded.logits, reference.decode_token(96).logits);
                ++cases;
            }
            // Asking to reuse rows that have not been computed must fall back
            // to a complete prefill, even if the allocation is large enough.
            active_small.release_runtime_graphs();
            active_small.prefill_embeddings_into_cache(pattern(3 * 64, .3f), 3, 1536, 128);
            if (active_small.retainable_prefix_steps(513, 1536, 128, 127) != 0) {
                throw std::runtime_error("unresident prefix retained");
            }
            close(active_small.prefill_embeddings_into_cache(original, 513, 1536, 128, 127).logits,
                reference.prefill_embeddings(original, 513).logits);
            ++cases;
            std::cout << "PASS " << cases << " prefill, input-kind, capacity and prefix cases" << std::endl;

            // Cross several windows through actual decode, including the final
            // writable slot. Exhaustion must leave the previous state intact.
            const auto input = pattern(255 * 64, .3f);
            active_small.release_runtime_graphs();
            active_large.release_runtime_graphs();
            const auto full = reference.prefill_embeddings(input, 255);
            active_small.prefill_embeddings_into_cache(input, 255, 1024, 128);
            active_large.prefill_embeddings_into_cache(input, 255, 1536, 128);
            reference.start_decode_tokens(full.state, 1536);
            for (int step = 255; step < 1024; ++step) {
                const int token = (step * 17) % 97;
                auto small = active_small.decode_token(token);
                exact(small.logits, active_large.decode_token(token).logits);
                close(small.logits, reference.decode_token(token).logits);
                if (active_small.decode_current_end() != step + 1) {
                    throw std::runtime_error("long decode position mismatch");
                }
            }
            reject([&] { active_small.decode_token(0); });
            if (active_small.decode_current_end() != 1024) throw std::runtime_error("exhaustion altered position");
            std::cout << "PASS 769 decode steps across 256/512/768/1024 boundaries and exhaustion" << std::endl;

            const auto one = pattern(64, .3f);
            auto clean = active_small.prefill_embeddings_into_cache(one, 1, 256, 1);
            reject([&] { active_small.prefill_embeddings_into_cache({}, -1, 256, 128); });
            reject([&] { active_small.prefill_embeddings_into_cache(one, 1, 0, 128); });
            reject([&] { active_small.prefill_embeddings_into_cache(one, 1, 256, -1); });
            reject([&] { active_small.prefill_embeddings_into_cache(one, 1, 256, 128, -1); });
            reject([&] { active_small.prefill_embeddings_into_cache(one, 1, std::numeric_limits<int32_t>::max(), 128); });
            reject([&] { active_small.prefill_embeddings_into_cache({}, 1, 256, 128); });
            reject([&] { active_small.decode_token(-1); });
            reject([&] { active_small.decode_token(97); });
            if (active_small.decode_current_end() != 1) throw std::runtime_error("invalid request altered position");
            exact(clean.logits, active_small.prefill_embeddings_into_cache(one, 1, 256, 1).logits);
            // End the next prefill with one row retained, then force the input
            // graph to alternate embeddings -> token -> embeddings -> token.
            const auto pair = pattern(2 * 64, .3f);
            auto pair_full = reference.prefill_embeddings(pair, 2);
            close(active_small.prefill_embeddings_into_cache(pair, 2, 256, 1, 1).logits, pair_full.logits);
            reference.start_decode_tokens(pair_full.state, 256);
            close(active_small.decode_token(0).logits, reference.decode_token(0).logits);
            close(active_small.prefill_embeddings_into_cache(pair, 2, 256, 1, 1).logits, pair_full.logits);
            reference.start_decode_tokens(pair_full.state, 256);
            close(active_small.decode_token(96).logits, reference.decode_token(96).logits);
            std::cout << "PASS invalid-request recovery and one-row graph transitions" << std::endl;
        }
        std::cout << "PASS Kitten live-window boundaries, reserved capacity, prefix reuse, growth, reset, errors and reference parity\n";
        std::cout << "Largest reference logit error: " << largest_error << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

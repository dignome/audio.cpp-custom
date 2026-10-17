#include <ggml.h>
#include <ggml-cpu.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
    try {
        auto * context = ggml_init({1024 * 1024, nullptr, false});
        if (!context) throw std::runtime_error("ggml initialization failed");
        ggml_cpu_init();
        const auto * tq = ggml_get_type_traits(GGML_TYPE_TQ2_1);
        const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_TQ2_1);
        const auto * act = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        if (ggml_type_size(GGML_TYPE_TQ2_1) != 68 || ggml_type_size(GGML_TYPE_Q8_K) != 292 ||
            cpu->vec_dot_type != GGML_TYPE_Q8_K || !cpu->vec_dot || !act->from_float ||
            !tq->from_float_ref || !tq->to_float)
            throw std::runtime_error("incorrect ternary type registration");
        std::mt19937 random(9876);
        std::uniform_int_distribution<int> code(-1, 1);
        std::normal_distribution<float> normal(0, 1);
        double max_normalized_error = 0;
        for (int n : {256, 512, 2048, 6144}) {
            for (int trial = 0; trial < 30; ++trial) {
                std::vector<float> weights(n), activation(n), restored(n), qa(n);
                for (int i = 0; i < n; ++i) {
                    const float scale = ((i / 128 + trial) % 9 == 0) ? 0.0f :
                        std::ldexp(1.0f, (i / 128 + trial) % 12 - 10);
                    weights[i] = code(random) * scale;
                    activation[i] = normal(random) * (trial % 3 == 0 ? 100000.0f : 1.0f);
                }
                std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_TQ2_1, n));
                std::vector<uint8_t> aq(ggml_row_size(GGML_TYPE_Q8_K, n));
                tq->from_float_ref(weights.data(), packed.data(), n);
                tq->to_float(packed.data(), restored.data(), n);
                if (weights != restored) throw std::runtime_error("ternary packing changed exact weights");
                if (!ggml_validate_row_data(GGML_TYPE_TQ2_1, packed.data(), packed.size()))
                    throw std::runtime_error("packed weights failed validation");
                act->from_float(activation.data(), aq.data(), n);
                // Q8_K is an intermediate activation type without a public to_float trait.
                for (int block = 0; block < n / 256; ++block) {
                    float scale;
                    std::memcpy(&scale, aq.data() + block * 292, sizeof(scale));
                    const auto * codes = reinterpret_cast<const int8_t *>(aq.data() + block * 292 + 4);
                    for (int j = 0; j < 256; ++j) qa[block * 256 + j] = scale * codes[j];
                }
                double expected = 0, magnitude = 0;
                for (int i = 0; i < n; ++i) {
                    const double product = double(weights[i]) * qa[i];
                    expected += product;
                    magnitude += std::abs(product);
                }
                float actual = 0;
                cpu->vec_dot(n, &actual, 0, packed.data(), 0, aq.data(), 0, 1);
                const double error = std::abs(actual - expected) / std::max(1.0, magnitude);
                if (!std::isfinite(actual) || error > 2e-6) {
                    std::cerr << "n=" << n << " trial=" << trial << " expected=" << expected
                              << " actual=" << actual << " error=" << error << std::endl;
                    throw std::runtime_error("native ternary dot product disagrees with scalar reference");
                }
                max_normalized_error = std::max(max_normalized_error, error);
                packed[0] |= 3;
                if (ggml_validate_row_data(GGML_TYPE_TQ2_1, packed.data(), packed.size()))
                    throw std::runtime_error("reserved ternary code was accepted");
            }
        }
        ggml_free(context);
        std::cout << "PASS: 120 exact packing/native dot-product cases, including large activations; "
                  << "max normalized error=" << max_normalized_error
                  << ", AVX2=" << ggml_cpu_has_avx2() << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

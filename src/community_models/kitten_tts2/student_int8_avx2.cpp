#include "engine/community_models/kitten_tts2/student_int8.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace engine::community_models::kitten_tts2 {
float student_int8_quantize_avx2(const float * x, uint8_t * q, int64_t count) {
#if defined(__AVX2__)
    const auto mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    auto vmax = _mm256_set1_ps(1e-5f);
    int64_t k = 0;
    for (; k + 8 <= count; k += 8) vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + k), mask));
    alignas(32) float parts[8]; _mm256_store_ps(parts, vmax);
    float maximum = 1e-5f;
    for (float v : parts) maximum = std::max(maximum, v);
    for (; k < count; ++k) maximum = std::max(maximum, std::abs(x[k]));
    const float scale = 127.0f / maximum; const auto vs = _mm256_set1_ps(scale);
    k = 0;
    for (; k + 8 <= count; k += 8) {
        const auto ints = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + k), vs));
        const auto words = _mm_packs_epi32(_mm256_castsi256_si128(ints), _mm256_extracti128_si256(ints, 1));
        const auto bytes = _mm_xor_si128(_mm_packs_epi16(words, words), _mm_set1_epi8(static_cast<char>(0x80)));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(q + k), bytes);
    }
    for (; k < count; ++k) q[k] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::nearbyint(x[k]*scale)), -128, 127) + 128);
    return scale;
#else
    (void)x; (void)q; (void)count; return 0; // Unreachable: runtime dispatcher rejects AVX2 here.
#endif
}
#if defined(__AVX2__)
template <bool full_range>
void dot_tile(const StudentInt8Weights & w, const uint8_t * x0,
    const uint8_t * x1, int64_t first, int outputs, int32_t * result) {
    auto a0=_mm256_setzero_si256(), a1=a0, a2=a0, a3=a0;
    auto b0=a0, b1=a0, b2=a0, b3=a0;
    const auto ones=_mm256_set1_epi16(1);
    auto dot = [ones](__m256i x, __m256i c) {
        if constexpr (full_range) {
            // maddubs saturates adjacent W8 products. Split the unsigned input
            // into 7-bit and high-bit parts; widen each sum before adding.
            const auto low = _mm256_and_si256(x, _mm256_set1_epi8(127));
            const auto high = _mm256_and_si256(x, _mm256_set1_epi8(static_cast<char>(0x80)));
            return _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_maddubs_epi16(low, c), ones),
                _mm256_madd_epi16(_mm256_maddubs_epi16(high, c), ones));
        } else {
            return _mm256_madd_epi16(_mm256_maddubs_epi16(x, c), ones);
        }
    };
    const int8_t * p[4];
    for (int i=0;i<4;++i) p[i]=w.codes.data()+(first+std::min(i,outputs-1))*w.stride;
    for (int64_t k=0;k<w.stride;k+=32) {
        const auto u=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(x0+k));
        const auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(x1+k));
        auto c=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p[0]+k));
        a0=_mm256_add_epi32(a0,dot(u,c));
        b0=_mm256_add_epi32(b0,dot(v,c));
        c=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p[1]+k));
        a1=_mm256_add_epi32(a1,dot(u,c));
        b1=_mm256_add_epi32(b1,dot(v,c));
        c=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p[2]+k));
        a2=_mm256_add_epi32(a2,dot(u,c));
        b2=_mm256_add_epi32(b2,dot(v,c));
        c=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p[3]+k));
        a3=_mm256_add_epi32(a3,dot(u,c));
        b3=_mm256_add_epi32(b3,dot(v,c));
    }
    auto horizontal=[](__m256i a) {
        auto sum=_mm_add_epi32(_mm256_castsi256_si128(a),_mm256_extracti128_si256(a,1));
        sum=_mm_hadd_epi32(sum,sum);sum=_mm_hadd_epi32(sum,sum);return _mm_cvtsi128_si32(sum);
    };
    const __m256i accum[8]={a0,a1,a2,a3,b0,b1,b2,b3};
    for(int r=0;r<2;++r) for(int o=0;o<outputs;++o)
        result[r*4+o]=horizontal(accum[r*4+o])-128*w.sums[first+o];
}
#endif
void student_int8_dot_tile_avx2(const StudentInt8Weights & w, const uint8_t * x0,
    const uint8_t * x1, int64_t first, int outputs, int32_t * result) {
#if defined(__AVX2__)
    if (w.full_range) dot_tile<true>(w, x0, x1, first, outputs, result);
    else dot_tile<false>(w, x0, x1, first, outputs, result);
#else
    (void)w;(void)x0;(void)x1;(void)first;(void)outputs;(void)result;
#endif
}
}

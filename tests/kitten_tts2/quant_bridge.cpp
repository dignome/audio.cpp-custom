#include <ggml.h>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define KITTEN_EXPORT extern "C" __declspec(dllexport)
#else
#define KITTEN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

KITTEN_EXPORT const char * kitten_quant_type_name(int type) {
    return ggml_type_name(static_cast<ggml_type>(type));
}

KITTEN_EXPORT size_t kitten_quant_row_size(int type, int64_t columns) {
    return ggml_row_size(static_cast<ggml_type>(type), columns);
}

KITTEN_EXPORT int kitten_quantize_decode(int type, const float * input, void * packed,
                                       float * restored, int64_t elements) {
    auto * context = ggml_init({1024 * 1024, nullptr, true});
    if (!context) return 0;
    const auto ggtype = static_cast<ggml_type>(type);
    const auto * traits = ggml_get_type_traits(ggtype);
    traits->from_float_ref(input, packed, elements);
    traits->to_float(packed, restored, elements);
    const bool valid = ggml_validate_row_data(ggtype, packed, ggml_row_size(ggtype, elements));
    ggml_free(context);
    return valid ? 1 : 0;
}

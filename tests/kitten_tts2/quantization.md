# NVFP4 and TQ2_1/student validation

Enable `ENGINE_BUILD_MODEL_TESTS` with `AUDIOCPP_MODELS=kitten_tts2`.
Build the following targets alongside the CLI/server:

```sh
cmake --build build-kitten2 --parallel 12 --target \
  kitten_tts2_ternary_probe kitten_tts2_student_int8_probe \
  kitten_tts2_student_probe kitten_tts2_session_probe \
  s3gen_input_transform_probe kitten_tts2_quant_bridge
```

## Kernel and shared S3 checks

```sh
build-kitten2/bin/kitten_tts2_ternary_probe
build-kitten2/bin/kitten_tts2_student_int8_probe /path/to/kitten-tts2-tq2_1-student-w4-multilingual.gguf
build-kitten2/bin/s3gen_input_transform_probe /path/to/s3gen_meanflow.safetensors cpu validation/s3-cpu
```

The ternary probe covers exact packing, reconstruction and 120 native dot-product
cases, including large activations. The INT8 probe checks all 426 student linears
against an independent integer reference, including zero/tiny inputs, rounding,
unaligned tails and the scalar fallback. The S3 probe checks both Conditional
and MeanFlow paths: identity hooks must produce exactly the default output,
selected transforms must take effect, and disabling flash attention must retain
finite, closely matching output. Run it with `cuda` in a CUDA build as well.

For upstream student component parity, use a Python environment with the local
KittenTTS `student_flow.py` dependencies:

```sh
python tests/kitten_tts2/check_student.py \
  --model /path/to/kitten-tts-2 --upstream /path/to/KittenTTS \
  --s3gen /path/to/s3gen_meanflow.safetensors \
  --probe build-kitten2/bin/kitten_tts2_student_probe --output validation/student
```

CPU student linears default to INT8. For a separate F32 reference comparison,
set `KITTEN_TTS2_STUDENT_LINEAR=f32` before running the component probe; unset it
afterwards. This switch affects only the student decoder. Normal deployment
requires no environment override.

## End-to-end and package checks

```sh
build-kitten2/bin/kitten_tts2_session_probe \
  /path/to/kitten-tts2-tq2_1-student-w4-multilingual.gguf validation/tq2 12 default /path/to/kitten-tts-2 cpu
build-kitten2/bin/kitten_tts2_session_probe \
  /path/to/kitten-tts2-nvfp4-multilingual.gguf validation/nvfp4 12 default /path/to/kitten-tts-2 cuda
```

The session probe checks stored precision, repeatability, preset switching,
chunking, cloning, cached conditioning and invalid input. Repeat with the existing
BF16/Q8/Q4 packages to cover the unchanged decoder path. TQ2_1 must reject CUDA
before allocating decoder weights. Use normal backend precision for session
checks; `NVIDIA_TF32_OVERRIDE=0` is only for controlled component comparisons.

Run `python tools/check_loader_catalog_sync.py`, inspect model-manager package
information, and rebuild the WebUI before building the server. The UI must expose
five choices, keep Q8 selected by default, and use separate install directories
for NVFP4 and TQ2_1. Generated frontend output is not included in this patch.

## Interpretation

Validation against upstream `977dc40a` on 2026-10-17:

- CPU and CUDA builds passed. The rebuilt WebUI passed type checking and its
  catalog exposed all five packages with Q8 as default.
- All 120 ternary cases passed; all 426 INT8 linears matched the integer
  reference exactly, including tail and scalar cases.
- Student component checks preserved all 474 packed weight matrices exactly.
  Relative L2 errors against upstream were 0.00914 for the encoder and 0.00895
  for mel, below the component probe's 0.02 tolerance.
- Conditional and MeanFlow S3 identity hooks produced exactly the default
  outputs on CPU and CUDA. Unfused attention remained within tolerance.
- TQ2_1 CPU and NVFP4 CUDA session checks passed preset/repeat/voice-switch,
  chunking, cloning/cache/reference-switch and invalid-input checks. The seven
  CPU student WAVs matched the earlier INT8 experiment byte for byte.
- Existing BF16/Q8/Q4 CUDA session checks passed. TQ2_1 rejected CUDA explicitly.
- The server with native model management enabled served the updated WebUI and
  produced identical seeded repeats using the final CPU and CUDA packages.
- Regenerated exports preserved all 2,856 NVFP4 and 3,804 student-package tensor
  payloads and all 48 voice entries from the tested experimental bundles.

These checks do not establish cross-platform performance or perceptual quality.

TQ2_1 body storage is exact relative to the supplied BF16 ternary weights; Q8_K
activation quantization and the Q8_0 embedding still affect inference. NVFP4
body reconstruction has about 2.3954% relative L2 error, which is not a speech
quality score. Both packages retain the existing Q8_0 embedding error.

Earlier controlled 12-thread CPU measurements on this machine found warm median
RTFs of 0.823, 0.873 and 0.999 for short, medium and long student-INT8 requests.
These are workload-specific results, not a guarantee of realtime performance.
Use seed 1234 and an explicit token budget from the first request when comparing
timings; cache allocation can otherwise differ. No new timing claim is based on
builds or validation running concurrently.

Earlier ASR checks covered preset and cloned speech, but chunked samples could
omit a sentence with both the default and student decoders. Multilingual
conditioning is retained; quality across every language and human listening
validation have not been established. The RTX 4060 Ti checks do not validate
RTX 5090 native FP4 performance or older NVIDIA cards.

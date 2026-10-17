# NVFP4 and TQ2_1 validation

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
  /path/to/kitten-tts2-tq2_1-default-multilingual.gguf validation/tq2-default 12 default /path/to/kitten-tts-2 cpu
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
seven choices, keep Q8 selected by default, and label the student packages
**GGUF TQ2_1 (CPU w4)** and **GGUF TQ2_1 (CPU w8)**, and the default decoder
**GGUF TQ2_1 (CPU)**. NVFP4 and the three TQ2_1 packages use separate install directories. Generated frontend output is not included in this patch.

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

## Default-decoder TQ2_1 follow-up (2026-10-17)

- Exported with `--type tq2_1 --decoder default`: 1,855,355,872 bytes,
  SHA-256 `4ffe596bfae906f2caf42b138aac85dac4e0dbff860480ebe28efe85127f9835`.
- Independently decoded all 196 TQ2_1 body matrices: 1,409,286,144 values
  matched the source BF16 checkpoint exactly. The Q8_0 embedding retains its
  existing quantization error.
- All 2,856 exported tensor payloads and all 48 default-decoder voice entries
  matched the package exercised through the server. The embedded configuration
  selects the default decoder; no student tensors are included.
- CPU server checks with 12 threads, seed 1234 and a 200-token limit from the
  first request passed preset repeatability, voice switching, non-silent chunked
  output, cloning repeatability, reference switching, and invalid-input rejection.
  The standalone Windows session probe exited before producing output, so these
  checks used the HTTP server instead.
- ASR screening found no word errors in the four distinct short preset/clone
  clips. The long chunked sample omitted one sentence and had 20% word error.
  This is a content-completeness limitation, not evidence of perceptual quality.
- WebUI type checking and production build passed. Catalog checks exposed all
  six packages, the requested CPU/CPU w4 labels, separate TQ2_1 directories and
  Q8 as default, without a forced Kitten weight-type override.

The rebuilt server served both labels, and synthesis from the final exported
GGUF matched the earlier checked output byte for byte.

No new performance comparison is claimed by these checks.

## Student W8 follow-up (2026-10-17)

The W8 source is the shipped frozen `cpp/student_w8/decoder.pt` export. The
companion W4 export and packed checkpoint provide a checked tensor-name mapping;
the frozen graphs must be identical. The one-step time-conditioning constants
remain folded. This does not reconstruct an unpublished unfrozen W8 checkpoint.

```sh
python tests/kitten_tts2/check_student_w8.py \
  --model /path/to/kitten-tts-2 \
  --s3gen /path/to/s3gen_meanflow.safetensors \
  --probe build-kitten2/bin/kitten_tts2_student_probe \
  --output validation/student-w8
build-kitten2/bin/kitten_tts2_student_int8_probe \
  /path/to/kitten-tts2-tq2_1-student-w8-multilingual.gguf
```

- GGUF: 1,437,030,016 bytes; SHA-256
  `e85faf1da56a50b9886647a2e6bd3e49fb3c9d427c655c7067d76fab4bf1c6a4`.
- All 3,770 tensor payloads were checked: 2,036 student tensors match the
  staged source; the other 1,734 tensors match the existing TQ2_1 package.
  The 196 ternary LM matrices retain their exact stored values.
- All 457 recovered student matrices reconstruct the exported F32 weights
  bit for bit. The package retains 17 folded time projections, executes 409
  linears with INT8 on CPU and expands 48 convolutions to F32.
- The integer-reference probe passed all 409 W8 and 426 W4 linears. Synthetic
  tests cover full-range positive/negative INT8 values that would saturate a
  single AVX2 `maddubs` pair sum, plus scalar and unaligned-tail paths.
- Against the frozen upstream graph with zero flow noise, mel relative L2
  error was 0.0162434 for F32 linears and 0.0162835 for INT8, within the existing
  0.02 component tolerance. These are execution differences, not added weight
  reconstruction error or audio quality scores.
- CPU server checks passed repeated preset and cloned requests, voice/reference
  switching, non-silent chunked output and invalid-input rejection. All 38
  shipped W8 voice entries retain their reference conditioning; the nine
  language presets and PreparedBruno use the existing native student policy.
  Automatic preparation also completed all 48 entries.
- ASR found no word errors in four distinct short preset/clone samples. The
  longer chunked sample omitted a sentence (17.5% word error), matching the
  known content-completeness limitation of the other packages.

These checks do not establish faster inference or better perceptual quality
than W4. W8 requires this updated native runtime and the CPU backend.

The final server served all three TQ2_1 labels. W4 and default-decoder seeded
outputs matched their prior baselines byte for byte; W8 matched its tested
output. WebUI type checking, production build and catalog sync passed.

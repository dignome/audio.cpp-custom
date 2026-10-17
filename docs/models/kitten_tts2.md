# Kitten TTS 2

Powered by Stellon Labs.

`kitten_tts2` supports offline multilingual TTS, preset voices, reference-audio cloning,
and chunked long-form speech. The complete inference path is native C++/GGML:
Qwen3, S3 meanflow, HiFT, S3 tokenizer, CAMPPlus, and XVectorSincNet.
Python, LibTorch, TorchScript, and ONNX Runtime are not runtime dependencies.

The family is experimental; a compatible community GGUF is available below.
CPU and NVIDIA CUDA are validated. Other GPU backends, streaming, automatic
reference transcription and the upstream text normalizer are not implemented
or validated by this port. TQ2_1 CPU packages offer the full default decoder,
`student_w4`, or the shipped one-step `student_w8` decoder export. Student
linears run as INT8 on CPU. The other packages use the full default decoder.

## Build

Shared S3 code is included through the `chatterbox` dependency. No Kitten-specific
build flag or Torch package is needed.

This port uses the fixes from [Chatterbox S3 encoder correctness PR #778](https://github.com/0xShug0/audio.cpp/pull/778)
(relative-shift padding and attention/feed-forward LayerNorm epsilon), which
merged before the Kitten integration. Current upstream includes both changes.
The original Kitten integration also added build registration, documentation
indexes and the WebUI catalog with its generated bundle. The updated weight
profiles below include a TQ2_1 CPU type and kernels in GGML, framework type
registration, and optional S3 projection hooks. The hooks preserve S3 defaults
and are enabled only by Kitten's student decoder. CUDA uses existing NVFP4 kernels.

```sh
cmake -S . -B build-kitten2 -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=kitten_tts2 -DENGINE_ENABLE_CUDA=OFF \
  -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build-kitten2 --config Release \
  --target audiocpp_cli audiocpp_server audiocpp_gguf audiocpp_kitten_tts2_prepare_voices --parallel 8
```

Visual Studio binaries are in `build-kitten2/bin/Release/`; single-configuration
builds use `build-kitten2/bin/`.
Run examples from the audio.cpp repository root and replace model/source path
placeholders with their actual locations. Commands use POSIX-shell `\` line
continuations; in PowerShell, join the lines or use backticks instead. Examples
using `audiocpp_cli` assume its binary directory is on `PATH`.

For NVIDIA, configure with `-DENGINE_ENABLE_CUDA=ON` and select
`--backend cuda` at runtime. The Kitten networks use the selected GGML backend;
reference resampling, sampling, and some small host computations remain on CPU.
Windows Release builds and preset/cloned speech were tested on a 16 GB RTX 4060 Ti
with CUDA Toolkit 13.3 and driver 610.88. Set `-DCMAKE_CUDA_ARCHITECTURES=89` when
building specifically for this card; choose the architecture for other cards.
CUDA 13.3 places its runtime DLLs in `CUDA_PATH/bin/x64`; ensure that directory
is on `PATH` when launching the executable on Windows.

The speaker's temporal max pooling uses a CUDA-supported 3x1 window. Inference
uses the existing backend precision and dispatch policy. For controlled CUDA
component comparisons, set `NVIDIA_TF32_OVERRIDE=0` before starting the native
probes, as described in the [test setup](../../tests/kitten_tts2/README.md).
This is a parity-test setting; ordinary synthesis is also tested without it.
See the [validation record](../../tests/kitten_tts2/validation.md)
for the current port's build, server/cloning checks, measured RTF and component
errors.

The WebUI shows **7 GB estimated VRAM** for the native Q8 package. An earlier
CUDA server run, before removing the shared precision edits, used a 238-character
passage and peaked approximately 4.7 GiB above the desktop baseline for preset
speech and 5.9 GiB for cloning. This measurement has not been repeated after
the cleanup. The estimate includes headroom; it is guidance, not a tested
minimum for every input or weight type.

## Download and preset speech

The ready-to-run [community packages](https://huggingface.co/dignome/kitten_tts2)
contain a native decoder, speaker encoders, and all 48 prepared voice
entries. **Q8 is the default and recommended package** and the initial WebUI
choice. BF16, hybrid Q4, NVFP4 and TQ2_1 CPU are also available; see
[weight profiles](#updated-weight-profiles).

```sh
python tools/model_manager_v2.py install kitten_tts2 --models-root models
```

The family ID resolves to `kitten_tts2_q8_v2`, which installs the GGUF and readable
license/NOTICE files under `models/kitten-tts2-v2/`. No Hugging Face token is
needed.

Direct download:
[kitten-tts2-q8-v2-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-q8-v2-multilingual.gguf)
(3,282,201,248 bytes, approximately 3.28 GB). Its SHA-256 is
`a9c14096dc9aac53ac38658ac41d0c296d0b5d24cfc26b1dd5a38e8dba0e2cfe`.
Separate model components or the original preset recordings are not needed at
runtime. Custom voice cloning still requires the reference clip and transcript
you supply.

```sh
audiocpp_cli --family kitten_tts2 \
  --model models/kitten-tts2-v2/kitten-tts2-q8-v2-multilingual.gguf \
  --backend cpu --threads 8 --task tts --voice-id Bruno \
  --request-option seed=1234 --text "Hello there. This is native Kitten speech." \
  --out hello.wav
```

### Upstream source assets

These assets are for direct safetensors loading, local conversion, or developer
validation. They are not additional downloads required by the published GGUF.

Use the upstream [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2)
layout, including the full `lm/model.safetensors`, LM configuration/tokenizer,
`config.json`, `cpp/default/voices.json`, and `speaker/model.safetensors`.
Preserve `LICENSE.md` and `speaker/LICENSE`.
For multilingual preset preparation, also keep `voices/voices.json` and its
referenced WAV/NPZ files. The upstream `cpp/default/decoder.pt` is needed only
for the optional PyTorch component comparisons described in the
[test guide](../../tests/kitten_tts2/README.md#validation-assets); the native
runtime and GGUF converter do not use it.

Add the official [S3 meanflow checkpoint](https://huggingface.co/ResembleAI/chatterbox-turbo/blob/main/s3gen_meanflow.safetensors)
as `native/s3gen_meanflow.safetensors`, and the
[Chatterbox MIT license](https://github.com/resemble-ai/chatterbox/blob/master/LICENSE)
as `native/LICENSE`. Validated decoder revision:
`749d1c1a46eb10492095d68fbcf55691ccf137cd`; SHA-256:
`d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f`.
The runtime never downloads assets.

The upstream custom `cpp/model-tq2_1.gguf` and packed ternary/emb4 safetensors
are not inputs to this loader. Use the full safetensors or convert to audio.cpp GGUF.

```sh
audiocpp_cli --family kitten_tts2 --model /path/to/kitten-tts-2 \
  --backend cpu --threads 8 --task tts --voice-id Bruno \
  --request-option seed=1234 --text "Hello there. This is native Kitten speech." \
  --out hello.wav
```

The complete native package contains all 47 upstream voices and `PreparedBruno`.
Output is mono 24 kHz audio, written with audio.cpp's existing WAV support.

## Languages

Use the language-named preset for Arabic, Chinese, French, German, Hindi,
Italian, Portuguese, Russian, or Spanish. The 38 named English voices remain
available. These ten languages are registered in the model spec.

```sh
audiocpp_cli --family kitten_tts2 --model models/kitten-tts2-v2/kitten-tts2-q8-v2-multilingual.gguf \
  --backend cpu --task tts --voice-id German \
  --text "Guten Morgen. Dies ist ein Test der deutschen Sprachausgabe." --out german.wav
```

The server accepts `"voice": "German"` (or another language preset) with UTF-8
text in `input`. A reference clip and transcript in the target language can also
be supplied for cloning. There is no separate language switch: the voice
reference supplies the accent and pronunciation context.

Text reaches the original Qwen tokenizer without English normalization, matching
upstream's recommendation to use `normalize=False` for other languages. Write
numbers, dates and abbreviations as spoken words in the target language.
The shared chunker preserves Unicode codepoints and supports CJK punctuation.

Upstream's README advertises 20 languages, but its published voice index provides
presets for English and the nine languages above. The remaining ten languages
have no shipped language presets in this snapshot and are not validated here.
Generation and tokenization checks do not establish pronunciation quality;
see the validation record for exactly what was exercised.

## Voice cloning

Supply a one-to-thirty-second reference clip and its transcript:

```sh
audiocpp_cli --family kitten_tts2 --model models/kitten-tts2-v2/kitten-tts2-q8-v2-multilingual.gguf \
  --backend cpu --threads 8 --task clon --voice-ref reference.wav \
  --reference-text "The exact words spoken in the reference clip." \
  --text "The new words to say in this voice." --request-option seed=1234 \
  --out cloned.wav
```

Cloning also works in TTS sessions. The C++ request accepts `voice.speaker.audio`
or `audio_input`, with `reference_text` in request options. Reference audio
takes precedence over a preset ID. The server speech endpoint accepts
`voice_ref` (path or base64) and `reference_text`.

Channels are averaged. Native sinc resampling supplies 16 kHz speaker/tokenizer
inputs and the 24 kHz decoder reference. The identity encoder uses the complete
clip. Decoder conditioning repeats short clips to at least six seconds and takes
at most ten seconds for the default decoder. The student decoder uses up to
four seconds and 112 decoder prompt tokens. Speaker identity and LM reference
conditioning still use their existing policies.
Upstream uses librosa/soxr for some S3 preprocessing, so reference tokens need not
be byte-identical with this native sinc resampler.

Reference encoders load on the first clone request. A session caches its last
clip and transcript; changing either invalidates conditioning and the LM prefix.
This in-memory cache does not register persistent voice IDs.

## Sampling and long-form text

Stable sampling defaults to temperature 0.8, top-p 0.8, top-k 50.
`preset=expressive` uses temperature/top-p 0.9. Explicit request options override
the preset; temperature zero selects greedy decoding.

The sampler uses the upstream 50-token repetition window, silence/end exemptions,
and token-run penalty. The tied head projects only speech codes and the two end
markers, deliberately excluding stray text tokens. Native RNG and arithmetic
differ from PyTorch CPU; equal seeds do not promise Python-identical waveforms.

Text defaults to 380-codepoint chunks in `tag_aware` mode. Each chunk reuses the
reference prefix and starts fresh target generation. Joins trim edge silence,
apply short fades, and insert 160 ms gaps. Expression tags reach the model
verbatim and enable its emotion prefix. Expand numbers/abbreviations when needed.

LM tensors retain their stored types by default. An explicit
`--session-option kitten_tts2.weight_type=...` changes projection storage only;
the embedding and tied output head retain their stored type. The decoder and
reference encoders use F32.

## Updated weight profiles

Use the `kitten_tts2` model family and leave `kitten_tts2.weight_type` unset
(or select `native`) to retain the GGUF's stored precision. The loader reads
tensor types directly; no custom export selector is embedded in the package.
The WebUI sends no Kitten precision override by default.

| Export `--type` | Download filename | Source LM | Body / tied embedding | Approximate complete size |
| --- | --- | --- | --- | --- |
| `bf16` | [kitten-tts2-bf16-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-bf16-multilingual.gguf) | `model.safetensors` | BF16 / BF16 | 4.60 GB |
| `q8_0` | [kitten-tts2-q8-v2-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-q8-v2-multilingual.gguf) | `model-ternary.safetensors` | Q8_0 / BF16 | 3.28 GB |
| `q4_0` | [kitten-tts2-q4-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-q4-multilingual.gguf) | `model.safetensors` | Q4_0 / Q8_0 | 2.27 GB |

`install kitten_tts2` selects Q8. Install a selected package with:

```sh
python tools/model_manager_v2.py install kitten_tts2_bf16 --models-root models
python tools/model_manager_v2.py install kitten_tts2_q8_v2 --models-root models
python tools/model_manager_v2.py install kitten_tts2_q4 --models-root models
```

All three packages install into `models/kitten-tts2-v2/`, with distinct GGUF
filenames. They can coexist and share the license/NOTICE files. Select the
chosen GGUF path in the CLI/server. Each package also downloads `LICENSE`, `NOTICE`,
`native/LICENSE`, `speaker/LICENSE`, and `SHA256SUMS` from the same repository.

The BF16, Q8 and Q4 profiles retain the existing default S3 MeanFlow decoder,
reference encoders and multilingual voice conditioning. They do not use the
student decoder or upstream TL2/emb4 embedding format.

The Q8 and Q4 body matrices preserve the updated BF16 checkpoint's ternary
values exactly in standard GGML blocks. Q4 is a mixed-precision export: its
embedding is quantized directly from full BF16 to Q8_0, with approximately
0.44% relative L2 weight error overall and 0.41% across speech-token rows.
These are reconstruction errors, not audio quality scores. Norms and speaker
projections retain BF16; S3 and reference components retain their original types.
Hybrid Q4 remains experimental: local ASR checks found missing words in longer
CPU samples. See the [validation record](../../tests/kitten_tts2/validation.md)
for speech-quality checks and limitations.

On CUDA, Q4 feed-forward down projections are loaded as BF16 to avoid
non-finite results with this model's large activations. Their values stay exact;
this adds about 483 MiB of resident weights compared with retaining those
projections as Q4. CPU keeps the Q4 body. File size is not a RAM/VRAM estimate.

Updated conversion requires NumPy, PyTorch and safetensors offline only.
The full `lm/model.safetensors` is needed for every updated profile to refresh
the preset speaker projections from the shipped NPZ identities. The existing
decoder conditioning and `PreparedBruno` alias are retained.

The exporter defaults to `--type q8_0`. Each type selects its source checkpoint
and tensor storage recipe, including the embedding precision shown above.

```sh
python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /path/to/updated-kitten-tts-2 --converter build-kitten2/bin/audiocpp_gguf \
  --type bf16 --output kitten-tts2-bf16-multilingual.gguf
python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /path/to/updated-kitten-tts-2 --converter build-kitten2/bin/audiocpp_gguf \
  --type q8_0 --output kitten-tts2-q8-v2-multilingual.gguf
python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /path/to/updated-kitten-tts-2 --converter build-kitten2/bin/audiocpp_gguf \
  --type q4_0 --output kitten-tts2-q4-multilingual.gguf
```

Use `--s3gen` and `--s3-license` when the original default decoder assets are
stored separately.

## NVFP4 and TQ2_1 CPU packages

| WebUI choice | Package ID | Download filename | Decoder |
| --- | --- | --- | --- |
| GGUF NVFP4 | `kitten_tts2_nvfp4` | [kitten-tts2-nvfp4-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-nvfp4-multilingual.gguf) | Default MeanFlow |
| GGUF TQ2_1 (CPU) | `kitten_tts2_tq2_1_default` | [kitten-tts2-tq2_1-default-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-tq2_1-default-multilingual.gguf) | Default MeanFlow |
| GGUF TQ2_1 (CPU w8) | `kitten_tts2_tq2_1_w8` | [kitten-tts2-tq2_1-student-w8-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-tq2_1-student-w8-multilingual.gguf) | Student W8, INT8 linears on CPU |
| GGUF TQ2_1 (CPU w4) | `kitten_tts2_tq2_1` | [kitten-tts2-tq2_1-student-w4-multilingual.gguf](https://huggingface.co/dignome/kitten_tts2/resolve/main/kitten-tts2-tq2_1-student-w4-multilingual.gguf) | Student W4, INT8 linears on CPU |

These packages install into `models/kitten-tts2-nvfp4/`,
`models/kitten-tts2-tq2_1-default/`, `models/kitten-tts2-tq2_1-w8/`, and
`models/kitten-tts2-tq2_1/` respectively. All four retain the Q8_0 embedding, BF16
norms and speaker projections, reference encoders and 48 prepared multilingual
voice entries. Leave `kitten_tts2.weight_type` unset or `native`.

For WebUI downloads, configure with `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON`
(or `-NativeModelManager` in `scripts/build_windows.ps1`) and start the server
with `--ui --ui-management`. Regenerate the frontend before building the server:

```sh
npm --prefix webui/native ci
npm --prefix webui/native run build
cmake --build build-kitten2 --config Release --target audiocpp_server
```

NVFP4 stores all 196 LM body projections in GGML NVFP4 (type 40), uses existing
CUDA kernels, and does not apply the Q4 package's BF16 down-projection fallback.
It is about 2.27 GB. Testing on an RTX 4060 Ti does not establish performance
on other NVIDIA generations.

TQ2_1 preserves the supplied BF16 ternary body values exactly in 68-byte blocks
of 256 weights. It uses **type ID 44 in this branch**; Kitten's own ID 43 conflicts
with audio.cpp's I2_S. The default-decoder bundle is about 1.86 GB; the student
W4 bundle is about 1.41 GB. Both require this branch's CPU support. Select
`--backend cpu`; GPU loading returns a clear error. CPU dot products have AVX2
and scalar implementations. The default package retains the full S3 MeanFlow
decoder for both preset speech and cloning. In the CPU w4 package, the 474 student
matrices remain packed W4 on disk. At load time, 426 linears expand to INT8;
48 convolutions use F32. INT8 is selected automatically on CPU. Preset speech
and reference-audio cloning both use the student decoder.

To export from the upstream assets, build `kitten_tts2_quant_bridge` with
`-DENGINE_BUILD_MODEL_TESTS=ON` in addition to the converter and voice preparer.
The shared library is an offline conversion helper, not an inference dependency.
Use `.dll` on Windows or `libkitten_tts2_quant_bridge.so` on Linux:

```sh
python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /absolute/path/kitten-tts-2 --converter build-kitten2/bin/audiocpp_gguf \
  --ggml-library build-kitten2/bin/libkitten_tts2_quant_bridge.so \
  --type nvfp4 --output kitten-tts2-nvfp4-multilingual.gguf

python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /absolute/path/kitten-tts-2 --converter build-kitten2/bin/audiocpp_gguf \
  --ggml-library build-kitten2/bin/libkitten_tts2_quant_bridge.so \
  --type tq2_1 --decoder student_w4 --output kitten-tts2-tq2_1-student-w4-multilingual.gguf

python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /absolute/path/kitten-tts-2 --converter build-kitten2/bin/audiocpp_gguf \
  --ggml-library build-kitten2/bin/libkitten_tts2_quant_bridge.so \
  --type tq2_1 --decoder default --output kitten-tts2-tq2_1-default-multilingual.gguf
```

All TQ2_1 packages use `lm/model.safetensors`. `--decoder default` uses the
original S3 MeanFlow assets and `cpp/default/voices.json`. `--decoder student_w4`
also requires `decoders/student_w4.pt` and uses `cpp/student_w4/voices.json`.
Omitting `--decoder` retains the student decoder for TQ2_1; other export types
default to the original decoder. Missing presets are prepared with the selected
decoder's reference policy, and `--prepared-voices` must match that decoder.
All exports still need the original S3/HiFT/tokenizer assets (`--s3gen`) and
license (`--s3-license`) if absent from the model checkout. The exporter validates
TQ2_1 body reconstruction exactly and preserves student codes and scale divisors
when packaging that decoder.

### Student W8 export

`--type tq2_1 --decoder student_w8` produces
`kitten-tts2-tq2_1-student-w8-multilingual.gguf` (about 1.44 GB). Use the latest
build of this branch; the W4/default-only build does not recognize W8.
The exporter reads `cpp/student_w8/decoder.pt` and its `voices.json`, plus the
companion `cpp/student_w4/decoder.pt` and `decoders/student_w4.pt` to identify
frozen tensor names. Both frozen graphs must match structurally.

The shipped W8 export has already folded its fixed one-step time conditioning.
Those constants are preserved. All 457 remaining quantized matrices are recovered
as per-channel INT8 only when their reconstructed F32 values exactly match the
export. CPU inference uses 409 INT8 linears; 48 convolutions expand to F32.
The wider INT8 kernel avoids saturating pair sums. W4 keeps its existing kernel.
This is not a reconstruction of an unpublished, unfrozen W8 checkpoint.

The package retains all 38 shipped W8 presets and prepares the nine language
presets plus `PreparedBruno` with the student reference policy. Cloning uses
the same policy. TorchScript is read during export/validation only; inference
requires no Python or LibTorch. Controlled component checks and short speech
checks passed, but longer chunked output can still omit sentences. No speed or
quality advantage over W4 is established by these checks.

The TQ2_1 reference codec and CPU dot product are adapted from
KittenML/kitten-tts-2-cpp commit `1ce0bb5`, under the MIT license retained in
`external/ggml/LICENSE`. See [quantization validation](../../tests/kitten_tts2/quantization.md)
for checks and limitations.

## Self-contained GGUF

The conversion commands above produce complete runtime packages. They require
the native converter and `audiocpp_kitten_tts2_prepare_voices` beside it.

The upstream prepared `cpp/default/voices.json` omits the nine language presets;
the helper fills those from `voices/voices.json`, the NPZ embeddings/tokens, and
reference WAVs. It refreshes every preset speaker projection with the updated
full LM while preserving decoder conditioning. These preparation dependencies
are not needed for inference.

Preparation uses CPU by default; `--voice-backend cuda` selects a CUDA-enabled
preparer. `--voice-preparer` can select an executable elsewhere. Save the complete
index with `--voices-output voices.json`, then reuse it with
`--prepared-voices voices.json` on later conversions. Incomplete indexes are
rejected so a multilingual spec cannot silently ship English-only voices.
For direct safetensors loading, install that complete index as
`MODEL/cpp/default/voices.json`; the original upstream index alone lacks the
language presets. The GGUF embeds it and needs no adjacent reference clips.

`--s3gen` and `--s3-license` can select assets outside the model directory.
All three tensor sources, preset conditioning, configuration, tokenizer, spec,
licenses, and NOTICE are packaged. No `.pt` file is embedded.
The package supports cloning without adjacent model files. Use the GGUF path as
`--model`.

Weights retain the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md),
including commercial registration/limits and attribution requirements.
Decoder and speaker components retain their respective upstream terms.
No model weights are committed with the integration.

See [validation instructions](../../tests/kitten_tts2/README.md) and
[measured results](../../tests/kitten_tts2/validation.md).

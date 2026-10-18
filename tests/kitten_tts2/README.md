# Kitten TTS 2 validation

Powered by Stellon Labs.

Run commands from the audio.cpp repository root. Replace
`/absolute/path/kitten-tts-2` with the local upstream model directory and
`/absolute/path/KittenTTS` with the upstream source checkout. They need not be
siblings of audio.cpp. Point GGUF arguments at the downloaded package or your
own converted file.

Commands below use POSIX-shell `\` line continuations. In PowerShell, join
the lines or replace those continuations with backticks. Environment-variable
setup is shown separately for both shells.

Build per the [model guide](../../docs/models/kitten_tts2.md), adding
`-DENGINE_BUILD_TESTS=ON`. Build `kitten_tts2_session_probe`,
`kitten_tts2_components_probe`, and `model_spec_system_test`.
The [validation record](validation.md)
includes the Windows configure/build commands and results for the current port.
On Visual Studio, use `bin/Release/` and append `.exe` below.

[Chatterbox S3 encoder correctness PR #778](https://github.com/0xShug0/audio.cpp/pull/778)
and the Kitten model PR #776 are included in the upstream base for this follow-up.
No additional correctness patch is needed on that base. The validation record
also retains the earlier pre-merge results and their prerequisite.

```sh
python tools/check_loader_catalog_sync.py
build-kitten2/bin/model_spec_system_test
```

## Optional text normalization

Build and run `kitten_tts2_text_test` with `ENGINE_BUILD_MODEL_TESTS=ON`.
Its normalization cases cover explicit `en`/`zh`, unchanged empty/auto/other
language inputs, shared English number normalization, Chinese numbers and
dates, expression markup and placeholder collisions, UTF-8 validation, expanded
chunk budgets, and composition with `sentence_pauses`.

For session checks, use a fixed voice, seed, preset, token budget and chunk size.
Compare `normalize_text=true` plus `language=en` with manually expanded English
text and normalization disabled; do the same for `zh`. Check that an omitted
option matches `false`, that empty/auto/other languages keep the original text,
and that a per-request false override wins over a server true default. These
checks should use identical waveforms for equivalent model inputs. Include a
request combining expression tags and sentence pauses.

## Optional sentence pauses

This option can help when the model ends speech after an early sentence and
omits later sentences in the same chunk. Use a reproducible omission case to
check whether the enabled output includes the previously missing sentences.

Build and run `kitten_tts2_text_test` with `ENGINE_BUILD_MODEL_TESTS=ON`.
It needs no GGUF and checks period boundaries, common English abbreviations,
decimals, ellipses, closing quotes, markup, existing pause tags and UTF-8 input.
It also checks that other punctuation and bare newlines stay unchanged, no
trailing pause is added, and the framework chunk count stays the same.

Compare `sentence_pauses=false` and `sentence_pauses=true` with the same
input, voice, seed, preset, token budget and chunk settings on CPU and CUDA.
Also compare automatic insertion with manually inserting ` <pause>` after
internal sentence-ending periods. Enable it in a model's
`default_request_options`, then check the per-request false override.
A single sentence must retain its previous output. `<pause>` enables
expression conditioning and can change pacing; check completeness separately
from cache invariance. This is a mitigation, not a guarantee of complete speech.

## Validation assets

For NVFP4, TQ2_1 and the student decoder, see [quantization validation](quantization.md).

### Cache-size regression

With `-DENGINE_BUILD_MODEL_TESTS=ON`, build and run the model-local cache test:

```sh
cmake --build build-kitten2 --target kitten_tts2_cache_test
build-kitten2/bin/kitten_tts2_cache_test --backend cpu
# For a CUDA-enabled build:
build-kitten2/bin/kitten_tts2_cache_test --backend cuda
```

The test uses small synthetic weights, so no GGUF is required. It checks exact
logit agreement across reserved capacities, live attention boundaries at 256
and 512 slots, reference-prefix reuse, cache growth, reset and invalid inputs.
It also compares against the shared decoder's independent full-prefill path.

Add `--extended` on either backend to sweep prompt lengths immediately below,
at and above 128-token block and 256-slot attention boundaries through 1,025
tokens. This also checks nondividing block sizes, one-row embedding/token graph
transitions, prefix clamping and suffix replacement, reserved capacities
through 8,193 slots, 769 consecutive decode steps through a full cache, and
recovery after invalid inputs. Different reserved capacities must produce
identical logits; the independent full-prefill reference uses a 0.003 absolute
logit tolerance because its graph shapes and attention reductions differ.

For speech regression, use Bruno, the stable preset, seed 1234, and:

> The morning sun fills the room with light. A small bird sings outside the window.

Compare fresh loads at `max_tokens` 200, 400 and 600 on CPU, and 718 and 719 on
CUDA. Repeat the 200-token request after another voice's 1,000-token request has
enlarged the cache. Within each backend and GGUF, the WAVs should match exactly.
Check speech completeness separately: the model can emit its end-of-speech token
before speaking all of the text, even when cache invariance passes. Also check a
longer prompt that crosses an active attention boundary. Token limits below the
generated speech length can still truncate output normally; this check requires
sufficient generation room.

For the 306-token Bruno prompt above, also compare `max_tokens` 206/207,
462/463, 718/719 and 974/975, then 8,192 followed by 200 in the same session.
Check rejected limits (0, -1 and 8,193) followed by another valid request.
Compare chunk sizes 80, 81 and 82 at budgets 200 and 1,000: each chunk size
must be invariant to the budget, although different chunking can legitimately
produce different speech. Check a longer single chunk at sufficient budgets
to exercise additional live attention windows. Limits immediately below, at
and above the generated speech-token count distinguish normal truncation from
cache-size sensitivity.

### Updated BF16, Q8 and hybrid Q4

Build `kitten_tts2_session_probe` with `-DENGINE_BUILD_MODEL_TESTS=ON` in both
CPU-only and CUDA builds. The probe checks all three stored precision combinations. Pass
`default` as its weight-type argument to exercise the package's
default precision policy, followed by the matching upstream asset directory
and `cpu` or `cuda`. Run the existing multilingual server checks with the
exporter's complete `voices-default.json` and the updated reference assets.

Also run with `native`, the default advertised by the current external model
contract. For each GGUF, it should match the same seeded run with the option
omitted. Run from the repository root and check the contract trace points to
`model_specs/kitten_tts2.json` so this exercises external-contract precedence,
not only the embedded GGUF fallback.

`python tools/model_manager_v2.py info kitten_tts2 --json` should select
`kitten_tts2_q8_v2`. The rebuilt WebUI should offer **GGUF Q8**,
**GGUF Q4** and **GGUF BF16**, with Q8 as the default
package. Selecting any of them must leave `kitten_tts2.weight_type` unset unless
the user explicitly supplied an override. Regenerate the frontend bundle before
building the server after changing these catalog entries:

```sh
npm --prefix webui/native ci
npm --prefix webui/native run build
cmake --build build-kitten2 --target audiocpp_server
```

Replace `build-kitten2` with the configured server build directory. CMake embeds
the generated `webui/native/dist/index.html` already on disk and does not invoke
the frontend build.

Offline checks require NumPy, PyTorch and safetensors:

```sh
python -m unittest discover -s tests/kitten_tts2 -p 'test_*.py'
python tests/kitten_tts2/check_updated_weights.py \
  --model /path/to/updated-kitten-tts-2 --output updated-weights.json
python tests/kitten_tts2/check_hybrid_gguf.py \
  --model /path/to/updated-kitten-tts-2 --gguf /path/to/kitten-tts2-q4-multilingual.gguf \
  --output hybrid-tensors.json
```

The source check requires all 196 body matrices to match the full BF16 values
before and after standard Q8_0/Q4_0 packing. The hybrid package check verifies
every LM tensor's stored values, a Q8_0 embedding, and embedding relative L2
below 0.5% against full BF16, both overall and on speech-token rows. Other dense
LM tensors must remain exact. This does not imply identical generated audio
across quantizations or backends.

### Default decoder

The published GGUF is sufficient for native preset inference and, with a user
reference clip/transcript, cloning. The developer tests below additionally use:

- **Session cloning checks:** the upstream `voices/voices.json` index and its
  Bruno/Bella reference WAVs. Without the optional reference-model argument,
  the session probe runs only the preset and invalid-voice checks.
- **Component parity:** `speaker/model.safetensors`, the LM speaker-projection
  weights, `native/s3gen_meanflow.safetensors`, `cpp/default/voices.json`, and
  the upstream **`cpp/default/decoder.pt`** TorchScript export. That export is
  a Python comparison reference only; the native decoder does not load it.
- **Multilingual checks:** the original voice index, referenced WAV/NPZ files,
  and a separate complete prepared voice JSON saved during conversion as shown
  below.

Follow the model guide for the additional S3 checkpoint. The validated upstream
snapshot/revisions are recorded in [validation.md](validation.md); component
graph extraction relies on the matching default `decoder.pt` export.
Python component comparisons use NumPy, SoundFile, PyTorch, and safetensors.
These are validation-only dependencies.

## Session regression

```sh
build-kitten2/bin/kitten_tts2_session_probe \
  models/kitten-tts2-v2/kitten-tts2-q8-v2-multilingual.gguf build-kitten2/native-session \
  8 native /absolute/path/kitten-tts-2 cpu > build-kitten2/session.log 2>&1
```

One session checks exact seeded repeats, voice switching, three-chunk synthesis,
unknown voice rejection, cloning from Bruno/Bella WAVs, cloned-voice cached
repeats, reference changes, and missing-transcript/short-clip rejection.
The optional reference model directory supplies WAVs/transcripts; inference
weights come from the first argument. A final `cpu` or `cuda` argument selects
the backend (default: `cpu`). Output WAVs are float32, mono, 24 kHz.
Reported RTF and the framework's `session.wall_ms` cover the entire `run()` call,
including reference conditioning and first-clone lazy encoder loading. Initial
model/session construction is outside this timer.

The production model has no custom tensor-dump environment variable or hooks.
Normal framework timing/profiling remains available. `check_parity.py` is retained
only for comparing archived prompt/logit traces from the original port; it cannot
capture fresh traces from this runtime. Those historical comparisons additionally
require Transformers and the full upstream LM weights/tokenizer/configuration.

## Native components

For controlled **CUDA parity**, disable TF32 before launching the component
probe. Setting it only for the later Python comparison cannot change captured
tensors. This is a test setting; ordinary synthesis uses the default backend
precision policy.

```sh
export NVIDIA_TF32_OVERRIDE=0
```

```powershell
$env:NVIDIA_TF32_OVERRIDE = "0"
```

```sh
build-kitten2/bin/kitten_tts2_components_probe \
  /absolute/path/kitten-tts-2/native/s3gen_meanflow.safetensors \
  /absolute/path/kitten-tts-2/cpp/default/voices.json \
  tests/kitten_tts2/component_codes.json \
  build-kitten2/native-components cpu /absolute/path/kitten-tts-2
python tests/kitten_tts2/check_native_components.py \
  --model /absolute/path/kitten-tts-2 --upstream /absolute/path/KittenTTS \
  --speaker-output build-kitten2/native-components \
  --decoder-output build-kitten2/native-components \
  --codes tests/kitten_tts2/component_codes.json
```

The optional reference-model directory makes the test-only probe exercise
the public speaker API with Bruno's reference WAV. It writes the resampled
input with the framework WAV writer and embedding/projection vectors as JSON.
The script checks XVectorSincNet using that exact input, then the BF16 speaker
projection independently. The fixed 72-token fixture was captured from a seeded
Bruno request during the original port; both decoder implementations receive
the same fixture plus the usual three lookahead tokens.
The script compares encoder hidden states and two-step
meanflow mel with the published decoder's Gaussian draws fixed to zero.
Finally, it supplies identical mel and zero random draws to HiFT and checks
pitch and waveform. The graph extraction rejects incompatible exports.
Replace `cpu` with `cuda` in the component probe command to check NVIDIA execution. Use the
same numerical comparison limits on CPU and CUDA, with TF32 disabled for the
controlled CUDA run as described above.

The encoder comparison fails with the original relative-shift padding order.
The shift fix alone passes its tolerance; using the upstream LayerNorm epsilon
further reduces the error (see the ablation in [PR #778](https://github.com/0xShug0/audio.cpp/pull/778)).
Max-error limits are 0.00005 for speaker identity, 0.0001 for encoder hidden
states, 0.005 for mel, 0.032 for BF16 projection, and 0.01 for pitch/waveform.

## Server and dependencies

Run ordinary synthesis/server checks without the parity override, in a fresh
shell or after `unset NVIDIA_TF32_OVERRIDE` (POSIX) /
`Remove-Item Env:NVIDIA_TF32_OVERRIDE -ErrorAction SilentlyContinue` (PowerShell).

Register the GGUF as `kitten_tts2`, task `tts`, mode `offline`.
POST to `/v1/audio/speech` with a preset `voice` or reference audio:

```json
{
  "model": "kitten-tts2",
  "input": "Hello from native voice cloning.",
  "voice_ref": {"type": "path", "path": "/absolute/path/reference.wav"},
  "reference_text": "The words spoken in that recording.",
  "seed": 1234
}
```

Inspect executable imports (`dumpbin /dependents` on Windows) to verify no
Torch/Python/ONNX DLL. Inspect the GGUF for its three tensor namespaces,
license/NOTICE sidecars, and absence of TorchScript resources.

CPU and NVIDIA CUDA are tested backends. Student decoders, streaming, exhaustive
testing of every English voice, and automatic transcript generation are outside
the implemented/validated scope.
Numerical checks do not measure intelligibility or speaker similarity;
listen to generated samples as well.

## Multilingual presets

Build `audiocpp_kitten_tts2_prepare_voices` alongside the converter. If you already
saved the complete prepared index during packaging, pass its path to `--voices`
below. Otherwise, create a local GGUF and retain its index with:

```sh
python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /absolute/path/kitten-tts-2 \
  --converter build-kitten2/bin/audiocpp_gguf --type q8_0 \
  --output build-kitten2/kitten-tts2-q8-v2-multilingual.gguf \
  --voices-output build-kitten2/kitten-tts2-multilingual-voices.json
```

Use the updated upstream assets, including both `lm/model.safetensors` and
`lm/model-ternary.safetensors`. The helper completes the voice index from WAV/NPZ
assets, refreshes preset speaker projections with the updated full LM, and
rejects a package missing any registered preset. The following test uses that
local GGUF; substitute the downloaded GGUF path to validate the
published package with the same complete index.

```sh
python tests/kitten_tts2/check_multilingual.py \
  --server build-kitten2/bin/audiocpp_server \
  --model build-kitten2/kitten-tts2-q8-v2-multilingual.gguf \
  --reference-model /absolute/path/kitten-tts-2 \
  --voices build-kitten2/kitten-tts2-multilingual-voices.json \
  --output build-kitten2/multilingual-validation --backend cpu
```

Use `--backend cuda` and the CUDA server to repeat on NVIDIA. This starts and
stops a temporary localhost server. NumPy and SoundFile are used only by the test.
It checks all nine language presets plus Bruno, 48 registered voices, exact
prepared reference tokens/transcripts against upstream assets, finite non-silent
24 kHz output, Chinese chunking, and German/Chinese cloning. WAVs and a JSON
report remain in the output directory. Without runtime dump hooks this test
does not inspect internal BPE IDs, logits, or generated token counts. These are
execution checks, not an ASR or native-speaker pronunciation assessment.

# inflectsynth

InflectSynth is the text/frontend layer for Inflect v2 synthesis: `inflectg2p` produces model-ready token IDs, while ONNXVoice owns the split ONNX Runtime sessions for Inflect Nano v2 and Inflect Micro v2.

The package has a flat layout (no `src/`) and dynamic Git-tag versioning through
`setuptools-scm`.

## Install

CPU:

```bash
python -m pip install "inflectsynth[cpu]"
```

CUDA or DirectML:

```bash
python -m pip install "inflectsynth[gpu]"
python -m pip install "inflectsynth[directml]"
```

## Python

```python
from inflectsynth import DEFAULT_MODEL, InflectVoice

with InflectVoice.from_pretrained(DEFAULT_MODEL, providers="cpu") as tts:
    result = tts.synthesize(
        "A small voice can still have something meaningful to say.",
        speed=1.0,
        variation=0.667,
        seed=7,
    )
    result.save_wav("sample.wav")
```

Use `micro-v2` (or alias `micro`) for the larger v2 model.

The first managed run downloads exactly two revision-pinned upstream artifacts,
`duration.onnx` and `decode.onnx`, and verifies both byte size and SHA-256 before opening.

## Voice contract

The upstream Inflect v2 base releases each contain **one fixed synthetic English voice**.
Accordingly:

```python
tts.available_voices == ("default",)
```

`voice="default"` is the only supported runtime voice. There are no hidden style rows,
speaker embeddings, or reference-audio voice cloning inputs in the published v2 ONNX ABI.

## Controls

The public upstream ranges are preserved:

- `speed`: `0.5` to `2.0`, default `1.0`
- `variation`: `0.0` to `1.0`, default `0.667`
- `seed`: integer, default `0`

A call processes one input through one frontend/runtime request. InflectSynth does not currently implement the official runner's long-text wrapper behavior: punctuation-aware chunking, per-chunk pauses/noise, concatenation, or edge fades.

## Atomic request ownership

One `InflectVoice.synthesize_prepared()` call processes one exact prepared-text request. The caller owns text boundaries, subdivision, and any composition of child requests. InflectSynth does not split or concatenate long text, add pauses, or perform sentence segmentation.

## Request capacity and token measurement

`InflectVoice.measure_prepared()` runs the same InflectG2P prepared-text frontend used by synthesis and reports the actual model-token count without acoustic inference:

```python
from inflectsynth import DEFAULT_MODEL, InflectVoice

with InflectVoice.from_pretrained(DEFAULT_MODEL, providers="cpu") as tts:
    measure = tts.measure_prepared("The exact prepared text to send.")
    print(measure.amount, measure.unit, measure.maximum, measure.fits)
```

This release does not publish a supported maximum. Measurements therefore report the token `amount`, with `maximum=None` and `fits=None`; callers must not infer a supported limit from the count alone, model name, or an ONNX graph. A local or unmanaged model also reports unknown capacity unless a future explicit policy is supplied.

If a future release ships a validated model-specific capacity policy, `maximum` will be a **supported safe request budget** and `fits` will state whether the measured request is within that budget. Such a budget is not an ONNX graph maximum unless authoritative runtime evidence establishes that distinction. Capacity guidance is also separate from speech-quality guidance: passing a supported budget does not guarantee a particular perceptual quality, and quality may degrade gradually rather than at a hard graph boundary.

For debugging, the optional CLI measurement command prints the same fields as JSON and does not synthesize audio:

```bash
python -m inflectsynth --measure "The exact prepared text to send."
```

## Local graphs

```python
from inflectsynth import InflectVoice

with InflectVoice.from_local(
    duration_path="duration.onnx",
    decode_path="decode.onnx",
    providers="cpu",
) as tts:
    result = tts.synthesize("Local inference.")
```

## Voice-level calibration

InflectSynth ships measured BS.1770 static-gain calibration for both managed v2 models. Calibration is **opt-in**; the default mode remains `off`, so existing synthesis returns the raw model level.

```python
from inflectsynth import DEFAULT_MODEL, InflectVoice, SynthesisConfig, VoiceLevelConfig

config = SynthesisConfig(
    voice_level=VoiceLevelConfig(mode="calibrated"),
)

with InflectVoice.from_pretrained(DEFAULT_MODEL, providers="cpu") as tts:
    result = tts.synthesize(
        "Use the packaged voice-level calibration.",
        config=config,
    )
```

The v0.1.0 calibration catalog contains:

- `inflect:nano-v2:default`: `-0.0423775698 dB`
- `inflect:micro-v2:default`: `-0.3353773193 dB`

Both records target `-24.0 LUFS` using the `inflectsynth-prepared-speech-v1` corpus and nine measurements per model (three prepared-speech stimuli × three seeds). The applied gain and catalog revision are reported under `result.metadata["voice_level"]`. Local/unmanaged graphs do not receive a managed calibration identity automatically.

## Catalog

Catalog discovery is delegated to ONNXVoice. Its Inflect adapter uses the authoritative `inflect-onnx-bundles` catalog source and cache; InflectSynth does not package a second model catalog. For sibling-repository testing or a separately updated catalog, pass `catalog_url="../inflect-onnx-bundles/catalog/models.json"` to `InflectVoice.from_pretrained()` or the corresponding discovery API. The authoritative catalog repo is intentionally not a Python package.

## Runtime ownership

InflectSynth owns text preparation and calls `inflectg2p` to produce the model-ready token IDs. It passes those IDs and the speed, variation, and seed controls to ONNXVoice's registered Inflect adapter. ONNXVoice owns catalog resolution, pinned artifact integrity checks, provider/session configuration, and ONNX graph execution; InflectSynth does not open ONNX Runtime sessions directly.

## Real-model release smoke

The real CPU integration smoke is excluded from normal unit runs and downloads model files only when explicitly enabled. Install the candidate ONNXVoice checkout/wheel and InflectSynth's CPU test dependencies, then run from this repository root:

```bash
python -m pip install -e "../onnxvoice[cpu]" -e ".[cpu,dev]"
env INFLECTSYNTH_RUN_REAL_MODEL_TESTS=1 python -m pytest -s -m "integration and network" tests/integration/test_inflect_real_models.py
```

The test runs both `nano-v2` and `micro-v2` through real InflectG2P and ONNXVoice CPU inference, checks output/seed behavior, and prints the resolved revision, artifact sizes and SHA-256 digests, and runtime/package versions. Run this gate against the candidate ONNXVoice build before release.

## Dynamic versioning

Versions come from Git tags through `setuptools-scm`; source archives without Git metadata use
`0.1.dev0`. `inflectsynth.__version__` reports the installed distribution version.

## License

Apache-2.0. Inflect code and weights referenced by this project are published by Owen Song under
Apache-2.0; model weights are downloaded from their official Hugging Face repositories and are not
included in this source package.

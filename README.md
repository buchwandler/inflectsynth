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
from inflectsynth import InflectVoice

with InflectVoice.from_pretrained("nano-v2", providers="cpu") as tts:
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

## Catalog

A bootstrap copy of the catalog is packaged for out-of-box use. For sibling-repository testing or a separately updated catalog, pass `catalog_url="../inflect-onnx-bundles/catalog/models.json"` to `InflectVoice.from_pretrained()` or the corresponding discovery API. The authoritative catalog repo is intentionally not a Python package.

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

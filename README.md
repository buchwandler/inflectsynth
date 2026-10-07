# inflectsynth

ONNX Runtime synthesis for **Inflect-Nano-v2** and **Inflect-Micro-v2**, backed by
`inflectg2p` and the pinned catalog format in `inflect-onnx-bundles`.

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

Long text uses the same punctuation-aware 280-character chunking, pause policy, seeded
per-chunk noise, and 5 ms edge fades as the official ONNX runner.

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

A bootstrap copy of the catalog is packaged for out-of-box use. For sibling-repository testing
or a separately updated catalog, pass `catalog_path="../inflect-onnx-bundles/catalog/models.json"`.
The authoritative catalog repo is intentionally not a Python package.

For this MVP, `inflectsynth` executes the two ONNX graphs directly through ONNX Runtime rather than
requiring an unpublished OnnxVoice Inflect adapter. `inflect-onnx-bundles/docs/onnxvoice-integration.md`
documents the future adapter boundary.

## Dynamic versioning

Versions come from Git tags through `setuptools-scm`; source archives without Git metadata use
`0.1.dev0`. `inflectsynth.__version__` reports the installed distribution version.

## License

Apache-2.0. Inflect code and weights referenced by this project are published by Owen Song under
Apache-2.0; model weights are downloaded from their official Hugging Face repositories and are not
included in this source package.

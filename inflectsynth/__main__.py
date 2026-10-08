"""Command-line synthesis and request-measurement tools."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .catalog import list_models
from .constants import DEFAULT_MODEL, DEFAULT_VOICE
from .voice import InflectVoice


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize speech with Inflect v2 ONNX models")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--catalog-url")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--measure",
        metavar="TEXT",
        help="report prepared-text model-token measurement as JSON without acoustic inference",
    )
    parser.add_argument("-o", "--output", default="inflect.wav")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        if args.measure is not None:
            raise SystemExit("--measure cannot be combined with --list-models")
        for model in list_models(catalog_url=args.catalog_url):
            print(f"{model.id}\t{model.name}\tvoices={','.join(v.id for v in model.voices)}")
        return 0

    if args.measure is not None:
        if args.text is not None:
            raise SystemExit("use either positional text or --measure, not both")
        with InflectVoice.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            catalog_url=args.catalog_url,
            offline=args.offline,
            force_download=args.force_download,
            providers=args.provider,
        ) as engine:
            measurement = engine.measure_prepared(args.measure)
        print(
            json.dumps(
                {
                    "model": measurement.model_id,
                    "amount": measurement.amount,
                    "maximum": measurement.maximum,
                    "unit": measurement.unit,
                    "fits": measurement.fits,
                },
                sort_keys=True,
            )
        )
        return 0

    if not args.text:
        raise SystemExit("text is required unless --list-models or --measure is used")
    with InflectVoice.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        catalog_url=args.catalog_url,
        offline=args.offline,
        force_download=args.force_download,
        providers=args.provider,
    ) as engine:
        result = engine.synthesize(
            args.text,
            voice=args.voice,
            speed=args.speed,
            variation=args.variation,
            seed=args.seed,
        )
        result.save_wav(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

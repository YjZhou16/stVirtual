from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .audit import inspect_training_config
from .config import TrainConfig
from .infer import decode_h5ad
from .train import train_decoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stVirtual_decoder",
        description="Train and run a generic raw-count negative-binomial decoder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate H5AD keys, raw-count scale, samples, and normalization."
    )
    inspect_parser.add_argument("--config", type=Path, required=True)

    train_parser = subparsers.add_parser("train", help="Train one decoder configuration.")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--device")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--checkpoint-name")
    train_parser.add_argument("--max-blocks-per-split", type=int)
    train_parser.add_argument("--skip-weight", type=float)
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use two epochs, a small model, and one block per sample/split.",
    )

    decode_parser = subparsers.add_parser(
        "decode", help="Decode one H5AD latent frame to NB expected counts."
    )
    decode_parser.add_argument("--input", type=Path, required=True)
    decode_parser.add_argument("--output", type=Path, required=True)
    decode_parser.add_argument("--checkpoint", type=Path, required=True)
    decode_parser.add_argument("--latent-key", required=True)
    decode_parser.add_argument(
        "--latent-is-normalized", action=argparse.BooleanOptionalAction, default=True
    )
    decode_parser.add_argument(
        "--output-key", default="layers/reconstructed_expression"
    )
    decode_parser.add_argument("--normalization-checkpoint", type=Path)
    decode_parser.add_argument("--device", default="cuda:0")
    decode_parser.add_argument("--batch-size", type=int, default=256)
    decode_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        report = inspect_training_config(TrainConfig.from_json(args.config))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.command == "train":
        config = TrainConfig.from_json(args.config).with_overrides(
            device=args.device,
            epochs=args.epochs,
            checkpoint_name=args.checkpoint_name,
            max_blocks_per_split=args.max_blocks_per_split,
            skip_weight=args.skip_weight,
            overwrite=True if args.overwrite else None,
        )
        if args.smoke_test:
            config = config.with_overrides(
                output_dir=config.output_dir / "smoke",
                epochs=2,
                patience=2,
                max_blocks_per_split=1,
                overwrite=args.overwrite,
            )
        checkpoint = train_decoder(config)
        print(f"decoder checkpoint: {checkpoint}")
        return
    if args.command == "decode":
        output = decode_h5ad(
            input_path=args.input,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            latent_key=args.latent_key,
            latent_is_normalized=args.latent_is_normalized,
            output_key=args.output_key,
            normalization_checkpoint=args.normalization_checkpoint,
            device_name=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(f"decoded expected counts: {output}")
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()

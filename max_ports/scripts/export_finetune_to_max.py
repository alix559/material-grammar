#!/usr/bin/env python3
"""CLI wrapper around :func:`matgram.checkpoint_export.export_finetune`."""

from __future__ import annotations

import argparse
from pathlib import Path

from matgram.checkpoint_export import (
    DEFAULT_BASE_ASSETS,
    DEFAULT_OUT_ROOT,
    TASK_META,
    export_finetune,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="IBM finetune .pt with MODEL_STATE",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_META),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: model_assets/smi-ted-{task}",
    )
    parser.add_argument(
        "--base-assets",
        type=Path,
        default=DEFAULT_BASE_ASSETS,
        help="Dir with pretrained config.json + bert_vocab_curated.txt",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    out = export_finetune(
        checkpoint=args.checkpoint,
        task=args.task,
        output_dir=args.output_dir,
        base_assets=args.base_assets,
    )
    print(f"MAX property assets ready: {out}")
    print(
        "Serve with:\n"
        f"  max serve --model-path {out} "
        "--custom-architectures ./materials_smi_ted "
        "--quantization-encoding float32"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

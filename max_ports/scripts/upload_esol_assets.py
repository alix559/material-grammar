#!/usr/bin/env python3
"""Upload exported ESOL MAX assets to a Hugging Face model repo.

Uploads only the files Railway needs (not the local finetune_source.pt symlink):

  model_weights.safetensors
  config.json
  bert_vocab_curated.txt  (materialized if currently a symlink)

Example::

  export HF_TOKEN=hf_...
  pixi run python scripts/upload_esol_assets.py --repo YOUR_USER/smi-ted-esol
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS = ROOT / "model_assets" / "smi-ted-esol"
UPLOAD_NAMES = (
    "model_weights.safetensors",
    "config.json",
    "bert_vocab_curated.txt",
)


def materialize_upload_dir(source: Path) -> Path:
    """Copy uploadable files into a temp dir; resolve symlinks to real files."""
    missing = [n for n in UPLOAD_NAMES if not (source / n).exists()]
    if missing:
        raise SystemExit(
            f"missing {missing} under {source}; export first "
            "(POST /load with the ESOL checkpoint, or export_finetune_to_max.py)"
        )

    staging = Path(tempfile.mkdtemp(prefix="smi-ted-esol-upload-"))
    for name in UPLOAD_NAMES:
        src = (source / name).resolve()
        if not src.is_file():
            raise SystemExit(f"not a file after resolve: {source / name} -> {src}")
        shutil.copy2(src, staging / name)
        print(f"staged {name} ({(staging / name).stat().st_size} bytes)")
    return staging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("HF_REPO") or os.environ.get("MATGRAM_HF_REPO"),
        help="Hugging Face model id (or set HF_REPO / MATGRAM_HF_REPO)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_ASSETS,
        help="Local exported asset directory",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch / revision to upload to (default: main)",
    )
    args = parser.parse_args()

    if not args.repo:
        raise SystemExit("pass --repo USER/smi-ted-esol (or set MATGRAM_HF_REPO)")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("set HF_TOKEN to a Hugging Face write token")

    staging = materialize_upload_dir(args.source.resolve())
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        create_repo(
            args.repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=os.environ["HF_TOKEN"],
        )
        info = api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo,
            repo_type="model",
            revision=args.revision,
            commit_message="Upload exported SMI-TED ESOL MAX assets",
        )
        print(f"uploaded to https://huggingface.co/{args.repo}")
        print(info)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download SMI-TED weights and vocab into the local model_assets directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ibm-research/materials.smi-ted"
FILES = (
    "model_weights.safetensors",
    "bert_vocab_curated.txt",
    "smi-ted-Light_40.pt",
)
VENDOR_VOCAB = (
    Path(__file__).resolve().parent.parent.parent
    / "vendor"
    / "ibm_materials"
    / "models"
    / "smi_ted"
    / "smi_ted_light"
    / "bert_vocab_curated.txt"
)


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.resolve() == src.resolve():
            print(f"already present: {dest}")
            return
        dest.unlink()
    try:
        dest.symlink_to(src)
        print(f"linked {dest} -> {src}")
    except OSError:
        shutil.copy2(src, dest)
        print(f"copied {dest} <- {src}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "model_assets"
        / "ibm-research_materials.smi-ted",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        dest = args.output_dir / filename
        if dest.exists() or dest.is_symlink():
            print(f"already present: {dest}")
            continue
        if filename == "bert_vocab_curated.txt" and VENDOR_VOCAB.is_file():
            link_or_copy(VENDOR_VOCAB, dest)
            continue
        path = Path(hf_hub_download(REPO_ID, filename))
        link_or_copy(path, dest)


if __name__ == "__main__":
    main()

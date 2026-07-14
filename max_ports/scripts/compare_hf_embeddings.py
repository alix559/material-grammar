#!/usr/bin/env python3
"""Compare SMI-TED embeddings between the IBM PyTorch reference and MAX."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# IBM reference implementation (vendored fast_transformers).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMI_TED_SRC = os.path.join(
    REPO_ROOT, "vendor", "ibm_materials", "models", "smi_ted", "inference", "smi_ted_light"
)


def _load_hf_reference(weights_path: str, vocab_path: str):
    sys.path.insert(0, SMI_TED_SRC)
    from load import load_smi_ted  # type: ignore[import-not-found]

    folder = os.path.dirname(weights_path)
    ckpt = os.path.basename(weights_path)
    model = load_smi_ted(folder=folder, ckpt_filename=ckpt, vocab_filename=os.path.basename(vocab_path))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles",
        default="CCO",
        help="Canonical SMILES string to encode",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to smi-ted-Light_40.pt or model_weights.safetensors directory",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    weights = args.weights
    if weights is None:
        pt_path = hf_hub_download("ibm-research/materials.smi-ted", "smi-ted-Light_40.pt")
        vocab_path = hf_hub_download(
            "ibm-research/materials.smi-ted", "bert_vocab_curated.txt"
        )
        folder = os.path.dirname(pt_path)
        model = _load_hf_reference(pt_path, vocab_path)
    else:
        vocab_path = hf_hub_download(
            "ibm-research/materials.smi-ted", "bert_vocab_curated.txt"
        )
        model = _load_hf_reference(weights, vocab_path)

    with torch.no_grad():
        emb = model.encode(args.smiles, return_torch=True).numpy()

    print(f"SMILES: {args.smiles}")
    print(f"HF embedding shape: {emb.shape}")
    print(f"HF embedding[:8]: {emb.flatten()[:8]}")
    print(f"HF embedding norm: {np.linalg.norm(emb):.6f}")
    print()
    print("MAX parity check: run `max serve` with --custom-architectures and compare outputs.")


if __name__ == "__main__":
    main()

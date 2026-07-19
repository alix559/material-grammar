#!/usr/bin/env python3
"""Compare pretrained SMI-TED embeddings: IBM PyTorch vs MAX serve.

Uses the same encode path as IBM ``load.py``, but patches the tokenizer to
``BertTokenizerLegacy``. Transformers ≥5 routes ``BertTokenizer`` through the
fast backend, which ignores IBM's ``_tokenize`` override and maps SMILES like
``CCO`` to ``[<bos>, <pad>, <eos>]``.

Published MoleculeNet scores (ESOL RMSE, BBBP AUC, …) need a probe/finetune
head — they are not raw embedding equality. This script checks whether MAX
matches IBM ``encode()`` on the same pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import regex as re
import torch
from transformers.models.bert.tokenization_bert_legacy import BertTokenizerLegacy

ROOT = Path(__file__).resolve().parents[1]
IBM_INFERENCE = (
    ROOT.parent
    / "vendor"
    / "ibm_materials"
    / "models"
    / "smi_ted"
    / "inference"
    / "smi_ted_light"
)
DEFAULT_ASSETS = ROOT / "model_assets" / "ibm-research_materials.smi-ted"
DEFAULT_MODEL = "./model_assets/ibm-research_materials.smi-ted"
DEFAULT_SMILES = (
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "C1CCCCC1",
    "CCN(CC)CC",
    "CC(C)O",
    "c1ccncc1",
)


def _patch_ibm_tokenizer() -> None:
    sys.path.insert(0, str(IBM_INFERENCE))
    import load as ibm_load  # type: ignore

    pattern = ibm_load.PATTERN

    class FixedMolTranBertTokenizer(BertTokenizerLegacy):
        def __init__(self, vocab_file: str = "", **kwargs):
            super().__init__(
                vocab_file,
                unk_token="<pad>",
                sep_token="<eos>",
                pad_token="<pad>",
                cls_token="<bos>",
                mask_token="<mask>",
                do_lower_case=False,
                **kwargs,
            )
            self.regex_tokenizer = re.compile(pattern)
            self.wordpiece_tokenizer = None
            self.basic_tokenizer = None
            with open(vocab_file) as handle:
                self.padding_idx = handle.readlines().index("<pad>\n")

        def _tokenize(self, text: str) -> list[str]:
            return self.regex_tokenizer.findall(text)

        def get_padding_idx(self) -> int:
            return self.padding_idx

    ibm_load.MolTranBertTokenizer = FixedMolTranBertTokenizer  # type: ignore


def load_ibm(assets: Path):
    _patch_ibm_tokenizer()
    from load import load_smi_ted  # type: ignore

    ckpt = "smi-ted-Light_40.pt"
    if not (assets / ckpt).is_file() and not (assets / ckpt).is_symlink():
        raise SystemExit(f"missing {assets / ckpt}; run: pixi run setup-model")
    model = load_smi_ted(
        folder=str(assets),
        ckpt_filename=ckpt,
        vocab_filename="bert_vocab_curated.txt",
    )
    model.eval()
    return model


def ibm_embed(model, smiles: str) -> np.ndarray:
    with torch.no_grad():
        emb = model.encode(smiles, return_torch=True)
    return np.asarray(emb.detach().cpu().float().numpy(), dtype=np.float64).reshape(-1)


def max_embed(base_url: str, model_name: str, smiles: str) -> np.ndarray:
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        json={"model": model_name, "input": smiles},
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    if "data" not in data:
        raise RuntimeError(f"unexpected MAX response: {data}")
    return np.asarray(data["data"][0]["embedding"], dtype=np.float64)


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    diff = a - b
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "cosine": cos,
        "ibm_norm": float(np.linalg.norm(a)),
        "max_norm": float(np.linalg.norm(b)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--url", default=os.getenv("MAX_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--smiles", nargs="+", default=list(DEFAULT_SMILES))
    parser.add_argument("--tol-cosine", type=float, default=0.999999)
    parser.add_argument("--tol-max-abs", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"IBM assets: {args.assets}")
    print(f"MAX URL:    {args.url}")
    print(f"MAX model:  {args.model}")
    print(f"SMILES:     {args.smiles}")
    print()
    print(
        "Paper benchmarks (Communications Chemistry / arXiv:2407.20267): "
        "frozen-probe ESOL RMSE ≈ 0.70, Lipo ≈ 0.65, BBBP AUC ≈ 0.92 — "
        "those need a task head, not embedding cosine."
    )
    print()

    ibm = load_ibm(args.assets)
    probe = ibm.tokenizer.encode("CCO")
    if probe != [0, 4, 4, 9, 1]:
        raise SystemExit(f"tokenizer still broken for CCO: {probe}")

    rows: list[tuple[str, dict[str, float]]] = []
    for smiles in args.smiles:
        left = ibm_embed(ibm, smiles)
        right = max_embed(args.url, args.model, smiles)
        row = metrics(left, right)
        rows.append((smiles, row))
        print(
            f"{smiles:12s}  dim={left.size}  cosine={row['cosine']:.8f}  "
            f"max_abs={row['max_abs']:.6e}  rmse={row['rmse']:.6e}"
        )

    worst_cos = min(m["cosine"] for _, m in rows)
    worst_abs = max(m["max_abs"] for _, m in rows)
    print()
    print(f"worst cosine:  {worst_cos:.8f}  (tol {args.tol_cosine})")
    print(f"worst max_abs: {worst_abs:.6e}  (tol {args.tol_max_abs})")

    ok = worst_cos >= args.tol_cosine and worst_abs <= args.tol_max_abs
    print("PASS: MAX matches IBM pretrained encode()" if ok else "FAIL")
    print(json.dumps({s: m for s, m in rows}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

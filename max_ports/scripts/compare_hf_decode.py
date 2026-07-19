#!/usr/bin/env python3
"""Compare IBM PyTorch decode vs MAX decode on shared embeddings.

Compares argmax token ids (strict) and detokenized SMILES for embeddings
produced by IBM ``encode()`` — not against the original SMILES (reconstruction
is lossy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from serve_api.decode_runtime import DecodeRuntime  # noqa: E402

from compare_hf_embeddings import (  # noqa: E402
    DEFAULT_ASSETS,
    DEFAULT_SMILES,
    ibm_embed,
    load_ibm,
)


def ibm_decode_ids(model, embeddings: np.ndarray) -> np.ndarray:
    """Match ``Smi_ted.decode`` up to argmax (before detokenize)."""
    model.decoder.eval()
    emb = torch.as_tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        pred_token_embds = model.decoder.autoencoder.decoder(emb)
        logits = model.decoder.lang_model(
            pred_token_embds.view(-1, model.max_len, model.n_embd)
        )
        return torch.argmax(logits, dim=-1).cpu().numpy()


def ibm_ids_to_smiles(model, token_ids: np.ndarray) -> list[str]:
    """Detokenize like IBM ``convert_tokens_to_string`` + strip pad."""
    tok = model.tokenizer
    out: list[str] = []
    for row in token_ids:
        tokens = tok.convert_ids_to_tokens([int(i) for i in row.tolist()])
        # IBM overrides convert_tokens_to_string on its tokenizer; Fixed*
        # from compare_hf_embeddings may not — mirror the join manually.
        stopwords = {"<bos>", "<eos>"}
        text = "".join(t for t in tokens if t not in stopwords)
        out.append(text.replace("<pad>", ""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--smiles", nargs="+", default=list(DEFAULT_SMILES))
    args = parser.parse_args()

    if not args.assets.is_dir():
        raise SystemExit(f"missing assets {args.assets}; run: pixi run setup-model")

    print("Loading IBM model…")
    ibm = load_ibm(args.assets)
    print("Encoding with IBM…")
    embeddings = np.stack([ibm_embed(ibm, s) for s in args.smiles], axis=0)

    print("IBM decode…")
    ibm_ids = ibm_decode_ids(ibm, embeddings)
    ibm_smiles = ibm_ids_to_smiles(ibm, ibm_ids)

    print("Loading MAX decode runtime…")
    runtime = DecodeRuntime()
    runtime.load(args.assets, device=args.device)
    max_out = runtime.decode_embeddings(embeddings)
    max_ids = np.asarray(max_out["token_ids"], dtype=np.int64)
    max_smiles = max_out["smiles"]

    id_match = np.array_equal(ibm_ids, max_ids)
    smiles_match = ibm_smiles == max_smiles
    print(f"token_ids match: {id_match}")
    print(f"smiles match:    {smiles_match}")
    for i, s in enumerate(args.smiles):
        same_ids = np.array_equal(ibm_ids[i], max_ids[i])
        print(
            f"  [{i}] in={s!r}\n"
            f"      ibm={ibm_smiles[i]!r}\n"
            f"      max={max_smiles[i]!r}\n"
            f"      ids_equal={same_ids}"
        )
        if not same_ids:
            diff = np.where(ibm_ids[i] != max_ids[i])[0]
            print(f"      first_diff_pos={diff[:8].tolist()}")

    if not id_match:
        return 1
    if not smiles_match:
        print("WARNING: token ids match but detokenized SMILES differ")
        return 2
    print("OK: MAX decode matches IBM on shared embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

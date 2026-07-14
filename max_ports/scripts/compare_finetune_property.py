#!/usr/bin/env python3
"""Compare IBM finetune PyTorch predictions vs MAX property serve.

Requires:
  - A finetune ``.pt`` with ``MODEL_STATE`` (or asset ``finetune_source.pt``)
  - MAX serve running on the matching ``model_assets/smi-ted-{task}/`` asset

Example::

    pixi run serve-esol   # terminal 1
    pixi run compare-finetune -- --task esol --smiles CCO --smiles c1ccccc1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = REPO_ROOT.parent / "vendor" / "ibm_materials" / "models" / "smi_ted"
FINETUNE_LIGHT = VENDOR_ROOT / "finetune" / "smi_ted_light"
DEFAULT_SERVE_URL = "http://127.0.0.1:8000"

TASK_MODEL_NAME = {
    "esol": "./model_assets/smi-ted-esol",
    "bbbp": "./model_assets/smi-ted-bbbp",
    "lipo": "./model_assets/smi-ted-lipo",
}


def _resolve_checkpoint(task: str, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    asset = REPO_ROOT / "model_assets" / f"smi-ted-{task}" / "finetune_source.pt"
    if asset.is_file() or asset.is_symlink():
        return asset.resolve()
    raise FileNotFoundError(
        f"no checkpoint for {task}; pass --checkpoint or export with "
        f"finetune_source.pt under model_assets/smi-ted-{task}/"
    )


def _load_finetune_model(checkpoint: Path):
    if not FINETUNE_LIGHT.is_dir():
        raise FileNotFoundError(f"missing vendored finetune code at {FINETUNE_LIGHT}")
    sys.path.insert(0, str(VENDOR_ROOT / "finetune"))
    from smi_ted_light.load import load_smi_ted  # type: ignore[import-not-found]

    folder = checkpoint.parent
    vocab = folder / "bert_vocab_curated.txt"
    if not vocab.is_file():
        # Ensure vocab sits next to the checkpoint for load_smi_ted.
        src = FINETUNE_LIGHT / "bert_vocab_curated.txt"
        if not src.is_file():
            raise FileNotFoundError(f"missing vocab at {src}")
        try:
            vocab.symlink_to(src.resolve())
        except OSError:
            import shutil

            shutil.copy2(src, vocab)

    model = load_smi_ted(
        folder=str(folder),
        ckpt_filename=checkpoint.name,
        vocab_filename="bert_vocab_curated.txt",
        eval=True,
    )
    model.eval()
    return model


def pytorch_predict(model, smiles: list[str]) -> np.ndarray:
    with torch.no_grad():
        emb = model.extract_embeddings(smiles)
        if not torch.is_tensor(emb):
            emb = torch.as_tensor(emb)
        preds = model.net(emb).squeeze(-1).cpu().numpy()
    if preds.ndim == 0:
        preds = preds.reshape(1)
    return preds.astype(np.float64)


def max_predict(
    smiles: list[str],
    *,
    base_url: str,
    model_name: str,
    timeout_s: float,
) -> np.ndarray:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        json={"model": model_name, "input": smiles},
        timeout=timeout_s,
    )
    response.raise_for_status()
    rows = response.json()["data"]
    rows.sort(key=lambda row: row["index"])
    vals = [float(row["embedding"][0]) for row in rows]
    return np.asarray(vals, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_MODEL_NAME), required=True)
    parser.add_argument("--smiles", action="append", default=["CCO"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--base-url", default=DEFAULT_SERVE_URL)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    smiles = list(args.smiles)
    ckpt = _resolve_checkpoint(args.task, args.checkpoint)
    model_name = args.model_name or TASK_MODEL_NAME[args.task]

    print(f"checkpoint: {ckpt}")
    print(f"MAX model:  {model_name}")
    print(f"smiles:     {smiles}")

    model = _load_finetune_model(ckpt)
    pt = pytorch_predict(model, smiles)
    mx = max_predict(
        smiles,
        base_url=args.base_url,
        model_name=model_name,
        timeout_s=args.timeout,
    )

    rows = []
    ok = True
    for smi, a, b in zip(smiles, pt, mx, strict=True):
        abs_err = float(abs(a - b))
        rel_err = float(abs_err / max(abs(a), 1e-12))
        match = bool(np.isclose(a, b, rtol=args.rtol, atol=args.atol))
        ok = ok and match
        rows.append(
            {
                "smiles": smi,
                "pytorch": float(a),
                "max": float(b),
                "abs_err": abs_err,
                "rel_err": rel_err,
                "match": match,
            }
        )
        print(
            f"{smi}\tpytorch={a:.6f}\tmax={b:.6f}\t"
            f"abs_err={abs_err:.3e}\tmatch={match}"
        )

    print(json.dumps({"ok": ok, "rows": rows}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

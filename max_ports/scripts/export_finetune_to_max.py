#!/usr/bin/env python3
"""Export an IBM SMI-TED finetune .pt into a MAX model-asset directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_ASSETS = ROOT / "model_assets" / "ibm-research_materials.smi-ted"
DEFAULT_OUT_ROOT = ROOT / "model_assets"

TASK_META: dict[str, dict[str, Any]] = {
    "esol": {
        "task_type": "regression",
        "target_name": "measured log solubility in mols per litre",
        "target_unit": "log10(mol/L)",
        "n_output": 1,
    },
    "bbbp": {
        "task_type": "classification",
        "target_name": "blood-brain barrier penetration (p_np)",
        "target_unit": "logit",
        "n_output": 1,
    },
    "lipo": {
        "task_type": "regression",
        "target_name": "octanol/water distribution coefficient",
        "target_unit": "logP/logD",
        "n_output": 1,
    },
}


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.resolve() == src.resolve():
            return
        dest.unlink()
    try:
        dest.symlink_to(src)
    except OSError:
        shutil.copy2(src, dest)


def load_model_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "MODEL_STATE" not in payload:
        raise SystemExit(
            f"{checkpoint} has no MODEL_STATE; is this an IBM finetune checkpoint?"
        )
    state = payload["MODEL_STATE"]
    if not isinstance(state, dict):
        raise SystemExit(f"MODEL_STATE must be a dict, got {type(state)}")
    return {str(k): v.detach().cpu().contiguous() for k, v in state.items()}


def base_config(base_assets: Path) -> dict[str, Any]:
    cfg_path = base_assets / "config.json"
    if cfg_path.is_file():
        return json.loads(cfg_path.read_text())
    return {
        "architectures": ["SmiTedModel"],
        "torch_dtype": "float32",
        "n_batch": 32,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "max_len": 202,
        "max_position_embeddings": 202,
        "d_dropout": 0.1,
        "dropout": 0.1,
        "num_feats": 32,
        "vocab_size": 2393,
        "pad_token_id": 2,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "smi_ted_version": "v1",
        "train_decoder": 1,
    }


def export_finetune(
    *,
    checkpoint: Path,
    task: str,
    output_dir: Path | None,
    base_assets: Path,
) -> Path:
    meta = TASK_META[task]
    out = output_dir or (DEFAULT_OUT_ROOT / f"smi-ted-{task}")
    out.mkdir(parents=True, exist_ok=True)

    state = load_model_state(checkpoint)
    if not any(k.startswith("net.") for k in state):
        raise SystemExit(f"{checkpoint} MODEL_STATE has no net.* weights")

    # Finetune .pt often omits AE-decoder / lang heads; merge from pretrained
    # so the encode graph (which constructs those modules) can load strictly.
    base_weights = base_assets / "model_weights.safetensors"
    if base_weights.is_file():
        from safetensors.torch import load_file

        base_state = load_file(str(base_weights))
        for key, tensor in base_state.items():
            if key.startswith(
                ("decoder.autoencoder.decoder.", "decoder.lang_model.")
            ) and key not in state:
                state[key] = tensor.detach().cpu().contiguous()

    save_file(state, str(out / "model_weights.safetensors"))

    config = base_config(base_assets)
    config["architectures"] = ["SmiTedModel"]
    config["smi_ted_output"] = "property"
    config["smi_ted_task"] = task
    config["n_output"] = int(meta["n_output"])
    config["task_type"] = meta["task_type"]
    config["target_name"] = meta["target_name"]
    config["target_unit"] = meta["target_unit"]
    config["finetune_checkpoint"] = str(checkpoint.resolve())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    vocab_src = base_assets / "bert_vocab_curated.txt"
    if not vocab_src.is_file():
        raise SystemExit(f"missing vocab at {vocab_src}; run: pixi run setup-model")
    link_or_copy(vocab_src, out / "bert_vocab_curated.txt")
    link_or_copy(checkpoint.resolve(), out / "finetune_source.pt")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_META), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-assets", type=Path, default=DEFAULT_BASE_ASSETS)
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
        "--custom-architectures ./mat_gram01 "
        "--quantization-encoding float32"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

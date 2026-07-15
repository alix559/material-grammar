from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .checkpoint_export import export_finetune

REQUIRED_ASSET_FILES = (
    "model_weights.safetensors",
    "config.json",
    "bert_vocab_curated.txt",
)


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "name": path.name,
    }


def prepare_assets(
    *,
    checkpoint: Path,
    task: str,
    output_dir: Path,
    base_assets: Path,
) -> tuple[Path, bool]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not (base_assets / "bert_vocab_curated.txt").is_file():
        raise FileNotFoundError(
            f"base vocabulary not found at {base_assets}; run setup-model first"
        )

    fingerprint = checkpoint_fingerprint(checkpoint)
    metadata_path = output_dir / "matgram-export.json"
    if metadata_path.is_file() and all(
        (output_dir / name).is_file() for name in REQUIRED_ASSET_FILES
    ):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("task") == task and metadata.get("checkpoint") == fingerprint:
            return output_dir, False

    output_dir.mkdir(parents=True, exist_ok=True)
    export_finetune(
        checkpoint=checkpoint,
        task=task,
        output_dir=output_dir,
        base_assets=base_assets,
    )
    metadata_path.write_text(
        json.dumps({"task": task, "checkpoint": fingerprint}, indent=2) + "\n"
    )
    return output_dir, True

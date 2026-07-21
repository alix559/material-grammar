"""Download exported ESOL MAX assets from Hugging Face for Railway/auto-load."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("config.json", "model_weights.safetensors", "bert_vocab_curated.txt")


def assets_root() -> Path:
    override = os.environ.get("MATGRAM_ASSETS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (REPO_ROOT / "model_assets").resolve()


def esol_asset_dir() -> Path:
    return assets_root() / "smi-ted-esol"


def assets_ready(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED)


def download_esol_assets(
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    dest: Path | None = None,
) -> Path:
    """Ensure exported ESOL assets exist locally; download from HF if needed."""
    repo_id = repo_id or os.environ.get("MATGRAM_HF_REPO")
    if not repo_id:
        raise ValueError("MATGRAM_HF_REPO is required to download ESOL assets")
    revision = revision or os.environ.get("MATGRAM_HF_REVISION", "main")
    dest = (dest or esol_asset_dir()).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if assets_ready(dest):
        return dest

    token = os.environ.get("HF_TOKEN") or None
    cache = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        allow_patterns=list(REQUIRED),
    )
    cache_path = Path(cache)
    for name in REQUIRED:
        src = cache_path / name
        if not src.is_file():
            raise FileNotFoundError(f"{repo_id} is missing {name}")
        target = dest / name
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(src, target)

    if not assets_ready(dest):
        raise FileNotFoundError(f"download incomplete under {dest}")
    return dest

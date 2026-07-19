"""Side InferenceSession for SMI-TED decode (embedding → token ids → SMILES)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import Accelerator, Buffer, CPU, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import load_weights

from mat_gram01.graph import build_decode_graph
from mat_gram01.model_config import SmiTedHFConfig, SmiTedModelConfig
from mat_gram01.tokenizer import MolTranBertTokenizer
from mat_gram01.weight_adapters import convert_safetensor_state_dict


def _device_for(name: str) -> Device:
    if name == "gpu":
        return Accelerator()
    return CPU()


def _config_for(asset_dir: Path, device: Device) -> SmiTedModelConfig:
    raw = json.loads((asset_dir / "config.json").read_text())
    hf = SmiTedHFConfig.from_dict(raw)
    return SmiTedModelConfig(
        dtype=DType.float32,
        device=DeviceRef.from_device(device),
        huggingface_config=hf,
        max_seq_len=hf.max_len,
    )


class DecodeRuntime:
    """Compile and run the decode graph for one asset directory."""

    def __init__(self) -> None:
        self.model: Model | None = None
        self.tokenizer: MolTranBertTokenizer | None = None
        self.device: Device | None = None
        self.n_embd: int = 768
        self.max_len: int = 202
        self.asset_dir: Path | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        self.asset_dir = None

    def load(self, asset_dir: Path, *, device: str = "cpu") -> None:
        asset_dir = asset_dir.resolve()
        weights_path = asset_dir / "model_weights.safetensors"
        vocab_path = asset_dir / "bert_vocab_curated.txt"
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing {weights_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"missing {vocab_path}")

        self.unload()
        drv = _device_for(device)
        config = _config_for(asset_dir, drv)
        weights = load_weights([weights_path])
        state_dict = convert_safetensor_state_dict(dict(weights.items()))
        if not any(k.startswith("decoder.autoencoder.decoder.") for k in state_dict):
            raise RuntimeError(
                f"{asset_dir} has no AE-decoder weights; cannot load decode graph"
            )

        graph = build_decode_graph(config, state_dict)
        session = InferenceSession(devices=[drv])
        decode_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("decoder.")
        }
        self.model = session.load(graph, weights_registry=decode_state)
        self.tokenizer = MolTranBertTokenizer(
            vocab_file=str(vocab_path), model_max_length=config.huggingface_config.max_len
        )
        self.device = drv
        self.n_embd = config.huggingface_config.n_embd
        self.max_len = config.huggingface_config.max_len
        self.asset_dir = asset_dir

    def decode_embeddings(self, embeddings: list[list[float]] | np.ndarray) -> dict[str, Any]:
        if not self.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError("decode runtime not loaded")

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.n_embd:
            raise ValueError(
                f"embeddings must have shape [batch, {self.n_embd}], got {arr.shape}"
            )

        assert self.device is not None
        buf = Buffer.from_numpy(arr).to(self.device)
        outputs = self.model.execute(buf)
        token_ids = np.asarray(outputs[0].to_numpy(), dtype=np.int64)
        token_ids = np.squeeze(token_ids)
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        elif token_ids.ndim != 2:
            raise RuntimeError(f"unexpected token_ids shape: {token_ids.shape}")

        smiles = [self.tokenizer.ids_to_smiles(row) for row in token_ids]
        return {
            "smiles": smiles,
            "token_ids": token_ids.tolist(),
        }

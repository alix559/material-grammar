#!/usr/bin/env python3
"""Train and run molecular property prediction with SMI-TED embeddings.

Workflow:
1. ``train`` — freeze the SMI-TED encoder, fit the small ``Net`` head on a
   MoleculeNet split (PyTorch reference embeddings).
2. ``predict`` — embed SMILES with MAX Serve or PyTorch, then run the saved head.

Examples::

    # Train an ESOL solubility model (CPU, a few minutes):
    pixi run train-property

    # Predict one compound with MAX Serve running:
    pixi run serve   # in another terminal
    pixi run predict-property -- --smiles "CCO"

    # Predict from a CSV column named ``smiles``:
    pixi run python scripts/predict_property.py predict --task esol \\
        --input-csv molecules.csv --backend max
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = REPO_ROOT.parent / "vendor" / "ibm_materials" / "models" / "smi_ted"
SMI_TED_SRC = VENDOR_ROOT / "inference" / "smi_ted_light"
MODEL_DIR = REPO_ROOT / "model_assets" / "ibm-research_materials.smi-ted"
DEFAULT_SERVE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_NAME = "./model_assets/ibm-research_materials.smi-ted"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    data_dir: Path
    target_col: str
    smiles_col: str
    task_type: Literal["regression", "classification"]
    metric: str


TASKS: dict[str, TaskSpec] = {
    "esol": TaskSpec(
        name="esol",
        data_dir=VENDOR_ROOT / "finetune" / "moleculenet" / "esol",
        target_col="measured log solubility in mols per litre",
        smiles_col="smiles",
        task_type="regression",
        metric="rmse",
    ),
    "bbbp": TaskSpec(
        name="bbbp",
        data_dir=VENDOR_ROOT / "finetune" / "moleculenet" / "bbbp",
        target_col="p_np",
        smiles_col="smiles",
        task_type="classification",
        metric="roc-auc",
    ),
    "lipo": TaskSpec(
        name="lipo",
        data_dir=VENDOR_ROOT / "finetune" / "moleculenet" / "lipophilicity",
        target_col="y",
        smiles_col="smiles",
        task_type="regression",
        metric="rmse",
    ),
}


class PropertyNet(nn.Module):
    """IBM SMI-TED downstream head (``net`` in the reference implementation)."""

    def __init__(self, embed_dim: int = 768, n_output: int = 1, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.final = nn.Linear(embed_dim, n_output)

    def forward(self, embeddings: torch.Tensor, multitask: bool = False) -> torch.Tensor:
        x_out = F.gelu(self.fc1(embeddings))
        x_out = self.dropout1(x_out)
        x_out = x_out + embeddings

        z = F.gelu(self.fc2(x_out))
        z = self.dropout2(z)
        z = self.final(z + x_out)

        if multitask:
            return torch.sigmoid(z)
        return z


def _read_split(path: Path, smiles_col: str, target_col: str) -> tuple[list[str], np.ndarray]:
    smiles: list[str] = []
    targets: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles.append(row[smiles_col])
            targets.append(float(row[target_col]))
    return smiles, np.asarray(targets, dtype=np.float32)


def _load_pytorch_encoder():
    if not SMI_TED_SRC.is_dir():
        raise FileNotFoundError(f"missing vendored SMI-TED source at {SMI_TED_SRC}")

    sys.path.insert(0, str(SMI_TED_SRC))
    from load import load_smi_ted  # type: ignore[import-not-found]

    from huggingface_hub import hf_hub_download

    model_dir = MODEL_DIR
    vocab_path = model_dir / "bert_vocab_curated.txt"
    pt_path = model_dir / "smi-ted-Light_40.pt"

    if not vocab_path.is_file():
        cached_vocab = Path(
            hf_hub_download("ibm-research/materials.smi-ted", "bert_vocab_curated.txt")
        )
        if not vocab_path.exists():
            vocab_path.symlink_to(cached_vocab)

    if not pt_path.is_file():
        cached_pt = Path(
            hf_hub_download("ibm-research/materials.smi-ted", "smi-ted-Light_40.pt")
        )
        if not pt_path.exists():
            pt_path.symlink_to(cached_pt)

    model = load_smi_ted(
        folder=str(model_dir),
        ckpt_filename=pt_path.name,
        vocab_filename=vocab_path.name,
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def embed_pytorch(model, smiles: list[str], batch_size: int = 64) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        with torch.no_grad():
            emb = model.encode(batch, return_torch=True).cpu().numpy()
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        chunks.append(emb.astype(np.float32))
    return np.vstack(chunks)


def embed_max(
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
    return np.asarray([row["embedding"] for row in rows], dtype=np.float32)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def evaluate_predictions(
    task: TaskSpec,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    if task.task_type == "regression":
        return {
            "rmse": _rmse(y_true, y_pred),
            "mae": _mae(y_true, y_pred),
        }
    probs = 1.0 / (1.0 + np.exp(-y_pred))
    return {"roc-auc": _roc_auc(y_true, probs)}


def checkpoint_dir(task: str) -> Path:
    return CHECKPOINT_ROOT / task


def save_checkpoint(
    path: Path,
    *,
    task: TaskSpec,
    state_dict: dict[str, torch.Tensor],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": asdict(task),
            "metrics": metrics,
            "state_dict": state_dict,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[TaskSpec, PropertyNet, dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    task = TaskSpec(**payload["task"])
    head = PropertyNet()
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return task, head, payload.get("metrics", {})


def train_head(
    task_name: str,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_train_samples: int | None,
    device: str,
) -> Path:
    task = TASKS[task_name]
    train_smiles, train_y = _read_split(
        task.data_dir / "train.csv", task.smiles_col, task.target_col
    )
    valid_smiles, valid_y = _read_split(
        task.data_dir / "valid.csv", task.smiles_col, task.target_col
    )

    if max_train_samples is not None:
        train_smiles = train_smiles[:max_train_samples]
        train_y = train_y[:max_train_samples]

    print(f"Task: {task.name} ({task.task_type}, target={task.target_col!r})")
    print(f"Train: {len(train_smiles)} molecules, valid: {len(valid_smiles)}")

    encoder = _load_pytorch_encoder()
    print("Encoding training SMILES...")
    train_x = embed_pytorch(encoder, train_smiles, batch_size=batch_size)
    print("Encoding validation SMILES...")
    valid_x = embed_pytorch(encoder, valid_smiles, batch_size=batch_size)

    head = PropertyNet().to(device)
    if task.task_type == "regression":
        loss_fn: nn.Module = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        head.train()
        running = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = head(batch_x).squeeze(-1)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        head.eval()
        with torch.no_grad():
            valid_preds = head(torch.from_numpy(valid_x).to(device)).squeeze(-1).cpu().numpy()
        metrics = evaluate_predictions(task, valid_y, valid_preds)
        score = metrics[task.metric]
        print(
            f"epoch {epoch:03d}  train_loss={running / len(train_loader):.4f}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

        if score < best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_valid_metrics = metrics

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    out_path = checkpoint_dir(task.name) / "net_head.pt"
    save_checkpoint(out_path, task=task, state_dict=best_state, metrics=best_valid_metrics)
    print(f"Saved head to {out_path}")
    print("Validation metrics:", best_valid_metrics)
    return out_path


def predict_properties(
    *,
    task_name: str,
    smiles: list[str],
    backend: Literal["pytorch", "max"],
    checkpoint: Path | None,
    base_url: str,
    model_name: str,
    timeout_s: float,
) -> list[float]:
    ckpt_path = checkpoint or (checkpoint_dir(task_name) / "net_head.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"missing checkpoint {ckpt_path}. Run: pixi run train-property -- --task {task_name}"
        )

    task, head, metrics = load_checkpoint(ckpt_path)
    if task.name != task_name:
        raise ValueError(f"checkpoint task {task.name!r} does not match --task {task_name!r}")

    if backend == "pytorch":
        encoder = _load_pytorch_encoder()
        embeddings = embed_pytorch(encoder, smiles)
    else:
        embeddings = embed_max(
            smiles,
            base_url=base_url,
            model_name=model_name,
            timeout_s=timeout_s,
        )

    with torch.no_grad():
        logits = head(torch.from_numpy(embeddings)).squeeze(-1).cpu().numpy()

    if task.task_type == "classification":
        return [float(x) for x in (1.0 / (1.0 + np.exp(-logits)))]
    return [float(x) for x in logits]


def _load_smiles_from_csv(path: Path, smiles_col: str) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [row[smiles_col] for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description="SMI-TED property prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Net head on MoleculeNet")
    train_parser.add_argument("--task", choices=sorted(TASKS), default="esol")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick experiments",
    )
    train_parser.add_argument("--device", default="cpu")

    predict_parser = subparsers.add_parser("predict", help="Predict properties for SMILES")
    predict_parser.add_argument("--task", choices=sorted(TASKS), default="esol")
    predict_parser.add_argument("--smiles", action="append", default=[])
    predict_parser.add_argument("--input-csv")
    predict_parser.add_argument("--smiles-col", default="smiles")
    predict_parser.add_argument("--checkpoint")
    predict_parser.add_argument("--backend", choices=("pytorch", "max"), default="max")
    predict_parser.add_argument("--base-url", default=DEFAULT_SERVE_URL)
    predict_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    predict_parser.add_argument("--timeout", type=float, default=120.0)
    predict_parser.add_argument("--output-json")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved head on the test split")
    eval_parser.add_argument("--task", choices=sorted(TASKS), default="esol")
    eval_parser.add_argument("--checkpoint")
    eval_parser.add_argument("--backend", choices=("pytorch", "max"), default="pytorch")
    eval_parser.add_argument("--base-url", default=DEFAULT_SERVE_URL)
    eval_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    eval_parser.add_argument("--timeout", type=float, default=120.0)

    args = parser.parse_args()

    if args.command == "train":
        train_head(
            args.task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_samples=args.max_train_samples,
            device=args.device,
        )
        return 0

    if args.command == "predict":
        smiles = list(args.smiles)
        if args.input_csv:
            smiles.extend(_load_smiles_from_csv(Path(args.input_csv), args.smiles_col))
        if not smiles:
            parser.error("provide --smiles and/or --input-csv")

        preds = predict_properties(
            task_name=args.task,
            smiles=smiles,
            backend=args.backend,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            base_url=args.base_url,
            model_name=args.model_name,
            timeout_s=args.timeout,
        )

        rows = [{"smiles": s, "prediction": p} for s, p in zip(smiles, preds, strict=True)]
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(rows, indent=2))
            print(f"Wrote {args.output_json}")
        else:
            task = TASKS[args.task]
            label = "probability" if task.task_type == "classification" else "value"
            for row in rows:
                print(f"{row['smiles']}\t{label}={row['prediction']:.6f}")
        return 0

    if args.command == "evaluate":
        task = TASKS[args.task]
        test_smiles, test_y = _read_split(
            task.data_dir / "test.csv", task.smiles_col, task.target_col
        )
        preds = np.asarray(
            predict_properties(
                task_name=args.task,
                smiles=test_smiles,
                backend=args.backend,
                checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                base_url=args.base_url,
                model_name=args.model_name,
                timeout_s=args.timeout,
            ),
            dtype=np.float32,
        )
        if task.task_type == "classification":
            preds_for_metric = preds
            test_logits = np.log(preds_for_metric / (1.0 - preds_for_metric + 1e-12))
            metrics = evaluate_predictions(task, test_y, test_logits)
        else:
            metrics = evaluate_predictions(task, test_y, preds)
        print(json.dumps(metrics, indent=2))
        return 0

    parser.error(f"unknown command {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

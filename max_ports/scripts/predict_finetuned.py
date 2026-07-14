#!/usr/bin/env python3
"""Predict molecular properties from a MAX property-serve asset.

Calls ``/v1/embeddings`` on a model exported with ``export_finetune_to_max.py``.
In property mode the returned vector is length 1 (the prediction). For BBBP the
raw value is a logit; pass ``--probability`` to apply sigmoid.

Examples::

    pixi run serve-esol   # terminal 1
    pixi run predict-finetuned -- --task esol --smiles CCO
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SERVE_URL = "http://127.0.0.1:8000"

TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "esol": {
        "model_name": "./model_assets/smi-ted-esol",
        "task_type": "regression",
        "target_unit": "log10(mol/L)",
        "label": "log_solubility",
    },
    "bbbp": {
        "model_name": "./model_assets/smi-ted-bbbp",
        "task_type": "classification",
        "target_unit": "logit",
        "label": "bbbp_logit",
    },
    "lipo": {
        "model_name": "./model_assets/smi-ted-lipo",
        "task_type": "regression",
        "target_unit": "logP/logD",
        "label": "lipophilicity",
    },
}


def _load_task_meta(task: str, model_path: Path | None) -> dict[str, Any]:
    meta = dict(TASK_DEFAULTS[task])
    cfg_path = (model_path or (REPO_ROOT / meta["model_name"].lstrip("./"))) / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        for key in ("task_type", "target_unit", "target_name", "smi_ted_task"):
            if key in cfg:
                meta[key] = cfg[key]
    return meta


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def embed_max(
    smiles: list[str],
    *,
    base_url: str,
    model_name: str,
    timeout_s: float,
) -> list[list[float]]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        json={"model": model_name, "input": smiles},
        timeout=timeout_s,
    )
    response.raise_for_status()
    rows = response.json()["data"]
    rows.sort(key=lambda row: row["index"])
    return [row["embedding"] for row in rows]


def _load_smiles_from_csv(path: Path, smiles_col: str) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [row[smiles_col] for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), required=True)
    parser.add_argument("--smiles", action="append", default=[])
    parser.add_argument("--input-csv")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--base-url", default=DEFAULT_SERVE_URL)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local asset dir (for reading config metadata)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--probability",
        action="store_true",
        help="For classification tasks, print sigmoid(logit) instead of logit",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()

    smiles = list(args.smiles)
    if args.input_csv:
        smiles.extend(_load_smiles_from_csv(Path(args.input_csv), args.smiles_col))
    if not smiles:
        parser.error("provide --smiles and/or --input-csv")

    meta = _load_task_meta(args.task, args.model_path)
    model_name = args.model_name or meta["model_name"]

    vectors = embed_max(
        smiles,
        base_url=args.base_url,
        model_name=model_name,
        timeout_s=args.timeout,
    )

    rows: list[dict[str, Any]] = []
    for smi, vec in zip(smiles, vectors, strict=True):
        if not vec:
            raise SystemExit(f"empty embedding for {smi!r}")
        value = float(vec[0])
        row: dict[str, Any] = {
            "smiles": smi,
            "task": args.task,
            "raw": value,
            "unit": meta.get("target_unit"),
        }
        if meta.get("task_type") == "classification":
            row["logit"] = value
            row["probability"] = _sigmoid(value)
            display = row["probability"] if args.probability else row["logit"]
            label = "probability" if args.probability else "logit"
        else:
            display = value
            label = meta.get("label", "value")
        row["prediction"] = display
        rows.append(row)
        print(f"{smi}\t{label}={display:.6f}\tunit={row['unit']}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote {args.output_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

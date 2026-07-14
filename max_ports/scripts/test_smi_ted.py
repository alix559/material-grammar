#!/usr/bin/env python3
"""Smoke and integration tests for the SMI-TED MAX port.

Examples::

    # Fast local checks (no server):
    pixi run test

    # Against a running ``pixi run serve``:
    pixi run test-serve

    # Local + optional HF reference comparison:
    pixi run python scripts/test_smi_ted.py --mode all --compare-hf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model_assets" / "ibm-research_materials.smi-ted"
DEFAULT_MODEL_NAME = "./model_assets/ibm-research_materials.smi-ted"
EMBEDDING_DIM = 768
TEST_SMILES = ("CCO", "c1ccccc1", "CC(=O)O")


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


def _fail(results: list[TestResult], name: str, detail: str) -> None:
    results.append(TestResult(name, False, detail))


def _pass(results: list[TestResult], name: str, detail: str = "") -> None:
    results.append(TestResult(name, True, detail))


def test_arch_registration(results: list[TestResult]) -> None:
    from materials_smi_ted import ARCHITECTURES

    if len(ARCHITECTURES) != 1:
        _fail(results, "arch_registration", f"expected 1 arch, got {len(ARCHITECTURES)}")
        return
    arch = ARCHITECTURES[0]
    if arch.name != "SmiTedModel":
        _fail(results, "arch_registration", f"unexpected arch name {arch.name!r}")
        return
    _pass(results, "arch_registration", arch.name)


def test_config(results: list[TestResult]) -> None:
    config_path = MODEL_DIR / "config.json"
    if not config_path.is_file():
        _fail(results, "config", f"missing {config_path}")
        return

    cfg = json.loads(config_path.read_text())
    if cfg.get("architectures") != ["SmiTedModel"]:
        _fail(results, "config", f"unexpected architectures: {cfg.get('architectures')}")
        return
    if "model_type" in cfg:
        _fail(
            results,
            "config",
            "model_type must be omitted so MAX uses PretrainedConfig fallback",
        )
        return
    if cfg.get("n_embd") != EMBEDDING_DIM:
        _fail(results, "config", f"expected n_embd={EMBEDDING_DIM}")
        return
    _pass(results, "config", f"max_len={cfg.get('max_len')}")


def _load_adapted_state_dict(weights_path: Path) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    from materials_smi_ted.weight_adapters import convert_safetensor_state_dict
    from max.graph.weights import WeightData

    class _WeightShim:
        def __init__(self, data: np.ndarray) -> None:
            self._data = data

        def data(self) -> np.ndarray:
            return self._data

    raw: dict[str, _WeightShim] = {}
    with safe_open(weights_path, framework="numpy") as handle:
        for key in handle.keys():
            raw[key] = _WeightShim(handle.get_tensor(key))
    return convert_safetensor_state_dict(raw)


def test_weights(results: list[TestResult]) -> None:
    weights_path = MODEL_DIR / "model_weights.safetensors"
    if not weights_path.is_file():
        _fail(results, "weights", f"missing {weights_path} (run: pixi run setup-model)")
        return

    adapted = _load_adapted_state_dict(weights_path)
    # Encode path (~224) + net.* property head (6).
    if len(adapted) < 220:
        _fail(results, "weights", f"expected ~230 tensors, got {len(adapted)}")
        return
    net_keys = [k for k in adapted if k.startswith("net.")]
    if len(net_keys) != 6:
        _fail(results, "weights", f"expected 6 net.* tensors, got {len(net_keys)}")
        return
    _pass(results, "weights", f"{len(adapted)} tensors after adapter ({len(net_keys)} net.*)")


def test_tokenizer(results: list[TestResult]) -> None:
    vocab_path = MODEL_DIR / "bert_vocab_curated.txt"
    if not vocab_path.is_file():
        _fail(results, "tokenizer", f"missing {vocab_path}")
        return

    from materials_smi_ted.tokenizer import MolTranBertTokenizer

    tok = MolTranBertTokenizer(vocab_file=str(vocab_path), model_max_length=202)
    encoded = tok("CCO", add_special_tokens=True, max_length=202, padding="max_length")
    if len(encoded["input_ids"]) != 202:
        _fail(results, "tokenizer", f"expected length 202, got {len(encoded['input_ids'])}")
        return
    if sum(encoded["attention_mask"]) == 0:
        _fail(results, "tokenizer", "attention mask is all zeros")
        return
    _pass(results, "tokenizer", f"{int(sum(encoded['attention_mask']))} active tokens for CCO")


def test_graph_build(results: list[TestResult]) -> None:
    weights_path = MODEL_DIR / "model_weights.safetensors"
    if not weights_path.is_file():
        _fail(results, "graph_build", "weights missing (skipped)")
        return

    from materials_smi_ted.graph import build_graph
    from materials_smi_ted.model_config import SmiTedHFConfig, SmiTedModelConfig
    from max.dtype import DType
    from max.graph import DeviceRef

    state_dict = _load_adapted_state_dict(weights_path)

    config = SmiTedModelConfig(
        dtype=DType.float32,
        device=DeviceRef.CPU(),
        huggingface_config=SmiTedHFConfig(),
        max_seq_len=202,
    )
    try:
        graph = build_graph(config, state_dict)
    except Exception as exc:
        _fail(results, "graph_build", str(exc))
        return
    _pass(results, "graph_build", graph.name)

    prop_config = SmiTedModelConfig(
        dtype=DType.float32,
        device=DeviceRef.CPU(),
        huggingface_config=SmiTedHFConfig(
            smi_ted_output="property",
            smi_ted_task="esol",
            n_output=1,
            task_type="regression",
        ),
        max_seq_len=202,
    )
    try:
        prop_graph = build_graph(prop_config, state_dict)
    except Exception as exc:
        _fail(results, "graph_build_property", str(exc))
        return
    _pass(results, "graph_build_property", prop_graph.name)


def wait_for_server(base_url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def fetch_embedding(
    base_url: str,
    model_name: str,
    smiles: str,
    timeout_s: float,
) -> np.ndarray:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        json={"model": model_name, "input": smiles},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload["data"]
    if len(data) != 1:
        raise ValueError(f"expected 1 embedding, got {len(data)}")
    return np.asarray(data[0]["embedding"], dtype=np.float32)


def test_server_health(results: list[TestResult], base_url: str, timeout_s: float) -> None:
    if wait_for_server(base_url, timeout_s):
        _pass(results, "server_health", base_url)
    else:
        _fail(results, "server_health", f"no response from {base_url} within {timeout_s:.0f}s")


def test_embeddings_api(
    results: list[TestResult],
    base_url: str,
    model_name: str,
    timeout_s: float,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/embeddings",
            json={"model": model_name, "input": list(TEST_SMILES)},
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        batch_rows = [np.asarray(row["embedding"], dtype=np.float32) for row in payload["data"]]
        batch = np.stack(batch_rows, axis=0)
    except Exception as exc:
        _fail(results, "embeddings_batch", str(exc))
        return embeddings

    if batch.shape != (len(TEST_SMILES), EMBEDDING_DIM):
        _fail(
            results,
            "embeddings_batch",
            f"expected shape {(len(TEST_SMILES), EMBEDDING_DIM)}, got {batch.shape}",
        )
    elif not np.isfinite(batch).all():
        _fail(results, "embeddings_batch", "non-finite values in batch response")
    else:
        _pass(results, "embeddings_batch", str(batch.shape))
        for smiles, emb in zip(TEST_SMILES, batch_rows, strict=True):
            embeddings[smiles] = emb

    for smiles, emb in embeddings.items():
        name = f"embedding:{smiles}"
        if emb.shape != (EMBEDDING_DIM,):
            _fail(results, name, f"expected shape ({EMBEDDING_DIM},), got {emb.shape}")
        elif not np.isfinite(emb).all():
            _fail(results, name, "non-finite values")
        elif np.linalg.norm(emb) == 0:
            _fail(results, name, "zero vector")
        else:
            _pass(results, name, f"norm={np.linalg.norm(emb):.4f}")

    if len(embeddings) >= 2:
        keys = list(embeddings)
        cos = np.dot(embeddings[keys[0]], embeddings[keys[1]]) / (
            np.linalg.norm(embeddings[keys[0]]) * np.linalg.norm(embeddings[keys[1]])
        )
        if abs(cos) > 0.999:
            _fail(results, "embedding_distinct", f"CCO vs benzene cosine too high: {cos:.4f}")
        else:
            _pass(results, "embedding_distinct", f"cos(CCO, c1ccccc1)={cos:.4f}")

    return embeddings


def compare_hf_reference(
    results: list[TestResult],
    max_embeddings: dict[str, np.ndarray],
    *,
    strict: bool,
) -> None:
    import os
    import torch

    smi_ted_src = (
        REPO_ROOT.parent
        / "vendor"
        / "ibm_materials"
        / "models"
        / "smi_ted"
        / "inference"
        / "smi_ted_light"
    )
    if not smi_ted_src.is_dir():
        _fail(results, "hf_reference", f"missing vendored source at {smi_ted_src}")
        return

    from huggingface_hub import hf_hub_download
    import transformers
    from transformers.models.bert.tokenization_bert_legacy import BertTokenizerLegacy

    # transformers>=5 made BertTokenizer a fast WordPiece backend; IBM's
    # MolTranBertTokenizer relies on the slow `_tokenize` override.
    transformers.BertTokenizer = BertTokenizerLegacy
    import transformers.models.bert.tokenization_bert as _bert_tok

    _bert_tok.BertTokenizer = BertTokenizerLegacy

    sys.path.insert(0, str(smi_ted_src))
    from load import load_smi_ted  # type: ignore[import-not-found]

    # Prefer local model_assets (vocab + checkpoint together). HF cache snapshots
    # often omit bert_vocab_curated.txt, which breaks load_smi_ted's join(folder, vocab).
    local_pt = MODEL_DIR / "smi-ted-Light_40.pt"
    local_vocab = MODEL_DIR / "bert_vocab_curated.txt"
    if local_pt.is_file() and local_vocab.is_file():
        folder = str(MODEL_DIR)
        ckpt_filename = local_pt.name
        vocab_filename = local_vocab.name
    else:
        pt_path = hf_hub_download("ibm-research/materials.smi-ted", "smi-ted-Light_40.pt")
        vocab_path = Path(
            hf_hub_download("ibm-research/materials.smi-ted", "bert_vocab_curated.txt")
        )
        # load_smi_ted joins folder with both filenames; keep them co-located.
        folder = str(vocab_path.parent)
        ckpt_filename = os.path.basename(pt_path)
        vocab_filename = vocab_path.name
        if os.path.dirname(pt_path) != folder:
            import shutil

            dest = Path(folder) / ckpt_filename
            if not dest.exists():
                shutil.copy2(pt_path, dest)

    model = load_smi_ted(
        folder=folder,
        ckpt_filename=ckpt_filename,
        vocab_filename=vocab_filename,
    )

    for smiles, max_emb in max_embeddings.items():
        name = f"hf_parity:{smiles}"
        with torch.no_grad():
            hf_emb = model.encode(smiles, return_torch=True).numpy().astype(np.float32)
            hf_emb = np.asarray(hf_emb).reshape(-1)
        if hf_emb.shape != max_emb.shape:
            _fail(results, name, f"shape mismatch hf={hf_emb.shape} max={max_emb.shape}")
            continue
        cos = float(
            np.dot(hf_emb, max_emb)
            / (np.linalg.norm(hf_emb) * np.linalg.norm(max_emb) + 1e-12)
        )
        mae = float(np.mean(np.abs(hf_emb - max_emb)))
        detail = f"cos={cos:.6f}, mae={mae:.6f}"
        if strict and cos < 0.99:
            _fail(results, name, detail)
        else:
            _pass(results, name, detail)


def run_local_tests() -> list[TestResult]:
    results: list[TestResult] = []
    test_arch_registration(results)
    test_config(results)
    test_weights(results)
    test_tokenizer(results)
    test_graph_build(results)
    return results


def run_serve_tests(
    base_url: str,
    model_name: str,
    *,
    wait_s: float,
    request_timeout_s: float,
    compare_hf: bool,
    strict: bool,
) -> list[TestResult]:
    results: list[TestResult] = []
    test_server_health(results, base_url, wait_s)
    if not results or not results[-1].passed:
        return results

    embeddings = test_embeddings_api(results, base_url, model_name, request_timeout_s)
    if compare_hf and embeddings:
        compare_hf_reference(results, embeddings, strict=strict)
    return results


def print_report(results: list[TestResult]) -> int:
    width = max((len(r.name) for r in results), default=4)
    failed = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        if not result.passed:
            failed += 1
        suffix = f" — {result.detail}" if result.detail else ""
        print(f"{status:<4} {result.name:<{width}}{suffix}")

    print()
    if failed:
        print(f"{failed}/{len(results)} test(s) failed.")
        return 1
    print(f"All {len(results)} test(s) passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the SMI-TED MAX port.")
    parser.add_argument(
        "--mode",
        choices=("local", "serve", "all"),
        default="local",
        help="local=offline checks, serve=HTTP API, all=both",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--wait",
        type=float,
        default=120.0,
        help="Seconds to wait for server /health in serve mode",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout for /v1/embeddings",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Compare MAX embeddings against the IBM PyTorch reference",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail HF parity checks unless cosine similarity >= 0.99",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))

    results: list[TestResult] = []
    if args.mode in ("local", "all"):
        results.extend(run_local_tests())
    if args.mode in ("serve", "all"):
        results.extend(
            run_serve_tests(
                args.base_url,
                args.model_name,
                wait_s=args.wait,
                request_timeout_s=args.timeout,
                compare_hf=args.compare_hf,
                strict=args.strict,
            )
        )

    return print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())

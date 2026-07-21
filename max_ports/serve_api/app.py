"""HTTP API for loading SMI-TED weights into MAX and calling embeddings/decode.

FastHTML routes only — no HTML UI. JSON in / JSON out.

  pixi run api
  POST /load        {\"weight_path\": \"...\"}  or  {\"checkpoint\": \"...\", \"task\": \"esol\"}
  POST /embeddings  {\"smiles\": \"CCO\"}  or  {\"smiles\": [\"CCO\", ...]}
  POST /decode      {\"embeddings\": [[...768...], ...]}
  POST /roundtrip   {\"smiles\": [\"CCO\", ...]}
  GET  /status
  POST /stop

Railway / production: set MATGRAM_AUTO_LOAD=1 and MATGRAM_HF_REPO to download
exported ESOL assets from Hugging Face and start MAX on boot (no POST /load).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
from fasthtml.common import JSONResponse, fast_app
from starlette.requests import Request
from starlette.responses import Response

from .bootstrap import download_esol_assets
from .manager import ServeManager

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_finetune_to_max import TASK_META, export_finetune  # noqa: E402

PRETRAINED = REPO_ROOT / "model_assets" / "ibm-research_materials.smi-ted"
ARCH = REPO_ROOT / "mat_gram01"

MAX_PORT = int(os.environ.get("MATGRAM_MAX_PORT", "8000"))
DEVICE = os.environ.get("MATGRAM_DEVICE", "cpu")
STARTUP_TIMEOUT = float(os.environ.get("MATGRAM_STARTUP_TIMEOUT", "180"))
AUTO_LOAD = os.environ.get("MATGRAM_AUTO_LOAD", "").lower() in ("1", "true", "yes")

app, rt = fast_app(pico=False, live=False)
manager = ServeManager(
    repo_root=REPO_ROOT,
    architecture_path=ARCH,
    max_port=MAX_PORT,
    device=DEVICE,
    startup_timeout=STARTUP_TIMEOUT,
)


@app.on_event("startup")
async def _startup_auto_load() -> None:
    """Download HF assets + start MAX in the background so /health stays up."""
    if not AUTO_LOAD:
        return

    async def _run() -> None:
        import asyncio

        try:
            asset_dir = await asyncio.to_thread(download_esol_assets)
            await manager.start(asset_dir, device=DEVICE)
        except Exception as e:  # noqa: BLE001 — surface on /status
            manager.last_error = f"auto-load failed: {e}"

    import asyncio

    asyncio.create_task(_run())


@app.on_event("shutdown")
async def _shutdown() -> None:
    await manager.stop()


def _json_error(status: int, detail: str) -> Response:
    return JSONResponse({"error": detail}, status_code=status)


async def _read_json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise ValueError("request body must be JSON") from None
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def _resolve_assets(
    *,
    weight_path: str | None,
    checkpoint: str | None,
    task: str | None,
) -> Path:
    if weight_path and checkpoint:
        raise ValueError("pass weight_path or checkpoint, not both")
    if not weight_path and not checkpoint:
        raise ValueError("pass weight_path or checkpoint")
    if checkpoint and not task:
        raise ValueError("task is required when checkpoint is set")

    if weight_path:
        path = Path(weight_path).expanduser()
        path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"weight_path is not a directory: {path}")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"missing config.json in {path}")
        return path

    assert checkpoint is not None and task is not None
    if task not in TASK_META:
        raise ValueError(f"task must be one of {sorted(TASK_META)}")
    ckpt = Path(checkpoint).expanduser()
    ckpt = ckpt.resolve() if ckpt.is_absolute() else (REPO_ROOT / ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not PRETRAINED.is_dir():
        raise FileNotFoundError(f"run setup-model first; missing {PRETRAINED}")
    return export_finetune(
        checkpoint=ckpt,
        task=task,
        output_dir=None,
        base_assets=PRETRAINED,
    ).resolve()


@rt("/health")
def health():
    return {"status": "ok"}


@rt("/status")
async def status():
    return await manager.status()


@rt("/tasks")
def tasks():
    return {"tasks": sorted(TASK_META), "meta": TASK_META}


@rt("/load", methods=["POST"])
async def load(request: Request):
    try:
        body = await _read_json(request)
        asset_dir = _resolve_assets(
            weight_path=body.get("weight_path"),
            checkpoint=body.get("checkpoint"),
            task=body.get("task"),
        )
        device = body.get("device") or manager.device
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be cpu or gpu")
        if body.get("max_port") is not None:
            manager.max_port = int(body["max_port"])
        await manager.start(asset_dir, device=device)
        return await manager.status()
    except (ValueError, FileNotFoundError, KeyError, TypeError, SystemExit) as e:
        return _json_error(400, str(e))
    except (RuntimeError, TimeoutError) as e:
        return _json_error(500, str(e))


@rt("/stop", methods=["POST"])
async def stop():
    await manager.stop()
    return await manager.status()


@rt("/embeddings", methods=["POST"])
async def embeddings(request: Request):
    st = await manager.status()
    if not st["ready"]:
        return _json_error(
            503, "MAX not ready; POST /load with weight_path or checkpoint first"
        )
    try:
        body = await _read_json(request)
        smiles = body.get("smiles")
        if smiles is None:
            raise ValueError("smiles is required")
        inputs = [smiles] if isinstance(smiles, str) else list(smiles)
        if not inputs:
            raise ValueError("smiles is empty")
    except ValueError as e:
        return _json_error(400, str(e))

    payload_input: str | list[str] = inputs[0] if len(inputs) == 1 else inputs
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{manager.base_url}/v1/embeddings",
                json={"model": manager.model_id(), "input": payload_input},
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return _json_error(502, f"MAX embeddings failed: {e}")

    rows = sorted(data["data"], key=lambda row: row["index"])
    vectors = [row["embedding"] for row in rows]
    mode = st.get("mode") or "embedding"
    return {
        "mode": mode,
        "task": st.get("task"),
        "model": manager.model_id(),
        "embeddings": vectors,
        "predictions": [float(v[0]) for v in vectors] if mode == "property" else None,
    }


@rt("/decode", methods=["POST"])
async def decode(request: Request):
    st = await manager.status()
    if not st["ready"] or not manager.decode.ready:
        return _json_error(
            503, "decode not ready; POST /load with weight_path or checkpoint first"
        )
    try:
        body = await _read_json(request)
        embeddings = body.get("embeddings")
        if embeddings is None:
            raise ValueError("embeddings is required")
        if isinstance(embeddings[0], (int, float)):
            embeddings = [embeddings]
        result = manager.decode.decode_embeddings(embeddings)
        return {
            "model": manager.model_id(),
            "smiles": result["smiles"],
            "token_ids": result["token_ids"],
        }
    except (ValueError, TypeError, IndexError) as e:
        return _json_error(400, str(e))
    except RuntimeError as e:
        return _json_error(500, str(e))


@rt("/roundtrip", methods=["POST"])
async def roundtrip(request: Request):
    "Encode SMILES via MAX serve, then decode embeddings back to SMILES."
    st = await manager.status()
    if not st["ready"] or not manager.decode.ready:
        return _json_error(
            503, "MAX not ready; POST /load with weight_path or checkpoint first"
        )
    try:
        body = await _read_json(request)
        smiles = body.get("smiles")
        if smiles is None:
            raise ValueError("smiles is required")
        inputs = [smiles] if isinstance(smiles, str) else list(smiles)
        if not inputs:
            raise ValueError("smiles is empty")
    except ValueError as e:
        return _json_error(400, str(e))

    payload_input: str | list[str] = inputs[0] if len(inputs) == 1 else inputs
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{manager.base_url}/v1/embeddings",
                json={"model": manager.model_id(), "input": payload_input},
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return _json_error(502, f"MAX embeddings failed: {e}")

    rows = sorted(data["data"], key=lambda row: row["index"])
    vectors = [row["embedding"] for row in rows]
    try:
        decoded = manager.decode.decode_embeddings(vectors)
    except (ValueError, RuntimeError) as e:
        return _json_error(500, str(e))

    return {
        "model": manager.model_id(),
        "input_smiles": inputs,
        "embeddings": vectors,
        "decoded_smiles": decoded["smiles"],
        "token_ids": decoded["token_ids"],
    }

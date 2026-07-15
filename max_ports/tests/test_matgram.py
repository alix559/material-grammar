from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from matgram.app import create_app
from matgram.export_cache import prepare_assets
from matgram.max_client import sigmoid
from matgram.serve_manager import ServeManager
from matgram.settings import ControllerSettings
from matgram.state import SessionStore


class FakeMaxClient:
    async def predict(self, smiles: list[str], task: str):
        return [
            {"task": task, "smiles": value, "raw": -0.5, "prediction": -0.5}
            for value in smiles
        ]


class FakeManager:
    def __init__(self):
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def status(self):
        return {"running": True, "ready": True, "port": 8000}

    def client(self):
        return FakeMaxClient()


class ControllerTests(unittest.TestCase):
    def test_session_store_persists_and_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            store = SessionStore(path, "esol")
            store.update([{"task": "esol", "smiles": "CCO", "prediction": -1.0}])
            self.assertEqual(len(SessionStore(path, "esol").snapshot()["rows"]), 1)
            store.update(
                [{"task": "esol", "smiles": "C", "prediction": -0.5}],
                replace=True,
            )
            self.assertEqual(store.snapshot()["rows"][0]["smiles"], "C")

    def test_export_cache_skips_unchanged_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            base = root / "base"
            base.mkdir()
            (base / "bert_vocab_curated.txt").write_text("<bos>\n")
            output = root / "assets"

            def fake_export(**kwargs):
                out = kwargs["output_dir"]
                out.mkdir(parents=True, exist_ok=True)
                for name in (
                    "model_weights.safetensors",
                    "config.json",
                    "bert_vocab_curated.txt",
                ):
                    (out / name).write_text("ok")

            with patch("matgram.export_cache.export_finetune", fake_export):
                _, exported = prepare_assets(
                    checkpoint=checkpoint,
                    task="esol",
                    output_dir=output,
                    base_assets=base,
                )
                _, exported_again = prepare_assets(
                    checkpoint=checkpoint,
                    task="esol",
                    output_dir=output,
                    base_assets=base,
                )
            self.assertTrue(exported)
            self.assertFalse(exported_again)

    def test_api_prediction_updates_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"x")
            dashboard = root / "dist"
            dashboard.mkdir()
            (dashboard / "index.html").write_text("dashboard")
            settings = ControllerSettings(
                task="esol",
                checkpoint=checkpoint,
                assets_root=root / "assets",
                data_root=root / "data",
                base_assets=root / "base",
                dashboard_dist=dashboard,
            )
            app = create_app(
                settings, manager=FakeManager(), manage_max=False
            )
            with TestClient(app) as client:
                response = client.post(
                    "/api/predict", json={"smiles": ["CCO"], "replace": False}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["rows"][0]["smiles"], "CCO")
                self.assertEqual(client.get("/api/state").json()["task"], "esol")
                self.assertEqual(client.delete("/api/state").json()["rows"], [])

    def test_sigmoid_is_stable(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(1000), 0.999)
        self.assertLess(sigmoid(-1000), 0.001)


class FakeProcess:
    pid = 1234
    returncode = None

    async def wait(self):
        self.returncode = 0
        return 0


class ServeManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_serve_manager_starts_after_health_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ServeManager(
                max_root=root,
                asset_dir=root / "asset",
                architecture_path=root / "arch",
                device="cpu",
                port=8000,
                startup_timeout=1,
                log_path=root / "serve.log",
            )
            fake_process = FakeProcess()
            with (
                patch(
                    "matgram.serve_manager.asyncio.create_subprocess_exec",
                    AsyncMock(return_value=fake_process),
                ),
                patch(
                    "matgram.max_client.MaxClient.healthy",
                    AsyncMock(return_value=True),
                ),
            ):
                await manager.start()
            self.assertIs(manager.process, fake_process)
            manager.process = None
            if manager._log_handle:
                manager._log_handle.close()


if __name__ == "__main__":
    unittest.main()

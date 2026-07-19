"""Manage a ``max serve`` subprocess for one model-path at a time."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path
from typing import Any


class ServeManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        architecture_path: Path,
        max_port: int = 8000,
        device: str = "cpu",
        startup_timeout: float = 180.0,
        log_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.architecture_path = architecture_path
        self.max_port = max_port
        self.device = device
        self.startup_timeout = startup_timeout
        self.log_path = log_path or (repo_root / ".max_serve_api.log")
        self.process: asyncio.subprocess.Process | None = None
        self.asset_dir: Path | None = None
        self.model_path_arg: str | None = None
        self.started_at: float | None = None
        self.last_error: str | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.max_port}"

    def model_id(self) -> str:
        return self.model_path_arg or ""

    async def healthy(self) -> bool:
        if not self.process or self.process.returncode is not None:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base_url}/v1/models")
                return r.status_code == 200
        except Exception:
            return False

    async def start(self, asset_dir: Path, *, device: str | None = None) -> None:
        asset_dir = asset_dir.resolve()
        if not (asset_dir / "config.json").is_file():
            raise FileNotFoundError(f"missing config.json in {asset_dir}")

        await self.stop()
        if device is not None:
            self.device = device

        try:
            rel = asset_dir.relative_to(self.repo_root)
            model_path = f"./{rel.as_posix()}" if not str(rel).startswith(".") else str(rel)
        except ValueError:
            model_path = str(asset_dir)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        cmd = [
            "max",
            "serve",
            "--model-path",
            model_path,
            "--custom-architectures",
            str(self.architecture_path),
            "--quantization-encoding",
            "float32",
            f"--devices={self.device}",
            "--port",
            str(self.max_port),
        ]
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.repo_root),
            stdout=self._log_handle,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self.asset_dir = asset_dir
        self.model_path_arg = model_path
        self.started_at = time.time()
        self.last_error = None

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                self.last_error = self._log_tail()
                raise RuntimeError(
                    f"MAX Serve exited ({self.process.returncode}): {self.last_error}"
                )
            if await self.healthy():
                return
            await asyncio.sleep(1)

        self.last_error = f"MAX Serve not ready after {self.startup_timeout:.0f}s"
        await self.stop()
        raise TimeoutError(f"{self.last_error}\n{self._log_tail()}")

    async def stop(self) -> None:
        proc = self.process
        if proc and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=20)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
        self.process = None
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    async def status(self) -> dict[str, Any]:
        running = bool(self.process and self.process.returncode is None)
        ready = await self.healthy() if running else False
        mode = None
        task = None
        if self.asset_dir and (self.asset_dir / "config.json").is_file():
            cfg = json.loads((self.asset_dir / "config.json").read_text())
            mode = cfg.get("smi_ted_output", "embedding")
            task = cfg.get("smi_ted_task")
        return {
            "running": running,
            "ready": ready,
            "pid": self.process.pid if running and self.process else None,
            "maxPort": self.max_port,
            "device": self.device,
            "assetDir": str(self.asset_dir) if self.asset_dir else None,
            "modelId": self.model_id() or None,
            "mode": mode,
            "task": task,
            "uptimeSeconds": (
                time.time() - self.started_at
                if running and self.started_at is not None
                else None
            ),
            "lastError": self.last_error,
            "maxUrl": self.base_url if running else None,
        }

    def _log_tail(self, max_bytes: int = 4096) -> str:
        if not self.log_path.is_file():
            return "no log"
        with self.log_path.open("rb") as handle:
            handle.seek(max(0, self.log_path.stat().st_size - max_bytes))
            return handle.read().decode("utf-8", errors="replace").strip()

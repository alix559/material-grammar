from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Any, BinaryIO

from .max_client import MaxClient


class ServeManager:
    def __init__(
        self,
        *,
        max_root: Path,
        asset_dir: Path,
        architecture_path: Path,
        device: str,
        port: int,
        startup_timeout: float,
        log_path: Path,
    ):
        self.max_root = max_root
        self.asset_dir = asset_dir
        self.architecture_path = architecture_path
        self.device = device
        self.port = port
        self.startup_timeout = startup_timeout
        self.log_path = log_path
        self.process: asyncio.subprocess.Process | None = None
        self.started_at: float | None = None
        self.last_error: str | None = None
        self._log_handle: BinaryIO | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def model_name(self) -> str:
        return str(self.asset_dir)

    def client(self) -> MaxClient:
        return MaxClient(self.base_url, self.model_name)

    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        command = [
            "max",
            "serve",
            "--model-path",
            str(self.asset_dir),
            "--custom-architectures",
            str(self.architecture_path),
            "--quantization-encoding",
            "float32",
            f"--devices={self.device}",
            "--port",
            str(self.port),
        ]
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.max_root,
            stdout=self._log_handle,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self.started_at = time.time()
        self.last_error = None

        deadline = time.monotonic() + self.startup_timeout
        client = self.client()
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                self.last_error = self._log_tail()
                raise RuntimeError(
                    f"MAX Serve exited with code {self.process.returncode}: "
                    f"{self.last_error}"
                )
            if await client.healthy():
                await asyncio.sleep(0.25)
                if self.process.returncode is None:
                    return
            await asyncio.sleep(1)

        self.last_error = f"MAX Serve was not ready after {self.startup_timeout:.0f}s"
        await self.stop()
        raise TimeoutError(self.last_error)

    async def stop(self) -> None:
        process = self.process
        if process and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=15)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        self.process = None
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    async def status(self) -> dict[str, Any]:
        running = bool(self.process and self.process.returncode is None)
        ready = await self.client().healthy() if running else False
        return {
            "running": running,
            "ready": ready,
            "pid": self.process.pid if running and self.process else None,
            "port": self.port,
            "device": self.device,
            "model": self.model_name,
            "uptimeSeconds": (
                time.time() - self.started_at
                if running and self.started_at is not None
                else None
            ),
            "lastError": self.last_error,
        }

    def _log_tail(self, max_bytes: int = 4096) -> str:
        if not self.log_path.is_file():
            return "no MAX Serve log available"
        with self.log_path.open("rb") as handle:
            handle.seek(max(0, self.log_path.stat().st_size - max_bytes))
            return handle.read().decode("utf-8", errors="replace").strip()

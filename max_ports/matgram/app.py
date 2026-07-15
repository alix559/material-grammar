from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .export_cache import prepare_assets
from .serve_manager import ServeManager
from .settings import MAX_PORT_ROOT, ControllerSettings, TASKS
from .state import SessionStore


class PredictionRequest(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=1000)
    replace: bool = False


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    async def publish(self, event: str, data: dict[str, Any]) -> None:
        message = {"event": event, "data": json.dumps(data)}
        for queue in tuple(self._subscribers):
            await queue.put(message)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=16)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)


class ControllerRuntime:
    def __init__(
        self,
        settings: ControllerSettings,
        *,
        manager: ServeManager | None = None,
        manage_max: bool = True,
    ):
        self.settings = settings
        self.store = SessionStore(settings.session_path, settings.task)
        self.broker = EventBroker()
        self.manage_max = manage_max
        self.exported = False
        self.manager = manager or ServeManager(
            max_root=MAX_PORT_ROOT,
            asset_dir=settings.asset_dir,
            architecture_path=settings.architecture_path,
            device=settings.device,
            port=settings.max_port,
            startup_timeout=settings.startup_timeout,
            log_path=settings.data_root / "max-serve.log",
        )

    async def start(self) -> None:
        if not self.manage_max:
            return
        _, self.exported = await asyncio.to_thread(
            prepare_assets,
            checkpoint=self.settings.checkpoint,
            task=self.settings.task,
            output_dir=self.settings.asset_dir,
            base_assets=self.settings.base_assets,
        )
        await self.manager.start()

    async def stop(self) -> None:
        if self.manage_max:
            await self.manager.stop()

    async def status(self) -> dict[str, Any]:
        serve = await self.manager.status()
        return {
            "task": self.settings.task,
            "taskInfo": TASKS[self.settings.task],
            "checkpoint": str(self.settings.checkpoint),
            "assetDir": str(self.settings.asset_dir),
            "exported": self.exported,
            "serve": serve,
            "predictionCount": len(self.store.snapshot()["rows"]),
        }


def create_app(
    settings: ControllerSettings,
    *,
    manager: ServeManager | None = None,
    manage_max: bool = True,
) -> FastAPI:
    runtime = ControllerRuntime(settings, manager=manager, manage_max=manage_max)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.start()
        await runtime.broker.publish("status", await runtime.status())
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="matgram", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return await runtime.status()

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return runtime.store.snapshot()

    @app.get("/api/results")
    async def results() -> dict[str, Any]:
        return runtime.store.snapshot()

    @app.delete("/api/state")
    async def clear_state() -> dict[str, Any]:
        snapshot = runtime.store.clear()
        await runtime.broker.publish("state", snapshot)
        return snapshot

    @app.post("/api/predict")
    async def predict(request: PredictionRequest) -> dict[str, Any]:
        smiles = [value.strip() for value in request.smiles if value.strip()]
        if not smiles:
            raise HTTPException(status_code=422, detail="no non-empty SMILES supplied")
        serve_status = await runtime.manager.status()
        if not serve_status["ready"]:
            raise HTTPException(status_code=503, detail="MAX Serve is not ready")
        try:
            rows = await runtime.manager.client().predict(smiles, settings.task)
        except Exception as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        snapshot = runtime.store.update(rows, replace=request.replace)
        await runtime.broker.publish("state", snapshot)
        return snapshot

    @app.get("/api/events")
    async def events() -> EventSourceResponse:
        async def stream():
            yield {"event": "state", "data": json.dumps(runtime.store.snapshot())}
            async with runtime.broker.subscribe() as queue:
                while True:
                    try:
                        yield await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": "{}"}

        return EventSourceResponse(stream())

    if settings.dashboard_dist.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=settings.dashboard_dist, html=True),
            name="dashboard",
        )
    else:
        @app.get("/")
        async def dashboard_missing() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "dashboard build not found",
                    "expected": str(settings.dashboard_dist),
                },
                status_code=503,
            )

    return app

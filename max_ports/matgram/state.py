from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, path: Path, task: str):
        self.path = path
        self.task = task
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "task": task,
            "updatedAt": None,
            "rows": [],
        }
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            value = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if value.get("task") == self.task and isinstance(value.get("rows"), list):
            self._state = value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def update(
        self, rows: list[dict[str, Any]], *, replace: bool = False
    ) -> dict[str, Any]:
        now = time.time()
        normalized = []
        for row in rows:
            normalized.append(
                {
                    **row,
                    "id": uuid.uuid4().hex,
                    "createdAt": now,
                }
            )
        with self._lock:
            if replace:
                self._state["rows"] = normalized
            else:
                self._state["rows"].extend(normalized)
            self._state["updatedAt"] = now
            self._save()
            return self.snapshot()

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._state = {
                "task": self.task,
                "updatedAt": time.time(),
                "rows": [],
            }
            self._save()
            return self.snapshot()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(self._state, indent=2) + "\n")
        os.replace(temporary, self.path)

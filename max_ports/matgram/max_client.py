from __future__ import annotations

import math
from typing import Any

import httpx


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class MaxClient:
    def __init__(self, base_url: str, model_name: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.is_success
        except httpx.HTTPError:
            return False

    async def predict(self, smiles: list[str], task: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model_name, "input": smiles},
            )
            response.raise_for_status()
        values = response.json()["data"]
        values.sort(key=lambda row: row["index"])

        rows: list[dict[str, Any]] = []
        for smi, item in zip(smiles, values, strict=True):
            vector = item.get("embedding") or []
            if not vector:
                raise ValueError(f"MAX returned an empty prediction for {smi!r}")
            raw = float(vector[0])
            prediction = sigmoid(raw) if task == "bbbp" else raw
            row: dict[str, Any] = {
                "task": task,
                "smiles": smi,
                "raw": raw,
                "prediction": prediction,
            }
            if task == "bbbp":
                row.update(logit=raw, probability=prediction)
            rows.append(row)
        return rows

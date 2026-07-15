from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TaskName = Literal["esol", "bbbp", "lipo"]

MAX_PORT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MAX_PORT_ROOT.parent

TASKS: dict[str, dict[str, str]] = {
    "esol": {
        "label": "ESOL",
        "task_type": "regression",
        "unit": "log10(mol/L)",
    },
    "lipo": {
        "label": "Lipophilicity",
        "task_type": "regression",
        "unit": "logP/logD",
    },
    "bbbp": {
        "label": "BBBP",
        "task_type": "classification",
        "unit": "probability",
    },
}


@dataclass(frozen=True)
class ControllerSettings:
    task: TaskName
    checkpoint: Path
    device: str = "cpu"
    max_port: int = 8000
    host: str = "0.0.0.0"
    port: int = 8080
    startup_timeout: float = 300.0
    assets_root: Path = MAX_PORT_ROOT / "model_assets"
    data_root: Path = REPOSITORY_ROOT / "mat_dataViz" / "data"
    base_assets: Path = (
        MAX_PORT_ROOT / "model_assets" / "ibm-research_materials.smi-ted"
    )
    dashboard_dist: Path = (
        REPOSITORY_ROOT / "mat_dataViz" / "hello-framework" / "dist"
    )
    architecture_path: Path = MAX_PORT_ROOT / "materials_smi_ted"

    @property
    def asset_dir(self) -> Path:
        return self.assets_root / f"smi-ted-{self.task}"

    @property
    def session_path(self) -> Path:
        return self.data_root / "session.json"

    @classmethod
    def from_env(
        cls,
        *,
        task: TaskName,
        checkpoint: Path,
        device: str = "cpu",
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> "ControllerSettings":
        return cls(
            task=task,
            checkpoint=checkpoint,
            device=device,
            max_port=int(os.getenv("MAX_PORT", "8000")),
            host=host,
            port=port,
            startup_timeout=float(os.getenv("MAX_STARTUP_TIMEOUT", "300")),
            assets_root=Path(
                os.getenv("MATGRAM_ASSETS_DIR", str(MAX_PORT_ROOT / "model_assets"))
            ),
            data_root=Path(
                os.getenv(
                    "MATGRAM_DATA_DIR",
                    str(REPOSITORY_ROOT / "mat_dataViz" / "data"),
                )
            ),
            base_assets=Path(
                os.getenv(
                    "MATGRAM_BASE_ASSETS",
                    str(
                        MAX_PORT_ROOT
                        / "model_assets"
                        / "ibm-research_materials.smi-ted"
                    ),
                )
            ),
            dashboard_dist=Path(
                os.getenv(
                    "MATGRAM_DASHBOARD_DIST",
                    str(
                        REPOSITORY_ROOT
                        / "mat_dataViz"
                        / "hello-framework"
                        / "dist"
                    ),
                )
            ),
            architecture_path=Path(
                os.getenv(
                    "MATGRAM_ARCHITECTURE_PATH",
                    str(MAX_PORT_ROOT / "materials_smi_ted"),
                )
            ),
        )

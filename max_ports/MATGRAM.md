# matgram

`matgram` is a Pixi-installable Python package for serving IBM SMI-TED finetune
checkpoints with MAX, loading `.pt` weights into exportable assets, and running a
live property dashboard.

## Install (this repo)

From `max_ports/`:

```bash
pixi install
```

Pixi installs `matgram` as an editable package and pulls MAX, PyTorch CPU, and
the API stack from the workspace `pixi.toml`.

One-time base assets (vocab + pretrained config for export):

```bash
pixi run setup-model
```

Build the dashboard static files once:

```bash
npm --prefix ../mat_dataViz/hello-framework install
npm --prefix ../mat_dataViz/hello-framework run build
```

## Run

Start the controller, MAX Serve, and dashboard:

```bash
pixi run matgram dashboard \
  --task esol \
  --checkpoint finetune_ckpts/esol/smoke_MODEL_STATE.pt \
  --device cpu
```

Open <http://localhost:8080>.

Submit predictions from another terminal:

```bash
pixi run matgram predict --smiles CCO
pixi run matgram status
pixi run matgram clear
pixi run matgram export-results --output mat_dataViz/data/results.json
```

## Use as a library

```python
from pathlib import Path

from matgram import create_app
from matgram.checkpoint_export import export_finetune
from matgram.settings import ControllerSettings

export_finetune(
    checkpoint=Path("finetune_ckpts/esol/model.pt"),
    task="esol",
    output_dir=Path("model_assets/smi-ted-esol"),
    base_assets=Path("model_assets/ibm-research_materials.smi-ted"),
)

settings = ControllerSettings.from_env(
    task="esol",
    checkpoint=Path("finetune_ckpts/esol/model.pt"),
)
app = create_app(settings)
```

## Use in another Pixi project

Add the package and the same MAX/PyTorch conda dependencies to your
`pixi.toml`:

```toml
[dependencies]
max = ">=26.3.0,<27"
max-pipelines = ">=26.3.0,<27"
python = ">=3.12,<3.15"
pytorch-cpu = ">=2.4"
safetensors = ">=0.4"
# ... fastapi, uvicorn, click, httpx, sse-starlette, transformers, etc.

[pypi-dependencies]
matgram = { path = "../material_grammar/max_ports", editable = true }
```

Point `MATGRAM_ARCHITECTURE_PATH` at a checkout of `materials_smi_ted` and mount
your checkpoint path. Docker is optional; Pixi is the supported workflow here.

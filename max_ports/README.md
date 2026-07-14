# mat-gram01

> Rebuild [IBM SMI-TED](https://huggingface.co/ibm-research/materials.smi-ted) as a [MAX](https://docs.modular.com/max/) custom architecture — one notebook at a time.

## What are we building?

A molecule arrives as a **SMILES** string — text like `CCO` for ethanol. We want a fixed **768-d embedding** we can serve from MAX — or, after IBM finetune export, a **property prediction** (ESOL / BBBP / lipo).

```text
SMILES → tokens → MoLEncoder → AutoEncoder → 768-d vector
                                         └─(property mode)→ Net → scalar
```

See [`PORT.md`](PORT.md) for finetune → `export-finetune` → `serve-esol` / `serve-bbbp` / `serve-lipo`.

This is a literate rewrite of `materials_smi_ted/` using [nbdev](https://nbdev.fast.ai/) notebooks in Jeremy Howard’s style: explain a little, code a little, export a little.

## Notebooks (`nbs/`)

| Notebook | Module | Topic |
|----------|--------|-------|
| `00_config` | `model_config` | HF + MAX config |
| `01_tokenizer` | `tokenizer` | SMILES regex tokenizer |
| `02_weights` | `weight_adapters` | Safetensors filter |
| `03_graph` | `graph` | Network math |
| `04_batch` | `batch_processor` | Pad to `max_len=202` |
| `05_model` | `model` | Pipeline load/execute |
| `06_arch` | `arch` | `SupportedArchitecture` |
| `07_package` | `__init__` | Public exports |
| `08_finetune` | *(not exported)* | GPU IBM finetune → `finetune_ckpts/` |

## Dev

```sh
pip install -e .
nbdev_export
```

```python
from mat_gram01 import ARCHITECTURES
ARCHITECTURES[0].name  # 'SmiTedModel'
```

Serve with the literate package (same assets as the reference port):

```sh
max serve \
  --model-path ./model_assets/ibm-research_materials.smi-ted \
  --custom-architectures ./mat_gram01 \
  --quantization-encoding float32
```

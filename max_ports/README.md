# mat-gram01 / SMI-TED on MAX

Rebuild [IBM SMI-TED](https://huggingface.co/ibm-research/materials.smi-ted) as a
[MAX](https://docs.modular.com/max/) custom architecture — then serve pretrained
or finetuned weights over HTTP.

## Setup

```bash
pixi install
pixi run setup-model   # downloads pretrained weights + vocab into model_assets/
```

## HTTP API

```bash
pixi run api
# → http://127.0.0.1:8080
```

Load pretrained (768-d embeddings):

```bash
curl -X POST http://127.0.0.1:8080/load \
  -H 'Content-Type: application/json' \
  -d '{"weight_path":"./model_assets/ibm-research_materials.smi-ted","device":"cpu"}'
```

Load a finetune `.pt` (exports + serves property mode):

```bash
curl -X POST http://127.0.0.1:8080/load \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint":"finetune_ckpts/esol/YOUR.pt","task":"esol","device":"cpu"}'
```

Embed / predict / decode:

```bash
curl -X POST http://127.0.0.1:8080/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"smiles":["CCO","c1ccccc1"]}'

# Decode 768-d embeddings back to SMILES
curl -X POST http://127.0.0.1:8080/decode \
  -H 'Content-Type: application/json' \
  -d '{"embeddings":[[0.1, 0.2, ...]]}'

# Encode then decode (round-trip smoke test)
curl -X POST http://127.0.0.1:8080/roundtrip \
  -H 'Content-Type: application/json' \
  -d '{"smiles":["CCO","c1ccccc1"]}'
```

| Task | Property |
|------|----------|
| esol | aqueous solubility, log₁₀(mol/L) |
| bbbp | blood–brain barrier logit |
| lipo | lipophilicity (logP / logD) |

Notebook client: `nbs/09_serve.ipynb`.

Railway (HF-hosted ESOL weights, no Git LFS): see [`docs/railway.md`](docs/railway.md).

## Notebooks (`nbs/`)

| Notebook | Module | Topic |
|----------|--------|-------|
| `00_config` … `07_package` | `mat_gram01/*` | Literate MAX architecture |
| `08_finetune` | *(not exported)* | GPU IBM finetune → `finetune_ckpts/` |
| `09_serve` | *(not exported)* | HTTP API client |

```bash
pixi run export-nbs    # nbdev_export → mat_gram01/
```

## Layout

```text
nbs/                 notebooks that build the MAX architecture
mat_gram01/          nbdev export (custom arch)
serve_api/           FastHTML JSON controller (`pixi run api`)
model_assets/        weight dirs
finetune_ckpts/      raw IBM .pt inputs
scripts/             setup-model, export helper, compare-hf
```

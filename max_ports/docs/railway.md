# Deploy ESOL serve API on Railway

Weights stay **out of Git** (~1.1GB). Host the exported MAX asset dir on
Hugging Face; Railway downloads it at boot and auto-loads the model.

## 1. Export + upload (once, on a machine that has the checkpoint)

```bash
cd max_ports
pixi install
pixi run setup-model   # IBM base assets (needed for export)

# Export if you do not already have model_assets/smi-ted-esol/
pixi run python scripts/export_finetune_to_max.py \
  --checkpoint finetune_ckpts/esol/smi-ted-Light-Finetune_seed0_esol_epoch=3_valloss=0.7972.pt \
  --task esol

export HF_TOKEN=hf_...   # write token
pixi run upload-esol -- --repo YOUR_USER/smi-ted-esol --private
# → https://huggingface.co/YOUR_USER/smi-ted-esol
```

Uploaded files: `model_weights.safetensors`, `config.json`, `bert_vocab_curated.txt`.

## 2. Push code to GitHub

Do **not** commit `.pt` / `.safetensors` (already in `.gitignore`).

If this repo is a monorepo, set the Railway service root to `max_ports`.

## 3. Railway service

1. New project → Deploy from GitHub → select this repo (root = `max_ports`).
2. Builder uses [`Dockerfile`](../Dockerfile) via [`railway.toml`](../railway.toml).
3. Variables:

| Variable | Value |
|----------|--------|
| `MATGRAM_HF_REPO` | `YOUR_USER/smi-ted-esol` |
| `MATGRAM_AUTO_LOAD` | `1` |
| `MATGRAM_DEVICE` | `cpu` |
| `HF_TOKEN` | (if the HF repo is private) |
| `MATGRAM_STARTUP_TIMEOUT` | `600` (optional; first MAX boot can be slow) |

4. **Volume** (recommended so redeploys skip the 1.1GB download):
   - Mount at `/data`
   - Set `HF_HOME=/data/huggingface`
   - Set `MATGRAM_ASSETS_DIR=/data/model_assets`

5. **RAM:** plan **≥8GB**. CPU-only; no GPU on Railway.

`PORT` is injected by Railway; the API binds to `0.0.0.0:$PORT`.

## 4. Smoke test

First deploy downloads weights then starts `max serve` in the background.
`GET /health` responds immediately; wait until `GET /status` shows `"ready": true`.

```bash
curl -s "$RAILWAY_URL/health"
curl -s "$RAILWAY_URL/status" | jq

curl -s -X POST "$RAILWAY_URL/embeddings" \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO"}' | jq
```

No `POST /load` on Railway — auto-load handles it.

## Local vs production

| | Local (`pixi run api`) | Railway |
|--|------------------------|---------|
| Weights | `POST /load` with local `.pt` | HF download + auto-load |
| Client | `nbs/09_serve.ipynb` → `http://127.0.0.1:8080` | Same notebook with `API = "https://….up.railway.app"`; skip `/load` |

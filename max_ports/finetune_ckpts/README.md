# IBM finetune checkpoints

Place GPU-produced IBM finetune `.pt` files here:

```text
finetune_ckpts/
  esol/*.pt
  bbbp/*.pt
  lipo/*.pt
```

Then load via the API (exports + serves):

```bash
pixi run api

curl -X POST http://127.0.0.1:8080/load \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint":"finetune_ckpts/esol/YOUR.pt","task":"esol","device":"cpu"}'
```

Or open `nbs/08_finetune.ipynb` (train) / `nbs/09_serve.ipynb` (API client).

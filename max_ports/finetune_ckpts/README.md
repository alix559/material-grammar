# IBM finetune checkpoints

Place GPU-produced IBM finetune `.pt` files here before export:

```text
finetune_ckpts/
  esol/*.pt
  bbbp/*.pt
  lipo/*.pt
```

**Interactive path:** open [`nbs/08_finetune.ipynb`](../nbs/08_finetune.ipynb) on a GPU machine — it trains, then copies the best `.pt` into this folder.

Then:

```bash
pixi run export-finetune -- --checkpoint finetune_ckpts/esol/YOUR.pt --task esol
```
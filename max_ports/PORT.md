# SMI-TED MAX Port

Port of [ibm-research/materials.smi-ted](https://huggingface.co/ibm-research/materials.smi-ted) for embeddings generation and finetuned property serve.

## Status

| Phase | Status |
|-------|--------|
| Phase 1 — planning | Done |
| Phase 2 — graph + weights | Graph dry-build passes; linear-attention normalizer (`Z`) added; optional `Net` head |
| Phase 3 — GPU verify | Blocked (no CUDA on this machine) |
| Phase 4 — property serve | Export + graph property mode ready; needs GPU finetune checkpoints |

## Architecture summary

- **Donor:** `bert` (embeddings task), heavily rewritten
- **Graph:** MoLEncoder (12× linear attention + RoPE) → flatten/pad → decoder autoencoder encoder → 768-d embedding → optional `Net` → `[batch, 1]` property
- **Weights:** `model_weights.safetensors` (encode path + `net.*`; LM / AE-decoder dropped)
- **Tokenizer:** SMILES regex + `bert_vocab_curated.txt`

## Serve embeddings (pretrained)

```bash
cd max_ports
pixi run serve
# or GPU: pixi run serve-gpu
```

## Finetune → MAX property serve (ESOL / BBBP / lipo)

IBM full finetune updates encoder + AE encoder + `Net`. Export that checkpoint into a MAX asset dir, then serve. In property mode `/v1/embeddings` returns a **length-1** vector (the prediction; BBBP is a logit).

### 1. Finetune on a GPU host

```bash
cd vendor/ibm_materials/models/smi_ted
# place smi-ted-Light_40.pt + fast_transformers under finetune/smi_ted_light/
cd finetune/smi_ted_light/esol && bash run_finetune_esol.sh
cd ../bbbp && bash run_finetune_bbbp.sh
cd ../lipo && bash run_finetune_lipo.sh
```

Copy best `*Finetune*.pt` files to e.g. `max_ports/finetune_ckpts/{esol,bbbp,lipo}/`.

### 2. Export to MAX assets

```bash
cd max_ports
pixi run setup-model
pixi run export-finetune -- \
  --checkpoint finetune_ckpts/esol/YOUR_FINETUNE.pt --task esol
# repeat for --task bbbp and --task lipo
```

Writes `model_assets/smi-ted-{task}/` (`model_weights.safetensors`, patched `config.json`, vocab, `finetune_source.pt`).

### 3. Serve and predict

```bash
pixi run serve-esol          # or serve-bbbp / serve-lipo (+ -gpu variants)
pixi run predict-finetuned -- --task esol --smiles CCO
pixi run compare-finetune -- --task esol --smiles CCO
```

| Task | Property | Served value |
|------|----------|--------------|
| esol | aqueous solubility | log₁₀(mol/L) |
| bbbp | blood–brain barrier | logit (`--probability` → sigmoid) |
| lipo | lipophilicity | logP / logD |

Pretrained 768-d embeddings remain available via `pixi run serve` (original asset dir).

## HF reference

```bash
cd max_ports
pixi run compare-hf -- --smiles CCO
```

## Known gaps (before parity)

1. Embedding parity still needs a live GPU compare.
2. Property parity needs real IBM finetune checkpoints + `pixi run compare-finetune`.

## Delta list (HF vs bert donor)

| Component | Change |
|-----------|--------|
| Attention | Performer linear attention + RoPE (not dot-product) |
| Block wiring | Post-attn residual, pre-FFN norm (fast_transformers layer) |
| Embeddings | Token only (no position/type) |
| Head | Decoder autoencoder encoder (not pooler); optional `Net` for property mode |
| Tokenizer | SMILES regex vocab |

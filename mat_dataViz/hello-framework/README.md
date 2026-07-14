# Molecular Property Evaluation Dashboard

An Observable Framework dashboard for evaluating IBM SMI-TED predictions on
the MoleculeNet ESOL, lipophilicity, and BBBP test sets.

The dashboard always loads the real test labels from the vendored IBM datasets.
Prediction files are optional: when they are absent, the site builds with a
clear empty state and does not invent metrics.

## Run locally

From `mat_dataViz`:

```bash
pixi run install-deps
pixi run dev
```

Open <http://localhost:3000>. Run the production checks with:

```bash
pixi run build
cd hello-framework && npm test
```

## Prediction files

Place prediction exports in:

```text
mat_dataViz/predictions/
├── esol.json
├── lipo.json
└── bbbp.json
```

Each file is the JSON array emitted by
`max_ports/scripts/predict_finetuned.py`. Rows are joined to test labels by
SMILES:

```json
[
  {
    "smiles": "CCO",
    "task": "esol",
    "prediction": -0.42
  }
]
```

For BBBP, the predictor also emits `probability`; the dashboard prefers that
field over a raw logit.

Set `MATERIAL_PREDICTIONS_DIR` to read these files from another directory.
Clear the Observable loader cache after changing prediction files:

```bash
cd mat_dataViz/hello-framework && npm run clean
```

## Generate an evaluation export

The three fine-tuned models are served separately. Export one task at a time,
restarting MAX Serve with the matching asset:

```bash
mkdir -p mat_dataViz/predictions
cd max_ports

# Terminal 1
pixi run serve-esol

# Terminal 2
pixi run predict-finetuned -- \
  --task esol \
  --input-csv ../vendor/ibm_materials/models/smi_ted/finetune/moleculenet/esol/test.csv \
  --output-json ../mat_dataViz/predictions/esol.json
```

Repeat with the matching server, dataset, and output:

```bash
# BBBP: run `pixi run serve-bbbp` first
pixi run predict-finetuned -- \
  --task bbbp \
  --input-csv ../vendor/ibm_materials/models/smi_ted/finetune/moleculenet/bbbp/test.csv \
  --output-json ../mat_dataViz/predictions/bbbp.json

# Lipophilicity: run `pixi run serve-lipo` first
pixi run predict-finetuned -- \
  --task lipo \
  --input-csv ../vendor/ibm_materials/models/smi_ted/finetune/moleculenet/lipophilicity/test.csv \
  --output-json ../mat_dataViz/predictions/lipo.json
```

These commands require the corresponding exported model assets under
`max_ports/model_assets/smi-ted-{task}`. After writing files, run
`npm run clean` and rebuild or restart the preview.

## Metrics

- ESOL and lipophilicity: RMSE, MAE, and R².
- BBBP: ROC-AUC and accuracy at a probability threshold of 0.5.

Metrics use only rows with both an observed label and a finite prediction.
Prediction coverage is shown separately so partial exports cannot be mistaken
for a full evaluation.

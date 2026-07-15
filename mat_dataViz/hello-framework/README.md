# Molecular Property Evaluation Dashboard

An Observable Framework dashboard for evaluating IBM SMI-TED predictions on
the MoleculeNet ESOL, lipophilicity, and BBBP test sets.

The dashboard loads real test labels from the vendored IBM datasets and receives
live prediction sessions from the `matgram` Pixi package over HTTP and
Server-Sent Events.

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

## Live matgram

Production builds load labels statically, then fetch prediction state from the
same-origin `matgram` server. The dashboard subscribes to `/api/events` and
updates immediately after CLI or form submissions; prediction JSON files and
rebuilds are no longer required.

From `max_ports/`:

```bash
pixi run matgram dashboard --task esol --checkpoint finetune_ckpts/esol/model.pt
```

For local frontend development, run `matgram dashboard` on port 8080 and set the
browser global `MATGRAM_CONTROLLER_URL` to `http://localhost:8080`, or serve the
built `dist/` directory directly through matgram.

See [`max_ports/MATGRAM.md`](../../max_ports/MATGRAM.md). Docker is optional;
see repository-level `DOCKER.md`.

## Metrics

- ESOL and lipophilicity: RMSE, MAE, and R².
- BBBP: ROC-AUC and accuracy at a probability threshold of 0.5.

Metrics use only rows with both an observed label and a finite prediction.
Prediction coverage is shown separately so partial exports cannot be mistaken
for a full evaluation.

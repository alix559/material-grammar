---
theme: dashboard
title: Molecular Property Evaluation
toc: false
---

```js
import {
  classificationMetrics,
  pairedRows,
  regressionMetrics,
  rocCurve
} from "./components/property-metrics.js";
import {
  connectSession,
  fetchSession,
  mergeSession,
  submitPredictions
} from "./components/evaluation-client.js";
import {predictionForm} from "./components/prediction-form.js";

const baseline = await FileAttachment("data/evaluations.json").json();
const controllerBaseUrl = globalThis.MATGRAM_CONTROLLER_URL ?? "";
```

```js
const liveSession = Generators.observe((notify) => {
  let connection = "connecting";
  let current = {task: null, updatedAt: null, rows: [], connection};
  notify(current);
  fetchSession(controllerBaseUrl)
    .then((state) => {
      current = {...state, connection};
      notify(current);
    })
    .catch((error) => {
      connection = "offline";
      current = {...current, connection, error: error.message};
      notify(current);
    });
  return connectSession({
    baseUrl: controllerBaseUrl,
    onState: (state) => {
      current = {...state, connection};
      notify(current);
    },
    onConnection: (value) => {
      connection = value;
      current = {...current, connection};
      notify(current);
    }
  });
});
```

```js
const evaluation = mergeSession(baseline, liveSession);
const selectedTask = liveSession.task ?? "esol";
const spec = evaluation.tasks[selectedTask];
const rows = evaluation.rows.filter((row) => row.task === selectedTask);
const paired = pairedRows(rows);
const coverage = rows.length ? paired.length / rows.length : 0;
const metrics =
  spec.type === "classification"
    ? classificationMetrics(rows)
    : regressionMetrics(rows);
const curve = spec.type === "classification" ? rocCurve(rows) : [];

const taskCopy = {
  esol: "Predicts how readily a molecule dissolves in water.",
  lipo: "Predicts whether a molecule favors an oily environment over water.",
  bbbp: "Predicts the probability that a molecule crosses the blood–brain barrier."
};
const metric = (value, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";
const percent = (value) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const sessionRows = evaluation.session.rows;
const predictionInput = predictionForm({
  onSubmit: (smiles) =>
    submitPredictions(smiles, {baseUrl: controllerBaseUrl})
});
```

<div class="dashboard-header">
  <div>
    <div class="eyebrow">IBM SMI-TED · MoleculeNet test sets</div>
    <h1>Molecular Property Evaluation</h1>
    <p>${taskCopy[selectedTask]}</p>
  </div>
  <div class="connection-state ${liveSession.connection}">${liveSession.connection}</div>
</div>

<div class="grid grid-cols-2">
  <div class="card prediction-card">
    <div class="eyebrow">Live MAX inference</div>
    <h2>Predict molecular properties</h2>
    ${predictionInput}
  </div>
  <div class="card prediction-summary">
    <div class="eyebrow">Current session</div>
    <span class="big">${sessionRows.length.toLocaleString()}</span>
    <span class="metric-note">predictions · ${spec.label} · updates stream automatically</span>
  </div>
</div>

```js
const primaryMetric =
  spec.type === "classification" ? metrics.rocAuc : metrics.rmse;
const secondaryMetric =
  spec.type === "classification" ? metrics.accuracy : metrics.r2;
const primaryLabel = spec.type === "classification" ? "ROC-AUC" : "RMSE";
const secondaryLabel =
  spec.type === "classification" ? "Accuracy @ 0.5" : "R²";
```

<div class="grid grid-cols-4">
  <div class="card metric-card">
    <h2>Test molecules</h2>
    <span class="big">${rows.length.toLocaleString()}</span>
    <span class="metric-note">${spec.description}</span>
  </div>
  <div class="card metric-card">
    <h2>Prediction coverage</h2>
    <span class="big">${percent(coverage)}</span>
    <span class="metric-note">${paired.length.toLocaleString()} of ${rows.length.toLocaleString()} matched</span>
  </div>
  <div class="card metric-card">
    <h2>${primaryLabel}</h2>
    <span class="big">${metric(primaryMetric)}</span>
    <span class="metric-note">${paired.length ? "computed on matched predictions" : "waiting for predictions"}</span>
  </div>
  <div class="card metric-card">
    <h2>${secondaryLabel}</h2>
    <span class="big">${metric(secondaryMetric)}</span>
    <span class="metric-note">${spec.type === "classification" ? "higher is better" : "1.0 is a perfect fit"}</span>
  </div>
</div>

<div class="empty-state">
  <strong>${sessionRows.length ? `${sessionRows.length} live ${spec.label} predictions loaded.` : `No ${spec.label} predictions yet.`}</strong>
  <span>${paired.length
    ? `${paired.length} predictions match labeled MoleculeNet rows and contribute to benchmark metrics.`
    : "Use the form above or the matgram predict CLI. Benchmark metrics appear when an input matches a labeled test molecule."
  }</span>
</div>

```js
function targetDistribution(data, {width} = {}) {
  if (spec.type === "classification") {
    const classes = data.map((row) => ({
      ...row,
      class: row.actual === 1 ? "Penetrating" : "Non-penetrating"
    }));
    return Plot.plot({
      title: "Observed class balance",
      width,
      height: 300,
      x: {label: null},
      y: {grid: true, label: "Molecules"},
      color: {legend: true},
      marks: [
        Plot.barY(classes, Plot.groupX(
          {y: "count"},
          {x: "class", fill: "class", tip: true}
        )),
        Plot.ruleY([0])
      ]
    });
  }
  return Plot.plot({
    title: `Observed ${spec.label} distribution`,
    width,
    height: 300,
    x: {label: `${spec.label} (${spec.unit})`},
    y: {grid: true, label: "Molecules"},
    marks: [
      Plot.rectY(data, Plot.binX(
        {y: "count"},
        {x: "actual", thresholds: 24, tip: true}
      )),
      Plot.ruleY([0])
    ]
  });
}

function regressionFit(data, {width} = {}) {
  const values = data.flatMap((row) => [row.actual, row.prediction]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  return Plot.plot({
    title: "Observed vs. predicted",
    width,
    height: 360,
    grid: true,
    x: {label: `Observed (${spec.unit})`, domain: [min, max]},
    y: {label: `Predicted (${spec.unit})`, domain: [min, max]},
    marks: [
      Plot.line([[min, min], [max, max]], {
        x: (d) => d[0],
        y: (d) => d[1],
        strokeDasharray: "5,5"
      }),
      Plot.dot(data, {
        x: "actual",
        y: "prediction",
        r: 3.5,
        opacity: 0.7,
        tip: true
      })
    ]
  });
}

function residualPlot(data, {width} = {}) {
  const residuals = data.map((row) => ({
    ...row,
    residual: row.prediction - row.actual
  }));
  return Plot.plot({
    title: "Residuals",
    width,
    height: 360,
    grid: true,
    x: {label: `Observed (${spec.unit})`},
    y: {label: "Prediction − observation"},
    marks: [
      Plot.ruleY([0]),
      Plot.dot(residuals, {
        x: "actual",
        y: "residual",
        r: 3.5,
        opacity: 0.7,
        tip: true
      })
    ]
  });
}

function probabilityPlot(data, {width} = {}) {
  const classified = data.map((row) => ({
    ...row,
    class: row.actual === 1 ? "Penetrating" : "Non-penetrating"
  }));
  return Plot.plot({
    title: "Predicted probability by observed class",
    width,
    height: 360,
    x: {label: "Predicted BBB penetration probability", domain: [0, 1]},
    y: {grid: true, label: "Molecules"},
    color: {legend: true},
    marks: [
      Plot.rectY(classified, Plot.binX(
        {y: "count"},
        {x: "prediction", fill: "class", thresholds: 20, tip: true}
      )),
      Plot.ruleX([0.5], {strokeDasharray: "5,5"})
    ]
  });
}

function rocPlot(data, {width} = {}) {
  return Plot.plot({
    title: "Receiver operating characteristic",
    width,
    height: 360,
    grid: true,
    x: {label: "False-positive rate", domain: [0, 1]},
    y: {label: "True-positive rate", domain: [0, 1]},
    marks: [
      Plot.line([[0, 0], [1, 1]], {
        x: (d) => d[0],
        y: (d) => d[1],
        strokeDasharray: "5,5"
      }),
      Plot.line(data, {x: "fpr", y: "tpr", strokeWidth: 3, tip: true}),
      Plot.dot(data, {x: "fpr", y: "tpr", r: 2})
    ]
  });
}
```

<div class="grid grid-cols-2">
  <div class="card">
    ${resize((width) => targetDistribution(rows, {width}))}
  </div>
  <div class="card context-card">
    <div class="eyebrow">How to read this task</div>
    <h2>${spec.label}</h2>
    <p>${taskCopy[selectedTask]}</p>
    <dl>
      <div><dt>Task type</dt><dd>${spec.type}</dd></div>
      <div><dt>Primary metric</dt><dd>${spec.metric}</dd></div>
      <div><dt>Output unit</dt><dd>${spec.unit}</dd></div>
      <div><dt>Predictions</dt><dd>${spec.available ? "loaded" : "not generated"}</dd></div>
    </dl>
  </div>
</div>

<div class="grid grid-cols-2">
  <div class="card">${
    paired.length
      ? resize((width) =>
        spec.type === "classification"
          ? probabilityPlot(paired, {width})
          : regressionFit(paired, {width})
      )
      : "Generate predictions to display the model fit."
  }</div>
  <div class="card">${
    paired.length
      ? resize((width) =>
        spec.type === "classification"
          ? rocPlot(curve, {width})
          : residualPlot(paired, {width})
      )
      : "Generate predictions to display model diagnostics."
  }</div>
</div>

```js
const tableRows = rows
  .map((row) => ({
    smiles: row.smiles,
    actual: row.actual,
    prediction: row.prediction,
    error:
      Number.isFinite(row.prediction)
        ? Math.abs(row.prediction - row.actual)
        : null
  }))
  .sort((a, b) => (b.error ?? -1) - (a.error ?? -1));
const sessionTableRows = [...sessionRows].reverse();
```

<div class="card results-card">
  <h2>Live prediction history</h2>
  <p class="muted">${sessionRows.length ? "Newest CLI and dashboard predictions appear first." : "No predictions in this session."}</p>
  ${Inputs.table(sessionTableRows, {
    columns: ["smiles", "prediction", "raw", "createdAt"],
    header: {
      smiles: "SMILES",
      prediction: spec.type === "classification" ? "Probability" : "Prediction",
      raw: spec.type === "classification" ? "Logit" : "Raw",
      createdAt: "Time"
    },
    format: {
      prediction: (value) => metric(value, 5),
      raw: (value) => metric(value, 5),
      createdAt: (value) => value ? new Date(value * 1000).toLocaleTimeString() : "—"
    },
    rows: 10
  })}
</div>

<div class="card results-card">
  <h2>MoleculeNet benchmark matches</h2>
  <p class="muted">${paired.length ? "Largest absolute errors appear first." : "Submit labeled test molecules to populate benchmark metrics."}</p>
  ${Inputs.table(tableRows, {
    columns: ["smiles", "actual", "prediction", "error"],
    header: {
      smiles: "SMILES",
      actual: "Observed",
      prediction: "Predicted",
      error: "Absolute error"
    },
    format: {
      actual: (value) => metric(value, 4),
      prediction: (value) => metric(value, 4),
      error: (value) => metric(value, 4)
    },
    rows: 12
  })}
</div>

<style>
.dashboard-header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 2rem;
  padding: 1.2rem 0 0.8rem;
}

.dashboard-header h1 {
  margin: 0.2rem 0;
  max-width: none;
  font-size: clamp(2rem, 5vw, 4.2rem);
  letter-spacing: -0.05em;
}

.dashboard-header p {
  margin: 0;
  color: var(--theme-foreground-muted);
  font-size: 1rem;
}

.eyebrow {
  color: var(--theme-foreground-muted);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.task-picker {
  min-width: 220px;
}

.connection-state {
  padding: 0.35rem 0.7rem;
  border: 1px solid var(--theme-foreground-faintest);
  border-radius: 999px;
  color: var(--theme-foreground-muted);
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
}

.connection-state.connected {
  color: var(--theme-foreground-focus);
}

.prediction-card,
.prediction-summary {
  min-height: 180px;
}

.prediction-summary {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.prediction-form label {
  display: grid;
  gap: 0.35rem;
  color: var(--theme-foreground-muted);
  font-size: 0.8rem;
}

.prediction-form textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 0.65rem;
  border: 1px solid var(--theme-foreground-faintest);
  border-radius: 0.35rem;
  background: var(--theme-background);
  color: var(--theme-foreground);
  font-family: var(--monospace);
  resize: vertical;
}

.prediction-form > div {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.65rem;
}

.prediction-form button {
  padding: 0.5rem 0.85rem;
  border: 0;
  border-radius: 0.35rem;
  background: var(--theme-foreground-focus);
  color: var(--theme-background);
  cursor: pointer;
  font-weight: 700;
}

.prediction-form button:disabled {
  cursor: wait;
  opacity: 0.6;
}

.form-status {
  color: var(--theme-foreground-muted);
  font-size: 0.75rem;
}

.metric-card {
  display: flex;
  min-height: 130px;
  flex-direction: column;
}

.metric-card .big {
  margin-top: auto;
  font-variant-numeric: tabular-nums;
}

.metric-note {
  color: var(--theme-foreground-muted);
  font-size: 0.75rem;
}

.empty-state {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  margin: 0.5rem 0;
  padding: 0.9rem 1rem;
  border: 1px dashed var(--theme-foreground-faintest);
  border-radius: 0.5rem;
  background: var(--theme-background-alt);
}

.empty-state span {
  color: var(--theme-foreground-muted);
}

.hidden {
  display: none !important;
}

.context-card {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 1.5rem 2rem;
}

.context-card p {
  color: var(--theme-foreground-muted);
}

.context-card dl {
  margin: 0.5rem 0 0;
}

.context-card dl div {
  display: flex;
  justify-content: space-between;
  gap: 2rem;
  padding: 0.45rem 0;
  border-top: 1px solid var(--theme-foreground-faintest);
}

.context-card dt {
  color: var(--theme-foreground-muted);
}

.context-card dd {
  margin: 0;
  font-weight: 600;
}

.results-card {
  overflow: auto;
}

@media (max-width: 700px) {
  .dashboard-header {
    align-items: stretch;
    flex-direction: column;
    gap: 1rem;
  }

  .empty-state {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>

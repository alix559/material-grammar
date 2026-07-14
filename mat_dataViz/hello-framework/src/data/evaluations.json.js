import {readFile} from "node:fs/promises";
import {existsSync} from "node:fs";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {csvParse} from "d3-dsv";

const dataDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(dataDirectory, "../../../..");
const predictionDirectory =
  process.env.MATERIAL_PREDICTIONS_DIR ||
  resolve(repositoryRoot, "mat_dataViz/predictions");

const tasks = {
  esol: {
    label: "ESOL",
    description: "Aqueous solubility",
    type: "regression",
    metric: "RMSE",
    unit: "log10(mol/L)",
    csv: "esol/test.csv",
    targetColumn: "measured log solubility in mols per litre"
  },
  lipo: {
    label: "Lipophilicity",
    description: "Octanol/water distribution",
    type: "regression",
    metric: "RMSE",
    unit: "logP/logD",
    csv: "lipophilicity/test.csv",
    targetColumn: "y"
  },
  bbbp: {
    label: "BBBP",
    description: "Blood–brain barrier penetration",
    type: "classification",
    metric: "ROC-AUC",
    unit: "probability",
    csv: "bbbp/test.csv",
    targetColumn: "p_np"
  }
};

const moleculeNetRoot = resolve(
  repositoryRoot,
  "vendor/ibm_materials/models/smi_ted/finetune/moleculenet"
);

async function readPredictions(task) {
  const path = resolve(predictionDirectory, `${task}.json`);
  if (!existsSync(path)) return {path, rows: []};

  const value = JSON.parse(await readFile(path, "utf8"));
  if (!Array.isArray(value)) {
    throw new Error(`${path} must contain a JSON array`);
  }
  return {path, rows: value};
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

const output = {
  generatedAt: new Date().toISOString(),
  predictionDirectory,
  tasks: {},
  rows: []
};

for (const [task, spec] of Object.entries(tasks)) {
  const csvPath = resolve(moleculeNetRoot, spec.csv);
  const labels = csvParse(await readFile(csvPath, "utf8"));
  const predictions = await readPredictions(task);
  const bySmiles = new Map();

  for (const row of predictions.rows) {
    if (row.task && row.task !== task) continue;
    const prediction =
      task === "bbbp"
        ? finiteNumber(row.probability ?? row.prediction)
        : finiteNumber(row.prediction ?? row.raw);
    if (typeof row.smiles === "string" && prediction !== null) {
      bySmiles.set(row.smiles, prediction);
    }
  }

  let predicted = 0;
  for (const row of labels) {
    const actual = finiteNumber(row[spec.targetColumn]);
    const prediction = bySmiles.get(row.smiles) ?? null;
    if (prediction !== null) predicted += 1;
    output.rows.push({
      task,
      smiles: row.smiles,
      actual,
      prediction
    });
  }

  output.tasks[task] = {
    ...spec,
    csv: undefined,
    targetColumn: undefined,
    total: labels.length,
    predicted,
    predictionFile: predictions.path,
    available: predicted > 0
  };
}

process.stdout.write(`${JSON.stringify(output)}\n`);

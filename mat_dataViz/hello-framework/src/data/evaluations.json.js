import {readFile} from "node:fs/promises";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {csvParse} from "d3-dsv";

const dataDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(dataDirectory, "../../../..");

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

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

const output = {
  generatedAt: new Date().toISOString(),
  tasks: {},
  rows: []
};

for (const [task, spec] of Object.entries(tasks)) {
  const csvPath = resolve(moleculeNetRoot, spec.csv);
  const labels = csvParse(await readFile(csvPath, "utf8"));

  for (const row of labels) {
    const actual = finiteNumber(row[spec.targetColumn]);
    output.rows.push({
      task,
      smiles: row.smiles,
      actual,
      prediction: null
    });
  }

  output.tasks[task] = {
    ...spec,
    csv: undefined,
    targetColumn: undefined,
    total: labels.length,
    predicted: 0,
    available: false
  };
}

process.stdout.write(`${JSON.stringify(output)}\n`);

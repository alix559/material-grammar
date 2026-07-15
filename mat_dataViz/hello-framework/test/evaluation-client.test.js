import assert from "node:assert/strict";
import test from "node:test";

import {mergeSession} from "../src/components/evaluation-client.js";

test("live session predictions merge into labeled rows", () => {
  const baseline = {
    tasks: {
      esol: {total: 2, predicted: 0, available: false},
      bbbp: {total: 0, predicted: 0, available: false}
    },
    rows: [
      {task: "esol", smiles: "CCO", actual: -0.8, prediction: null},
      {task: "esol", smiles: "C", actual: -0.4, prediction: null}
    ]
  };
  const session = {
    task: "esol",
    updatedAt: 123,
    rows: [
      {task: "esol", smiles: "CCO", prediction: -1.0},
      {task: "esol", smiles: "CCO", prediction: -0.9},
      {task: "esol", smiles: "unlabeled", prediction: -0.2}
    ]
  };

  const evaluation = mergeSession(baseline, session);
  assert.equal(evaluation.rows[0].prediction, -0.9);
  assert.equal(evaluation.rows[1].prediction, null);
  assert.equal(evaluation.tasks.esol.predicted, 1);
  assert.equal(evaluation.tasks.esol.available, true);
  assert.equal(evaluation.session.rows.length, 3);
  assert.equal(baseline.rows[0].prediction, null);
});

test("session for another task does not alter labels", () => {
  const baseline = {
    tasks: {esol: {total: 1, predicted: 0, available: false}},
    rows: [{task: "esol", smiles: "CCO", actual: -0.8, prediction: null}]
  };
  const evaluation = mergeSession(baseline, {
    task: "bbbp",
    rows: [{task: "bbbp", smiles: "CCO", prediction: 0.9}]
  });
  assert.equal(evaluation.rows[0].prediction, null);
});

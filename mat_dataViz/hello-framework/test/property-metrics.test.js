import assert from "node:assert/strict";
import test from "node:test";

import {
  classificationMetrics,
  regressionMetrics,
  rocCurve
} from "../src/components/property-metrics.js";

test("regression metrics match a small known fixture", () => {
  const metrics = regressionMetrics([
    {actual: 1, prediction: 1},
    {actual: 2, prediction: 3},
    {actual: 3, prediction: 2},
    {actual: 4, prediction: null}
  ]);

  assert.equal(metrics.count, 3);
  assert.ok(Math.abs(metrics.rmse - Math.sqrt(2 / 3)) < 1e-12);
  assert.ok(Math.abs(metrics.mae - 2 / 3) < 1e-12);
  assert.equal(metrics.r2, 0);
});

test("classification metrics handle perfect separation and ties", () => {
  const rows = [
    {actual: 1, prediction: 0.9},
    {actual: 1, prediction: 0.8},
    {actual: 0, prediction: 0.2},
    {actual: 0, prediction: 0.1}
  ];
  const metrics = classificationMetrics(rows);

  assert.equal(metrics.count, 4);
  assert.equal(metrics.rocAuc, 1);
  assert.equal(metrics.accuracy, 1);
  assert.deepEqual(rocCurve(rows).at(-1), {
    fpr: 1,
    tpr: 1,
    threshold: 0.1
  });

  assert.equal(
    classificationMetrics([
      {actual: 1, prediction: 0.5},
      {actual: 0, prediction: 0.5}
    ]).rocAuc,
    0.5
  );
});

test("metrics return null values when predictions are absent", () => {
  assert.deepEqual(regressionMetrics([{actual: 1, prediction: null}]), {
    count: 0,
    rmse: null,
    mae: null,
    r2: null
  });
  assert.equal(
    classificationMetrics([{actual: 1, prediction: null}]).rocAuc,
    null
  );
});

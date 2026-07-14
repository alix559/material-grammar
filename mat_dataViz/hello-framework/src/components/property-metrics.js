function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function pairedRows(rows) {
  return rows.filter(
    (row) => isFiniteNumber(row.actual) && isFiniteNumber(row.prediction)
  );
}

export function regressionMetrics(rows) {
  const paired = pairedRows(rows);
  if (paired.length === 0) {
    return {count: 0, rmse: null, mae: null, r2: null};
  }

  const mean =
    paired.reduce((sum, row) => sum + row.actual, 0) / paired.length;
  let squaredError = 0;
  let absoluteError = 0;
  let totalVariance = 0;

  for (const row of paired) {
    const error = row.prediction - row.actual;
    squaredError += error * error;
    absoluteError += Math.abs(error);
    totalVariance += (row.actual - mean) ** 2;
  }

  return {
    count: paired.length,
    rmse: Math.sqrt(squaredError / paired.length),
    mae: absoluteError / paired.length,
    r2: totalVariance > 0 ? 1 - squaredError / totalVariance : null
  };
}

export function rocCurve(rows) {
  const paired = pairedRows(rows)
    .map((row) => ({actual: row.actual === 1 ? 1 : 0, score: row.prediction}))
    .sort((a, b) => b.score - a.score);
  const positives = paired.reduce((sum, row) => sum + row.actual, 0);
  const negatives = paired.length - positives;

  if (positives === 0 || negatives === 0) return [];

  const points = [{fpr: 0, tpr: 0, threshold: Infinity}];
  let truePositives = 0;
  let falsePositives = 0;

  for (let index = 0; index < paired.length; ) {
    const threshold = paired[index].score;
    while (index < paired.length && paired[index].score === threshold) {
      if (paired[index].actual === 1) truePositives += 1;
      else falsePositives += 1;
      index += 1;
    }
    points.push({
      fpr: falsePositives / negatives,
      tpr: truePositives / positives,
      threshold
    });
  }

  return points;
}

export function classificationMetrics(rows, threshold = 0.5) {
  const paired = pairedRows(rows);
  const curve = rocCurve(paired);
  if (paired.length === 0) {
    return {count: 0, rocAuc: null, accuracy: null, positives: 0};
  }

  let correct = 0;
  let positives = 0;
  for (const row of paired) {
    const actual = row.actual === 1 ? 1 : 0;
    positives += actual;
    if ((row.prediction >= threshold ? 1 : 0) === actual) correct += 1;
  }

  let rocAuc = null;
  if (curve.length > 1) {
    rocAuc = 0;
    for (let index = 1; index < curve.length; index += 1) {
      const left = curve[index - 1];
      const right = curve[index];
      rocAuc +=
        (right.fpr - left.fpr) * (right.tpr + left.tpr) * 0.5;
    }
  }

  return {
    count: paired.length,
    rocAuc,
    accuracy: correct / paired.length,
    positives
  };
}

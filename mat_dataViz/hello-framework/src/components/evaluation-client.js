export function mergeSession(baseline, session) {
  const evaluation = structuredClone(baseline);
  const task = session?.task;
  const sessionRows = Array.isArray(session?.rows) ? session.rows : [];
  const latest = new Map();

  for (const row of sessionRows) {
    if (row.task === task && Number.isFinite(Number(row.prediction))) {
      latest.set(row.smiles, Number(row.prediction));
    }
  }

  let predicted = 0;
  for (const row of evaluation.rows) {
    if (row.task !== task) continue;
    row.prediction = latest.get(row.smiles) ?? null;
    if (row.prediction !== null) predicted += 1;
  }

  if (task && evaluation.tasks[task]) {
    evaluation.tasks[task].predicted = predicted;
    evaluation.tasks[task].available = predicted > 0;
  }
  evaluation.session = {
    task,
    updatedAt: session?.updatedAt ?? null,
    rows: sessionRows
  };
  return evaluation;
}

export async function fetchSession(baseUrl = "") {
  const response = await fetch(`${baseUrl}/api/state`);
  if (!response.ok) {
    throw new Error(`controller state request failed (${response.status})`);
  }
  return response.json();
}

export function connectSession({
  baseUrl = "",
  onState,
  onConnection = () => {}
}) {
  const events = new EventSource(`${baseUrl}/api/events`);
  events.addEventListener("open", () => onConnection("connected"));
  events.addEventListener("state", (event) => {
    onState(JSON.parse(event.data));
  });
  events.addEventListener("error", () => onConnection("reconnecting"));
  return () => events.close();
}

export async function submitPredictions(smiles, {replace = false, baseUrl = ""} = {}) {
  const response = await fetch(`${baseUrl}/api/predict`, {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify({smiles, replace})
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`prediction failed (${response.status}): ${body}`);
  }
  return response.json();
}

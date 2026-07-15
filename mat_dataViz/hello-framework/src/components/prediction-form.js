export function predictionForm({onSubmit, initialValue = "CCO"}) {
  const form = document.createElement("form");
  form.className = "prediction-form";

  const label = document.createElement("label");
  label.textContent = "SMILES (one per line)";
  const textarea = document.createElement("textarea");
  textarea.value = initialValue;
  textarea.rows = 3;
  textarea.required = true;
  label.append(textarea);

  const actions = document.createElement("div");
  const button = document.createElement("button");
  button.type = "submit";
  button.textContent = "Predict with MAX";
  const status = document.createElement("span");
  status.className = "form-status";
  actions.append(button, status);
  form.append(label, actions);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const smiles = textarea.value
      .split(/\r?\n/)
      .map((value) => value.trim())
      .filter(Boolean);
    if (!smiles.length) return;
    button.disabled = true;
    status.textContent = "Predicting…";
    try {
      await onSubmit(smiles);
      status.textContent = `${smiles.length} prediction${smiles.length === 1 ? "" : "s"} added`;
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : String(error);
    } finally {
      button.disabled = false;
    }
  });

  return form;
}

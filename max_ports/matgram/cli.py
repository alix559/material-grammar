from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import click
import httpx
import uvicorn

from .app import create_app
from .settings import TASKS, ControllerSettings

DEFAULT_CONTROLLER_URL = "http://127.0.0.1:8080"


def _controller_url(value: str | None) -> str:
    return (value or os.getenv("MATGRAM_URL", DEFAULT_CONTROLLER_URL)).rstrip("/")


def _request(method: str, path: str, *, url: str | None, **kwargs) -> Any:
    try:
        response = httpx.request(
            method,
            f"{_controller_url(url)}{path}",
            timeout=kwargs.pop("timeout", 300.0),
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        detail = error.response.text
        raise click.ClickException(
            f"controller returned {error.response.status_code}: {detail}"
        ) from error
    except httpx.HTTPError as error:
        raise click.ClickException(f"cannot reach controller: {error}") from error


def _csv_smiles(path: Path, column: str) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise click.ClickException(f"{path} has no {column!r} column")
        return [row[column] for row in reader if row.get(column)]


@click.group()
def cli() -> None:
    """Run SMI-TED MAX predictions and the live dashboard."""


@cli.command()
@click.option("--task", type=click.Choice(sorted(TASKS)), required=True)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option("--device", default="cpu", show_default=True)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, type=int, show_default=True)
def dashboard(
    task: str, checkpoint: Path, device: str, host: str, port: int
) -> None:
    """Export CHECKPOINT, start MAX, and serve the live dashboard."""
    settings = ControllerSettings.from_env(
        task=task,
        checkpoint=checkpoint,
        device=device,
        host=host,
        port=port,
    )
    click.echo(f"Task: {task}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Dashboard: http://localhost:{port}")
    app = create_app(settings)
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command()
@click.option("--smiles", multiple=True)
@click.option("--input-csv", type=click.Path(path_type=Path, exists=True))
@click.option("--smiles-col", default="smiles", show_default=True)
@click.option("--replace/--append", default=False, show_default=True)
@click.option("--url", default=None, help="Controller URL.")
def predict(
    smiles: tuple[str, ...],
    input_csv: Path | None,
    smiles_col: str,
    replace: bool,
    url: str | None,
) -> None:
    """Predict one or more SMILES and update the dashboard."""
    values = list(smiles)
    if input_csv:
        values.extend(_csv_smiles(input_csv, smiles_col))
    if not values:
        raise click.UsageError("provide --smiles and/or --input-csv")
    state = _request(
        "POST",
        "/api/predict",
        url=url,
        json={"smiles": values, "replace": replace},
    )
    for row in state["rows"][-len(values) :]:
        click.echo(f"{row['smiles']}\tprediction={row['prediction']:.6f}")


@cli.command()
@click.option("--url", default=None, help="Controller URL.")
def status(url: str | None) -> None:
    """Show controller and MAX Serve status."""
    click.echo(json.dumps(_request("GET", "/api/status", url=url), indent=2))


@cli.command()
@click.option("--url", default=None, help="Controller URL.")
def clear(url: str | None) -> None:
    """Clear the live prediction session."""
    state = _request("DELETE", "/api/state", url=url)
    click.echo("Cleared session.")


@cli.command("export-results")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--url", default=None, help="Controller URL.")
def export_results(output: Path, url: str | None) -> None:
    """Write the current prediction session to JSON."""
    state = _request("GET", "/api/results", url=url)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2) + "\n")
    click.echo(f"Wrote {output}")


if __name__ == "__main__":
    cli()

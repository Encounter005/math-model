from __future__ import annotations

import csv
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

LITERATURE_STATUS = {
    "reconstruction": "TgRN latent adaptation",
    "gated_fusion": "AGFN local-missing extension",
    "missmodal_alignment": "MissModal local-block extension",
    "self_distillation": "UMDF contiguous-block self-distillation",
}


def render_route_report(route_dir: Path, route: str) -> Path:
    """Render a standalone route report from its persisted seed artifacts."""
    if route not in LITERATURE_STATUS:
        raise ValueError(f"Unsupported route: {route}")
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = environment.get_template("report.html.j2")
    source_paths = sorted(path.relative_to(route_dir).as_posix() for path in route_dir.rglob("*") if path.is_file())
    competition = _read_seed_rows(route_dir, "perturbation_metrics.csv", "competition")
    literature_random = _read_seed_rows(route_dir, "perturbation_metrics.csv", "literature_random")
    whole_modality = _read_seed_rows(route_dir, "whole_modality_metrics.csv")
    gating_diagnostics = _read_seed_rows(route_dir, "gating_diagnostics.csv")
    route_losses = _read_history(route_dir)
    summary = _read_csv(route_dir / "summary.csv")
    output = route_dir / "report.html"
    output.write_text(
        template.render(
            route=route,
            literature_status=LITERATURE_STATUS[route],
            source_paths=source_paths,
            summary=summary,
            competition=competition,
            literature_random=literature_random,
            whole_modality=whole_modality,
            gating_diagnostics=gating_diagnostics,
            route_losses=route_losses,
        ),
        encoding="utf-8",
    )
    return output


def _read_seed_rows(route_dir: Path, filename: str, protocol: str | None = None) -> list[dict[str, str]]:
    rows = []
    for path in sorted(route_dir.glob(f"seed_*/{filename}")):
        for row in _read_csv(path):
            if protocol is None or row.get("protocol") == protocol:
                rows.append({"seed": path.parent.name.removeprefix("seed_"), **row})
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_history(route_dir: Path) -> list[dict[str, str | float | int]]:
    rows = []
    for path in sorted(route_dir.glob("seed_*/train_history.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append({"seed": path.parent.name.removeprefix("seed_"), **json.loads(line)})
    return rows

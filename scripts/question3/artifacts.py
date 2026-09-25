from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_attachment4_outputs(
    directory: Path, predictions: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "attachment4_predictions.csv", predictions)
    _write_csv(directory / "attachment4_evidence.csv", evidence)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        if fields:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

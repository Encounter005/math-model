from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_attachment4_outputs(
    directory: Path, predictions: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "attachment4_predictions.csv", predictions)
    _write_csv(directory / "attachment4_evidence.csv", evidence)


def write_seed_predictions(directory: Path, seed: int, rows: list[dict[str, Any]]) -> None:
    path = directory / "seed_predictions" / f"seed_{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, rows)


def write_explanation_cards(directory: Path, cards: dict[str, dict[str, Any]]) -> None:
    path = directory / "explanation_cards"
    path.mkdir(parents=True, exist_ok=True)
    for identifier, card in cards.items():
        (path / f"{identifier}.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        if fields:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def validate_output(directory: Path) -> None:
    path = directory / "attachment4_predictions.csv"
    if not path.is_file():
        raise ValueError("Expected 20 Attachment 4 IDs")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    ids = {row.get("id") for row in rows}
    expected = {f"{index:02d}" for index in range(1, 21)}
    if ids != expected or len(rows) != 20:
        raise ValueError("Expected 20 Attachment 4 IDs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Question 3 Attachment 4 outputs."
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_output(
        Path(__file__).resolve().parents[2] / config["paths"]["artifact_root"]
    )


if __name__ == "__main__":
    main()

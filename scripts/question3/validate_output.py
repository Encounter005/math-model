from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def validate_output(directory: Path, seeds: list[int] | None = None) -> None:
    path = directory / "attachment4_predictions.csv"
    if not path.is_file():
        raise ValueError("Expected 20 Attachment 4 IDs")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    ids = {row.get("id") for row in rows}
    expected = {f"{index:02d}" for index in range(1, 21)}
    if ids != expected or len(rows) != 20:
        raise ValueError("Expected 20 Attachment 4 IDs")
    for seed in seeds or []:
        audit = directory / "seed_predictions" / f"seed_{seed}.csv"
        if not audit.is_file():
            raise ValueError(f"Missing seed audit: {audit}")
        with audit.open(newline="", encoding="utf-8") as file:
            audit_ids = {row.get("id") for row in csv.DictReader(file)}
        if audit_ids != expected:
            raise ValueError(f"Invalid seed audit: {audit}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Question 3 Attachment 4 outputs."
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Inference seed subset being validated.")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_output(
        Path(__file__).resolve().parents[2] / config["paths"]["artifact_root"],
        args.seeds or config["training"]["seeds"],
    )


if __name__ == "__main__":
    main()

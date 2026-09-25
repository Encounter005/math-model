from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import yaml

from scripts.question3.artifacts import write_attachment4_outputs


def ensemble_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("At least one seed prediction is required")
    probabilities = [
        statistics.fmean(row["probabilities"][index] for row in predictions)
        for index in range(3)
    ]
    return {
        "probabilities": probabilities,
        "class_prediction": max(range(3), key=probabilities.__getitem__),
        "regression_prediction": statistics.fmean(
            row["regression_prediction"] for row in predictions
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Question 3 Attachment 4 inference."
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    root = Path(__file__).resolve().parents[2]
    paths = {key: root / value for key, value in config["paths"].items()}
    missing = [
        str(paths["artifact_root"] / "seeds" / f"seed_{seed}" / "checkpoint_best.pt")
        for seed in config["training"]["seeds"]
        if not (
            paths["artifact_root"] / "seeds" / f"seed_{seed}" / "checkpoint_best.pt"
        ).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing selected checkpoints: " + ", ".join(missing))
    write_attachment4_outputs(paths["artifact_root"], [], [])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.question3.data import Question3Dataset, collate_question3_samples
from scripts.question3.models import FrozenTextEncoder, GatedSelfDistillationModel
from scripts.question3.training import fix_seed, train, write_seed_artifacts

PATH_KEYS = (
    "training_data",
    "attachment4_dir",
    "video_dir",
    "artifact_root",
    "model_cache",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Question 3 seed.")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise TypeError(f"Configuration must contain paths: {path}")
    root = Path(__file__).resolve().parents[2]
    for key in PATH_KEYS:
        config["paths"][key] = str((root / config["paths"][key]).resolve())
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        for key in PATH_KEYS:
            print(f"{key}: {config['paths'][key]}")
        return
    run_seed(config, args.seed, args.resume)


def run_seed(config: dict[str, Any], seed: int, resume: bool = False) -> None:
    fix_seed(seed)
    _validate_inputs(config)
    datasets = _load_datasets(Path(config["paths"]["training_data"]))
    directory = Path(config["paths"]["artifact_root"]) / "seeds" / f"seed_{seed}"
    checkpoint = directory / "checkpoint_last.pt"
    if resume:
        _validate_checkpoint(checkpoint)
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_question3_samples,
    )
    valid_loader = DataLoader(
        datasets["valid"], batch_size=batch_size, collate_fn=collate_question3_samples
    )
    sample = datasets["train"][0]
    model = GatedSelfDistillationModel(
        FrozenTextEncoder(Path(config["paths"]["model_cache"])),
        audio_dim=sample["audio"].shape[-1],
        vision_dim=sample["vision"].shape[-1],
        hidden_dim=int(config["model"]["hidden_dim"]),
        temporal_dropout=float(config["model"]["temporal_dropout"]),
        representation_coefficient=float(config["loss"]["representation_coefficient"]),
        classification_coefficient=float(config["loss"]["classification_coefficient"]),
        regression_coefficient=float(config["loss"]["regression_coefficient"]),
        temperature=float(config["loss"]["temperature"]),
    )
    if resume:
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
    result = train(
        model,
        train_loader,
        valid_loader,
        config,
        seed,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    write_seed_artifacts(directory, config, result)


def _load_datasets(path: Path) -> dict[str, Question3Dataset]:
    with path.open("rb") as file:
        payload = pickle.load(file)
    datasets = {}
    labels = {"Negative": 0, "Neutral": 1, "Positive": 2}
    for split in ("train", "valid"):
        source = payload[split]
        datasets[split] = Question3Dataset(
            [
                {
                    "id": str(source.get("id", range(len(source["text_bert"])))[index]),
                    "raw_text": source.get("raw_text", [""] * len(source["text_bert"]))[
                        index
                    ],
                    "text_bert": source["text_bert"][index],
                    "audio": source["audio"][index],
                    "vision": source["vision"][index],
                    "class_label": torch.tensor(
                        labels.get(
                            source["classification_labels"][index],
                            source["classification_labels"][index],
                        ),
                        dtype=torch.long,
                    ),
                    "regression_label": torch.tensor(
                        source["regression_labels"][index], dtype=torch.float32
                    ),
                }
                for index in range(len(source["text_bert"]))
            ]
        )
    return datasets


def _validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint unavailable: {path}")
    if not isinstance(torch.load(path, map_location="cpu", weights_only=True), dict):
        raise TypeError(f"Invalid resume checkpoint: {path}")


def _validate_inputs(config: dict[str, Any]) -> None:
    training_data = Path(config["paths"]["training_data"])
    snapshots = Path(config["paths"]["model_cache"]) / "snapshots"
    if not training_data.is_file():
        raise FileNotFoundError(f"Training data unavailable: {training_data}")
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        raise FileNotFoundError(f"Cached DistilBERT snapshot unavailable: {snapshots}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.question2.artifacts import (
    append_progress,
    write_competition_summary,
    write_route_summary,
    write_seed_artifacts,
)
from scripts.question2.data import (
    collate_samples,
    load_inference_dataset,
    load_training_datasets,
)
from scripts.question2.models import (
    FrozenTextEncoder,
    GatedFusionModel,
    MissModalAlignmentModel,
    ReconstructionModel,
    SelfDistillationModel,
)
from scripts.question2.report import render_route_report
from scripts.question2.training import fix_seed, predict, train_and_evaluate

ROUTES = ("reconstruction", "gated_fusion", "missmodal_alignment", "self_distillation")
PATH_KEYS = ("training_data", "inference_dir", "artifact_root", "model_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Question 2 experiment route.")
    parser.add_argument("--route", required=True, choices=ROUTES)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")

    project_root = path.resolve().parents[2]
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise TypeError("Configuration must contain a paths mapping")
    for key in PATH_KEYS:
        value = paths.get(key)
        if not isinstance(value, str):
            raise TypeError(f"Configuration paths.{key} must be a string")
        paths[key] = str((project_root / value).resolve())
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(f"route: {args.route}")
        for key in PATH_KEYS:
            print(f"{key}: {config['paths'][key]}")
        return
    run_route(args.route, config, args.seed, args.resume)


def run_route(route: str, config: dict[str, Any], seed: int | None = None, resume: bool = False) -> Path:
    """Run configured seeds serially, then publish their aggregate report."""
    _validate_inputs(config)
    seeds = [seed] if seed is not None else config["training"]["seeds"]
    route_dir = Path(config["paths"]["artifact_root"]) / route
    summaries = []
    for current_seed in seeds:
        if resume:
            _validate_resume_checkpoint(route_dir / f"seed_{current_seed}" / "checkpoint_last.pt")
        summaries.append(run_seed(route, config, current_seed, resume))
    write_route_summary(route_dir, summaries)
    if all((route_dir / f"seed_{current_seed}" / "perturbation_metrics.csv").is_file() for current_seed in seeds):
        write_competition_summary(route_dir, seeds)
    return render_route_report(route_dir, route)


def run_seed(route: str, config: dict[str, Any], seed: int, resume: bool) -> dict[str, float | int | None]:
    """Run one labelled attachment-2 seed and final unlabelled attachment-3 inference."""
    fix_seed(seed)
    datasets = load_training_datasets(Path(config["paths"]["training_data"]))
    inference_dataset = load_inference_dataset(Path(config["paths"]["inference_dir"]))
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_samples)
    validation_loader = DataLoader(datasets["valid"], batch_size=batch_size, collate_fn=collate_samples)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, collate_fn=collate_samples)
    model = _build_model(route, config, datasets["train"][0])
    route_dir = Path(config["paths"]["artifact_root"]) / route
    progress_path = route_dir / f"seed_{seed}" / "train_progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")
    if resume:
        checkpoint = torch.load(route_dir / f"seed_{seed}" / "checkpoint_last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_and_evaluate(
        model,
        route,
        train_loader,
        validation_loader,
        config,
        seed,
        device,
        lambda record: _report_progress(progress_path, record),
    )
    test_predictions = predict(model, route, inference_loader, device)
    write_seed_artifacts(
        route_dir,
        seed,
        config,
        {
            "training_source": config["paths"]["training_data"],
            "train_count": len(datasets["train"]),
            "validation_count": len(datasets["valid"]),
            "inference_source": config["paths"]["inference_dir"],
            "inference_count": len(inference_dataset),
            "inference_usage": "unlabelled final inference only",
        },
        result,
        test_predictions,
    )
    return {"seed": seed, "best_epoch": result["best_epoch"], **result["best_metrics"]}


def _build_model(route: str, config: dict[str, Any], sample: dict[str, Any]) -> torch.nn.Module:
    text_encoder = FrozenTextEncoder(Path(config["paths"]["model_cache"]))
    dimensions = {"audio_dim": sample["audio"].shape[-1], "vision_dim": sample["vision"].shape[-1]}
    hidden_dim = int(config["model"]["hidden_dim"])
    if route == "reconstruction":
        return ReconstructionModel(text_encoder, hidden_dim=hidden_dim, **dimensions)
    if route == "gated_fusion":
        return GatedFusionModel(text_encoder, hidden_dim=hidden_dim, entropy_temperature=config["model"]["entropy_temperature"], **dimensions)
    if route == "missmodal_alignment":
        loss = config["loss"]
        return MissModalAlignmentModel(
            text_encoder,
            hidden_dim=hidden_dim,
            semantic_coefficient=loss["missmodal_semantic_coefficient"],
            distance_coefficient=loss["missmodal_distance_coefficient"],
            geometry_coefficient=loss["missmodal_geometry_coefficient"],
            temperature=loss["missmodal_temperature"],
            **dimensions,
        )
    if route == "self_distillation":
        loss = config["loss"]
        return SelfDistillationModel(
            text_encoder,
            hidden_dim=hidden_dim,
            mmd_coefficient=loss["self_distillation_mmd_coefficient"],
            classification_coefficient=loss["self_distillation_classification_coefficient"],
            regression_coefficient=loss["self_distillation_regression_coefficient"],
            temperature=loss["self_distillation_temperature"],
            mmd_bandwidth=loss["self_distillation_mmd_bandwidth"],
            **dimensions,
        )
    raise ValueError(f"Unsupported route: {route}")


def _validate_inputs(config: dict[str, Any]) -> None:
    paths = config["paths"]
    if not Path(paths["training_data"]).is_file():
        raise FileNotFoundError(f"Training data unavailable: {paths['training_data']}")
    inference_dir = Path(paths["inference_dir"])
    if not any(inference_dir.glob("*.pkl")):
        raise FileNotFoundError(f"Inference samples unavailable: {inference_dir}")
    snapshots = Path(paths["model_cache"]) / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        raise FileNotFoundError(f"Cached DistilBERT snapshot unavailable: {snapshots}")


def _validate_resume_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint unavailable: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid resume checkpoint: {path}")


def _report_progress(path: Path, record: dict[str, float | int]) -> None:
    append_progress(path, record)
    eta = int(record["eta_seconds"])
    print(
        f"epoch {record['epoch']}/{record['epochs']} batch {record['batch']}/{record['batches']} "
        f"progress {record['progress']:.1%} eta {eta // 60:02d}:{eta % 60:02d} loss {record['loss']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

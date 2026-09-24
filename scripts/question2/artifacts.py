from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_seed_artifacts(
    route_dir: Path,
    seed: int,
    config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    test_predictions: Iterable[Mapping[str, Any]],
    environment: Mapping[str, Any] | None = None,
) -> Path:
    """Persist derived outputs for one seed without copying source samples."""
    seed_dir = route_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    _write_json(seed_dir / "config.json", config)
    _write_json(seed_dir / "environment.json", environment or environment_manifest())
    _write_json(seed_dir / "dataset_manifest.json", dataset_manifest)
    _write_jsonl(seed_dir / "train_history.jsonl", result["history"])
    if "progress_history" in result:
        _write_jsonl(seed_dir / "train_progress.jsonl", result["progress_history"])
    _write_csv(seed_dir / "validation_predictions.csv", result["validation"]["predictions"])
    perturbations = [
        {"protocol": "competition", **row} for row in result["competition_metrics"]
    ] + [{"protocol": "literature_random", **row} for row in result["literature_random_metrics"]]
    _write_csv(seed_dir / "perturbation_metrics.csv", perturbations)
    _write_csv(seed_dir / "whole_modality_metrics.csv", result["whole_modality_metrics"])
    _write_masks(seed_dir / "masks_validation.npz", result["validation_masks"])
    _write_csv(seed_dir / "test_predictions.csv", test_predictions)
    torch.save(result["checkpoint_best"], seed_dir / "checkpoint_best.pt")
    torch.save(result["checkpoint_last"], seed_dir / "checkpoint_last.pt")
    if "reconstruction_metrics" in result or route_dir.name == "reconstruction":
        reconstruction_metrics = result.get("reconstruction_metrics")
        if reconstruction_metrics is None:
            reconstruction_metrics = [
                {"id": row["id"], "reconstruction_errors": row["reconstruction_errors"]}
                for row in result["validation"]["predictions"]
                if "reconstruction_errors" in row
            ]
        _write_csv(seed_dir / "reconstruction_metrics.csv", reconstruction_metrics)
    return seed_dir


def write_route_summary(route_dir: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write the aggregate seed-level result table for one route."""
    route_dir.mkdir(parents=True, exist_ok=True)
    path = route_dir / "summary.csv"
    _write_csv(path, rows)
    return path


def environment_manifest() -> dict[str, Any]:
    """Record executable and accelerator versions needed to reproduce a run."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def append_progress(path: Path, record: Mapping[str, Any]) -> None:
    """Append one completed training iteration so progress survives interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        if not fields:
            return
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in materialized])


def _write_masks(path: Path, masks: Mapping[str, Mapping[str, Any]]) -> None:
    flattened = {
        f"{condition}__{modality}": np.asarray(mask)
        for condition, modalities in masks.items()
        for modality, mask in modalities.items()
    }
    np.savez_compressed(path, **flattened)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_json_default)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")

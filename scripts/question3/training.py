from __future__ import annotations

import copy
import json
import random
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from scripts.question2.metrics import compute_metrics
from scripts.question3.augmentation import make_augmented_view


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_best_epoch(history: list[Mapping[str, float | int]]) -> int:
    if not history:
        raise ValueError("Training history must not be empty")
    return int(
        max(history, key=lambda row: (float(row["weighted_f1"]), -float(row["mae"])))[
            "epoch"
        ]
    )


def train(
    model: torch.nn.Module,
    train_loader: Iterable[dict[str, Any]],
    valid_loader: Iterable[dict[str, Any]],
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    class_weights = _class_weights(train_loader, device)
    history: list[dict[str, float | int | None]] = []
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses = []
        for batch_index, raw_batch in enumerate(train_loader):
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad()
            full = model(batch)
            first = model(
                make_augmented_view(
                    batch,
                    seed + epoch * 10_000 + 2 * batch_index,
                    tuple(config["augmentation"]["rates"]),
                    float(config["augmentation"]["feature_dropout"]),
                )
            )
            second = model(
                make_augmented_view(
                    batch,
                    seed + epoch * 10_000 + 2 * batch_index + 1,
                    tuple(config["augmentation"]["rates"]),
                    float(config["augmentation"]["feature_dropout"]),
                )
            )
            progress = (epoch - 1) / max(int(config["loss"]["warmup_epochs"]) - 1, 1)
            terms = model.losses(
                full,
                first,
                second,
                batch["class_label"],
                batch["regression_label"],
                class_weights,
                progress,
            )
            terms["loss"].backward()
            optimizer.step()
            losses.append(float(terms["loss"].detach()))
        validation = evaluate(model, valid_loader, device)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **validation["metrics"],
        }
        history.append(record)
        if select_best_epoch(history) == epoch:
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    validation = evaluate(model, valid_loader, device)
    return {
        "history": history,
        "validation": validation,
        "checkpoint_best": best_state,
        "checkpoint_last": last_state,
        "best_epoch": select_best_epoch(history),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    model.eval()
    rows = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        outputs = model(batch)
        probabilities = torch.softmax(outputs["classification_logits"], dim=-1)
        for index, identifier in enumerate(batch["id"]):
            rows.append(
                {
                    "id": identifier,
                    "class_label": int(batch["class_label"][index]),
                    "class_prediction": int(probabilities[index].argmax()),
                    "regression_label": float(batch["regression_label"][index]),
                    "regression_prediction": float(
                        outputs["regression_prediction"][index]
                    ),
                }
            )
    return {
        "metrics": compute_metrics(
            [row["class_label"] for row in rows],
            [row["class_prediction"] for row in rows],
            [row["regression_label"] for row in rows],
            [row["regression_prediction"] for row in rows],
        ),
        "predictions": rows,
        "prediction_count": len(rows),
    }


def write_seed_artifacts(
    directory: str | Any, config: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "history.json").write_text(
        json.dumps(result["history"]) + "\n", encoding="utf-8"
    )
    (directory / "validation_predictions.json").write_text(
        json.dumps(result["validation"]["predictions"]) + "\n", encoding="utf-8"
    )
    torch.save(result["checkpoint_best"], directory / "checkpoint_best.pt")
    torch.save(result["checkpoint_last"], directory / "checkpoint_last.pt")


def _class_weights(
    loader: Iterable[dict[str, Any]], device: torch.device
) -> torch.Tensor:
    labels = torch.cat([batch["class_label"] for batch in loader])
    counts = torch.bincount(labels, minlength=3).float()
    return (counts.sum() / (len(counts) * counts.clamp_min(1))).to(device)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

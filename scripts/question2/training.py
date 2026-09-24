from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.question2.masking import (
    apply_contiguous_mask,
    apply_random_mask,
    apply_whole_modality_mask,
)
from scripts.question2.metrics import compute_metrics

MODALITIES = ("text", "audio", "vision")
WHOLE_MODALITY_SUBSETS = ("T", "A", "V", "TA", "TV", "AV")


def fix_seed(seed: int) -> None:
    """Make Python, NumPy, PyTorch, and CUDA stochasticity reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_best_epoch(history: list[Mapping[str, float | int]]) -> int:
    """Select maximum weighted-F1, resolving ties by minimum MAE."""
    if not history:
        raise ValueError("Training history must not be empty")
    return int(max(history, key=lambda row: (float(row["weighted_f1"]), -float(row["mae"])))["epoch"])


def train_and_evaluate(
    model: nn.Module,
    route: str,
    train_loader: Iterable[dict[str, Any]],
    validation_loader: Iterable[dict[str, Any]],
    config: Mapping[str, Any],
    seed: int,
    device: torch.device | str = "cpu",
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> dict[str, Any]:
    """Train one route on attachment 2 and evaluate controlled validation masks."""
    if route not in {"reconstruction", "gated_fusion", "missmodal_alignment"}:
        raise ValueError(f"Unsupported route: {route}")
    fix_seed(seed)
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    class_weights = _class_weights(train_loader, device)
    history: list[dict[str, float | int]] = []
    progress_history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | None] | None = None
    batches = len(train_loader)
    total_steps = batches * int(config["training"]["epochs"])
    started = time.monotonic()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[dict[str, float]] = []
        for batch_index, raw_batch in enumerate(train_loader):
            batch = _to_device(raw_batch, device)
            masks = _training_masks(batch, seed + epoch * 10_000 + batch_index)
            optimizer.zero_grad()
            loss_terms = _loss_terms(model, route, batch, masks, class_weights, config)
            loss_terms["loss"].backward()
            optimizer.step()
            scalar_losses = {name: float(value.detach()) for name, value in loss_terms.items()}
            losses.append(scalar_losses)
            progress = make_progress_record(
                epoch,
                int(config["training"]["epochs"]),
                batch_index + 1,
                batches,
                (epoch - 1) * batches + batch_index + 1,
                total_steps,
                time.monotonic() - started,
                scalar_losses,
            )
            progress_history.append(progress)
            if progress_callback:
                progress_callback(progress)
        validation = evaluate(model, route, validation_loader, class_weights, device)
        record: dict[str, float | int] = {"epoch": epoch, **_mean_losses(losses), **validation["metrics"]}
        history.append(record)
        if select_best_epoch(history) == epoch:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation["metrics"]
    if best_state is None or best_metrics is None:
        raise RuntimeError("Training produced no checkpoint")
    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    validation = evaluate(model, route, validation_loader, class_weights, device)
    perturbations = evaluate_perturbations(model, route, validation_loader, class_weights, config, seed, device)
    return {
        "history": history,
        "progress_history": progress_history,
        "best_epoch": select_best_epoch(history),
        "best_metrics": best_metrics,
        "checkpoint_best": best_state,
        "checkpoint_last": last_state,
        "validation": validation,
        **perturbations,
    }


def make_progress_record(
    epoch: int,
    epochs: int,
    batch: int,
    batches: int,
    completed_steps: int,
    total_steps: int,
    elapsed_seconds: float,
    losses: Mapping[str, float],
) -> dict[str, float | int]:
    """Estimate remaining training time from elapsed batch throughput."""
    eta_seconds = elapsed_seconds * (total_steps - completed_steps) / completed_steps if completed_steps else 0.0
    return {
        "epoch": epoch,
        "epochs": epochs,
        "batch": batch,
        "batches": batches,
        "progress": completed_steps / total_steps if total_steps else 1.0,
        "eta_seconds": eta_seconds,
        **losses,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    route: str,
    loader: Iterable[dict[str, Any]],
    class_weights: torch.Tensor,
    device: torch.device,
    mask_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one complete or synthetically masked validation view."""
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[dict[str, float]] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        active_mask = _mask_for_batch(mask_result, batch) if mask_result else _empty_masks(batch)
        terms = _loss_terms(model, route, batch, active_mask, class_weights, {})
        outputs = _outputs(model, route, batch, active_mask)
        losses.append({name: float(value) for name, value in terms.items()})
        rows.extend(_prediction_rows(batch, outputs, active_mask))
    metrics = compute_metrics(
        [row["class_label"] for row in rows],
        [row["class_prediction"] for row in rows],
        [row["regression_label"] for row in rows],
        [row["regression_prediction"] for row in rows],
    )
    return {"metrics": metrics, "predictions": rows, "losses": _mean_losses(losses)}


def evaluate_perturbations(
    model: nn.Module,
    route: str,
    loader: Iterable[dict[str, Any]],
    class_weights: torch.Tensor,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Keep competition, random-position, and whole-modality results separate."""
    batches = [_to_device(batch, device) for batch in loader]
    competition: list[dict[str, Any]] = []
    literature_random: list[dict[str, Any]] = []
    whole_modality: list[dict[str, Any]] = []
    masks: dict[str, dict[str, torch.Tensor]] = {}
    index = 0
    for modalities in _modality_subsets(include_all=True):
        for rate in config["masking"]["competition_rates"]:
            for position in config["masking"]["positions"]:
                result = _evaluate_masked_batches(
                    model,
                    route,
                    batches,
                    class_weights,
                    device,
                    lambda batch, modalities=modalities, rate=rate, position=position, condition_seed=seed + index: (
                        apply_contiguous_mask(batch, modalities, rate, position, condition_seed)
                    ),
                )
                key = f"competition-{''.join(modalities)}-{rate}-{position}"
                competition.append({"modalities": "".join(modalities), "rate": rate, "position": position, **result["metrics"]})
                masks[key] = result["masks"]
                index += 1
    for rate in config["masking"]["literature_rates"]:
        result = _evaluate_masked_batches(
            model,
            route,
            batches,
            class_weights,
            device,
            lambda batch, rate=rate, condition_seed=seed + index: apply_random_mask(
                batch, MODALITIES, rate, condition_seed
            ),
        )
        key = f"literature-random-{rate}"
        literature_random.append({"rate": rate, **result["metrics"]})
        masks[key] = result["masks"]
        index += 1
    for subset in WHOLE_MODALITY_SUBSETS:
        result = _evaluate_masked_batches(
            model,
            route,
            batches,
            class_weights,
            device,
            lambda batch, subset=subset: apply_whole_modality_mask(batch, subset),
        )
        key = f"literature-whole-{subset}"
        whole_modality.append({"subset": subset, **result["metrics"]})
        masks[key] = result["masks"]
    return {
        "competition_metrics": competition,
        "literature_random_metrics": literature_random,
        "whole_modality_metrics": whole_modality,
        "validation_masks": masks,
    }


@torch.no_grad()
def predict(model: nn.Module, route: str, loader: Iterable[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    """Generate attachment-3 predictions without labels, fitting, or selection."""
    model.eval()
    rows = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        outputs = _outputs(model, route, batch, _empty_masks(batch))
        probabilities = torch.softmax(outputs["classification_logits"], dim=-1)
        for index, identifier in enumerate(batch["id"]):
            row = {
                "id": identifier,
                "class_prediction": int(probabilities[index].argmax()),
                "probabilities": probabilities[index].cpu().tolist(),
                "regression_prediction": float(outputs["regression_prediction"][index]),
            }
            for name in ("reliability_gates", "importance_gates", "fused_modality_weights"):
                if name in outputs:
                    row[name] = outputs[name][index].cpu().tolist()
            if "mixture_coefficient" in outputs:
                row["mixture_coefficient"] = float(outputs["mixture_coefficient"][index])
            rows.append(row)
    return rows


@torch.no_grad()
def _evaluate_masked_batches(
    model: nn.Module,
    route: str,
    batches: list[dict[str, Any]],
    class_weights: torch.Tensor,
    device: torch.device,
    make_mask: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    collected_masks = {name: [] for name in MODALITIES}
    for batch in batches:
        result = make_mask(batch)
        artificial = result["artificial_masks"]
        outputs = _outputs(model, route, batch, artificial)
        rows.extend(_prediction_rows(batch, outputs, artificial))
        for name in MODALITIES:
            collected_masks[name].append(artificial[name].cpu())
    return {
        "metrics": compute_metrics(
            [row["class_label"] for row in rows], [row["class_prediction"] for row in rows],
            [row["regression_label"] for row in rows], [row["regression_prediction"] for row in rows],
        ),
        "masks": {name: torch.cat(values).numpy() for name, values in collected_masks.items()},
    }


def _outputs(model: nn.Module, route: str, batch: dict[str, Any], masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if route == "reconstruction":
        return model(batch, masks)
    if route == "gated_fusion":
        return model(_observed_batch(batch, masks))
    outputs = model(batch, [masks])
    return outputs["incomplete"][0] if any(mask.any() for mask in masks.values()) else outputs["complete"]


def _loss_terms(
    model: nn.Module,
    route: str,
    batch: dict[str, Any],
    masks: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if route == "missmodal_alignment":
        outputs = model(batch, [masks])
        return model.alignment_losses(outputs, batch["class_label"], batch["regression_label"], class_weights)
    outputs = _outputs(model, route, batch, masks)
    terms = model.supervised_loss(outputs, batch["class_label"], batch["regression_label"], class_weights)
    if route == "reconstruction":
        reconstruction = outputs["reconstruction_loss"]
        terms["reconstruction_loss"] = reconstruction
        terms["loss"] = terms["loss"] + float(config.get("loss", {}).get("reconstruction_coefficient", 1.0)) * reconstruction
    elif route == "gated_fusion":
        vat = model.vat_loss(outputs["fused_representation"]) if torch.is_grad_enabled() else terms["loss"] * 0
        terms["vat_loss"] = vat
        terms["loss"] = terms["loss"] + float(config.get("loss", {}).get("vat_coefficient", 0.1)) * vat
    return terms


def _training_masks(batch: dict[str, Any], seed: int) -> dict[str, torch.Tensor]:
    return apply_contiguous_mask(batch, MODALITIES, 0.3, "middle", seed)["artificial_masks"]


def _observed_batch(batch: dict[str, Any], masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    observed = dict(batch)
    for name in MODALITIES:
        observed[f"{name}_mask"] = batch[f"{name}_mask"] & ~masks[name]
    for name in ("audio", "vision"):
        observed[name] = batch[name].masked_fill(~observed[f"{name}_mask"].unsqueeze(-1), 0)
    return observed


def _empty_masks(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(batch[f"{name}_mask"], dtype=torch.bool) for name in MODALITIES}


def _mask_for_batch(mask_result: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    del mask_result
    return _empty_masks(batch)


def _prediction_rows(batch: dict[str, Any], outputs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    probabilities = torch.softmax(outputs["classification_logits"], dim=-1)
    rows = []
    for index, identifier in enumerate(batch["id"]):
        rows.append({
            "id": identifier,
            "class_label": int(batch["class_label"][index]),
            "class_prediction": int(probabilities[index].argmax()),
            "probabilities": probabilities[index].detach().cpu().tolist(),
            "regression_label": float(batch["regression_label"][index]),
            "regression_prediction": float(outputs["regression_prediction"][index]),
            "artificial_masks": {name: masks[name][index].detach().cpu().tolist() for name in MODALITIES},
            "original_masks": {name: batch[f"{name}_mask"][index].detach().cpu().tolist() for name in MODALITIES},
            "observed_fractions": {
                name: float((batch[f"{name}_mask"][index] & ~masks[name][index]).float().mean()) for name in MODALITIES
            },
        })
        for name in (
            "reliability_gates",
            "importance_gates",
            "fused_modality_weights",
            "reconstruction_errors",
        ):
            if name in outputs:
                rows[-1][name] = outputs[name][index].detach().cpu().tolist()
        if "mixture_coefficient" in outputs:
            rows[-1]["mixture_coefficient"] = float(outputs["mixture_coefficient"][index])
    return rows


def _class_weights(loader: Iterable[dict[str, Any]], device: torch.device) -> torch.Tensor:
    labels = torch.cat([batch["class_label"] for batch in loader])
    counts = torch.bincount(labels, minlength=3).float()
    return (counts.sum() / (len(counts) * counts.clamp_min(1))).to(device)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def _mean_losses(losses: list[Mapping[str, float]]) -> dict[str, float]:
    if not losses:
        raise ValueError("No batches were available")
    return {name: float(np.mean([loss[name] for loss in losses if name in loss])) for name in set().union(*losses)}


def _modality_subsets(include_all: bool) -> tuple[tuple[str, ...], ...]:
    subsets = (("text",), ("audio",), ("vision",), ("text", "audio"), ("text", "vision"), ("audio", "vision"))
    return (*subsets, MODALITIES) if include_all else subsets

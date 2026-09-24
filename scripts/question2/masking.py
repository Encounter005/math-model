from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import torch

MODALITIES = ("text", "audio", "vision")
COMPETITION_SUBSETS = (
    ("T", ("text",)),
    ("A", ("audio",)),
    ("V", ("vision",)),
    ("TA", ("text", "audio")),
    ("TV", ("text", "vision")),
    ("AV", ("audio", "vision")),
    ("TAV", MODALITIES),
)


def apply_contiguous_mask(
    masks: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
    rate: float,
    position: str,
    seed: int,
) -> dict[str, Any]:
    """Hide one contiguous valid interval per selected modality and sample."""
    _validate(rate, modalities)
    if position not in {"front", "middle", "back"}:
        raise ValueError(f"Unsupported position: {position}")
    artificial = _empty_masks(masks)
    for modality in modalities:
        for index, mask in enumerate(_original_masks(masks)[modality]):
            valid = torch.where(mask)[0]
            count = _mask_count(len(valid), rate)
            if not count:
                continue
            start = {"front": 0, "middle": (len(valid) - count) // 2, "back": len(valid) - count}[position]
            artificial[modality][index, valid[start : start + count]] = True
    return _result(masks, artificial, "competition", {"rate": rate, "position": position, "seed": seed})


def sample_competition_mask(
    masks: dict[str, torch.Tensor], rates: Sequence[float], positions: Sequence[str], seed: int
) -> dict[str, Any]:
    """Sample one reproducible competition-protocol condition."""
    if not rates or not positions:
        raise ValueError("Competition rates and positions must not be empty")
    generator = random.Random(seed)
    modalities, selected = generator.choice(COMPETITION_SUBSETS)
    rate = generator.choice(rates)
    position = generator.choice(positions)
    result = apply_contiguous_mask(masks, selected, rate, position, seed)
    result["condition"]["modalities"] = modalities
    return result


def apply_random_mask(
    masks: dict[str, torch.Tensor], modalities: tuple[str, ...], rate: float, seed: int
) -> dict[str, Any]:
    """Hide independently selected valid positions for literature comparison."""
    _validate(rate, modalities)
    artificial = _empty_masks(masks)
    generator = torch.Generator().manual_seed(seed)
    originals = _original_masks(masks)
    for modality in modalities:
        for index, mask in enumerate(originals[modality]):
            valid = torch.where(mask)[0]
            count = _mask_count(len(valid), rate)
            if count:
                chosen = valid[torch.randperm(len(valid), generator=generator)[:count]]
                artificial[modality][index, chosen] = True
    return _result(masks, artificial, "literature_random", {"rate": rate, "seed": seed})


def apply_whole_modality_mask(masks: dict[str, torch.Tensor], subset: str) -> dict[str, Any]:
    """Hide each originally valid timestep in a literature missing-modality subset."""
    hidden = {"T": "text", "A": "audio", "V": "vision"}
    if subset not in {"T", "A", "V", "TA", "TV", "AV"}:
        raise ValueError(f"Unsupported whole-modality subset: {subset}")
    artificial = _empty_masks(masks)
    originals = _original_masks(masks)
    for code in subset:
        modality = hidden[code]
        artificial[modality] = originals[modality].clone()
    return _result(masks, artificial, "literature_whole_modality", {"subset": subset})


def _original_masks(masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if set(MODALITIES).issubset(masks):
        source_masks = masks
    elif all(f"{modality}_mask" in masks for modality in MODALITIES):
        source_masks = {modality: masks[f"{modality}_mask"] for modality in MODALITIES}
    else:
        raise ValueError(f"Expected masks for {MODALITIES}")
    return {modality: source_masks[modality].bool().clone() for modality in MODALITIES}


def _empty_masks(masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {modality: torch.zeros_like(mask, dtype=torch.bool) for modality, mask in _original_masks(masks).items()}


def _mask_count(valid_count: int, rate: float) -> int:
    return min(valid_count, max(1, math.floor(valid_count * rate + 0.5))) if valid_count else 0


def _validate(rate: float, modalities: tuple[str, ...]) -> None:
    if not 0 < rate <= 1:
        raise ValueError("Mask rate must be in (0, 1]")
    if not modalities or any(modality not in MODALITIES for modality in modalities):
        raise ValueError(f"Unsupported modalities: {modalities}")


def _result(
    masks: dict[str, torch.Tensor],
    artificial: dict[str, torch.Tensor],
    protocol: str,
    condition: dict[str, Any],
) -> dict[str, Any]:
    originals = _original_masks(masks)
    observed = {modality: originals[modality] & ~artificial[modality] for modality in MODALITIES}
    result = {
        "original_masks": originals,
        "artificial_masks": artificial,
        "observed_masks": observed,
        "protocol": protocol,
        "condition": condition,
    }
    if "audio" in masks and "vision" in masks:
        masked_inputs = dict(masks)
        for modality in ("audio", "vision"):
            masked_inputs[modality] = masks[modality].masked_fill(~observed[modality].unsqueeze(-1), 0)
        result["masked_inputs"] = masked_inputs
    return result

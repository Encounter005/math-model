from __future__ import annotations

from typing import Any

import torch


def make_augmented_view(
    batch: dict[str, Any],
    seed: int,
    rates: tuple[float, ...],
    feature_dropout: float,
) -> dict[str, Any]:
    """Hide a contiguous natural span from randomly selected modalities."""
    if not rates or any(not 0 < rate <= 1 for rate in rates):
        raise ValueError("rates must contain values in (0, 1]")
    if not 0 <= feature_dropout <= 1:
        raise ValueError("feature_dropout must be in [0, 1]")

    augmented = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    generator = torch.Generator().manual_seed(seed)
    modalities = ("text", "audio", "vision")
    masks = {name: augmented[f"{name}_mask"] for name in modalities}

    for index in range(masks["text"].shape[0]):
        available = [name for name in modalities if masks[name][index].any()]
        selected_count = int(
            torch.randint(1, len(available) + 1, (1,), generator=generator).item()
        )
        for selected_index in torch.randperm(len(available), generator=generator)[:selected_count]:
            name = available[int(selected_index)]
            _hide_contiguous_span(masks[name][index], rates, generator)

    for name in ("audio", "vision"):
        if feature_dropout:
            dropout = torch.rand(augmented[name].shape, generator=generator)
            dropout = dropout.to(augmented[name].device) < feature_dropout
            augmented[name].masked_fill_(dropout & masks[name].unsqueeze(-1), 0)
    return augmented


def _hide_contiguous_span(
    mask: torch.Tensor, rates: tuple[float, ...], generator: torch.Generator
) -> None:
    cpu_mask = mask.detach().cpu()
    spans = _valid_spans(cpu_mask)
    valid_count = int(cpu_mask.sum())
    rate = rates[int(torch.randint(len(rates), (1,), generator=generator).item())]
    length = max(1, round(valid_count * rate))
    candidates = [span for span in spans if span[1] - span[0] >= length]
    start, end = candidates[int(torch.randint(len(candidates), (1,), generator=generator).item())] if candidates else max(spans, key=lambda span: span[1] - span[0])
    length = min(length, end - start)
    offset = int(torch.randint(end - start - length + 1, (1,), generator=generator).item())
    mask[start + offset : start + offset + length] = False


def _valid_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, valid in enumerate(mask.tolist() + [False]):
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            spans.append((start, index))
            start = None
    return spans

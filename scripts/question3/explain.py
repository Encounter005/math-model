from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch

MODALITIES = ("text", "audio", "vision")


@torch.no_grad()
def explain_modalities(
    model: torch.nn.Module, sample: dict[str, Any]
) -> dict[str, Any]:
    full = model(sample)
    predicted_class = int(full["classification_logits"][0].argmax())
    score = full["classification_logits"][0, predicted_class]
    drops = {}
    for name in MODALITIES:
        hidden = dict(sample)
        hidden[f"{name}_mask"] = torch.zeros_like(sample[f"{name}_mask"])
        drops[name] = max(
            float(score - model(hidden)["classification_logits"][0, predicted_class]),
            0.0,
        )
    total = sum(drops.values())
    fallback = None
    if not total:
        primary = MODALITIES[int(full["fusion_weights"][0].argmax())]
        contributions = {name: 1.0 if name == primary else 0.0 for name in MODALITIES}
        fallback = "gate_tie_or_zero"
    else:
        contributions = {name: value / total for name, value in drops.items()}
        primary = max(contributions, key=contributions.get)
    return {
        "predicted_class": predicted_class,
        "raw_score_drops": drops,
        "contributions": contributions,
        "primary_modality": primary,
        "contribution_fallback": fallback,
    }


@torch.no_grad()
def explain_windows(
    model: torch.nn.Module,
    sample: dict[str, Any],
    window_length: int,
    stride: int,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    full = model(sample)
    predicted = int(full["classification_logits"][0].argmax())
    score = full["classification_logits"][0, predicted]
    result = {name: [] for name in MODALITIES}
    for modality in MODALITIES:
        mask = sample[f"{modality}_mask"][0]
        candidates = []
        for start in range(0, len(mask) - window_length + 1, stride):
            end = start + window_length
            if not mask[start:end].all():
                continue
            hidden = dict(sample)
            hidden[f"{modality}_mask"] = sample[f"{modality}_mask"].clone()
            hidden[f"{modality}_mask"][:, start:end] = False
            drop = max(
                float(score - model(hidden)["classification_logits"][0, predicted]), 0.0
            )
            candidates.append(
                {
                    "start_position": start,
                    "end_position": end - 1,
                    "raw_score_drop": drop,
                    "temporal_weight": float(
                        full["temporal_weights"][
                            0, MODALITIES.index(modality), start:end
                        ].mean()
                    ),
                }
            )
        selected = []
        for row in sorted(
            candidates, key=lambda row: row["raw_score_drop"], reverse=True
        ):
            if all(
                row["end_position"] < old["start_position"]
                or row["start_position"] > old["end_position"]
                for old in selected
            ):
                selected.append(row)
            if len(selected) == top_k:
                break
        total = sum(row["raw_score_drop"] for row in selected)
        for row in selected:
            row["normalized_importance"] = (
                row["raw_score_drop"] / total if total else 0.0
            )
        result[modality] = selected
    return result


def video_interval(
    start: int, end: int, duration_seconds: float, average_fps: float
) -> dict[str, float | int]:
    start_sec = start * duration_seconds / 50
    end_sec = (end + 1) * duration_seconds / 50
    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "frame_start": int(start_sec * average_fps),
        "frame_end": int(end_sec * average_fps) - 1,
    }


def probe_video(path: Path) -> tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Video unavailable: {path}")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frame_rate, duration = result.stdout.splitlines()
    numerator, denominator = frame_rate.split("/")
    return float(duration), float(numerator) / float(denominator)

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_question3_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors while preserving sample identifiers and source text."""
    return {
        key: [sample[key] for sample in samples]
        if key in {"id", "raw_text"}
        else None
        if samples[0][key] is None
        else torch.stack([sample[key] for sample in samples])
        for key in samples[0]
    }


class Question3Dataset(Dataset[dict[str, Any]]):
    """Aligned 50-step samples with natural modality-validity masks."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples
        self.ids = [str(sample["id"]) for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        text_bert = np.asarray(sample["text_bert"])
        audio = np.asarray(sample["audio"], dtype=np.float32)
        vision = np.asarray(sample["vision"], dtype=np.float32)
        text_mask = text_bert[1].astype(bool)
        return {
            "id": str(sample["id"]),
            "raw_text": sample["raw_text"],
            "input_ids": torch.as_tensor(text_bert[0], dtype=torch.long),
            "attention_mask": torch.as_tensor(text_bert[1], dtype=torch.long),
            "audio": torch.as_tensor(audio),
            "vision": torch.as_tensor(vision),
            "text_mask": torch.as_tensor(text_mask),
            "audio_mask": torch.as_tensor(text_mask & np.any(audio != 0, axis=-1)),
            "vision_mask": torch.as_tensor(text_mask & np.any(vision != 0, axis=-1)),
            "class_label": sample.get("class_label"),
            "regression_label": sample.get("regression_label"),
        }


def load_attachment4_dataset(directory: Path) -> Question3Dataset:
    """Load the unlabelled Attachment 4 aligned samples in lexical ID order."""
    samples = [_load_attachment4_sample(path) for path in sorted(directory.glob("??.pkl"))]
    if not samples:
        raise ValueError(f"No top-level two-character pickle files in {directory}")
    return Question3Dataset(samples)


def _load_attachment4_sample(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a sample payload in {path}")

    sample = payload.get("test", payload)
    if not isinstance(sample, dict):
        raise TypeError(f"Expected a sample payload in {path}")
    required = ("text_bert", "audio", "vision", "raw_text")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"Missing required fields in {path}: {', '.join(missing)}")
    wrapped = "test" in payload
    if wrapped and any(len(sample[key]) != 1 for key in required):
        raise ValueError(f"Expected exactly one sample in {path}")

    text_bert = np.asarray(sample["text_bert"][0] if wrapped else sample["text_bert"])
    audio = np.asarray(sample["audio"][0] if wrapped else sample["audio"])
    vision = np.asarray(sample["vision"][0] if wrapped else sample["vision"])
    if text_bert.ndim != 2 or text_bert.shape[0] < 2 or text_bert.shape[1] != 50:
        raise ValueError(f"Expected text_bert shape (*, 50) in {path}, got {text_bert.shape}")
    if audio.shape != (50, 74):
        raise ValueError(f"Expected audio shape (50, 74) in {path}, got {audio.shape}")
    if vision.shape != (50, 35):
        raise ValueError(f"Expected vision shape (50, 35) in {path}, got {vision.shape}")
    return {
        "id": path.stem,
        "raw_text": sample["raw_text"][0] if wrapped else sample["raw_text"],
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
    }

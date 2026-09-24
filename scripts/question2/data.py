from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SENTIMENT_LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate labelled training and unlabelled attachment-3 samples alike."""
    return {
        key: [sample[key] for sample in samples]
        if key == "id"
        else None
        if samples[0][key] is None
        else torch.stack([sample[key] for sample in samples])
        for key in samples[0]
    }


class MultimodalDataset(Dataset[dict[str, Any]]):
    """Aligned multimodal samples with original validity masks."""

    def __init__(self, payload: dict[str, Any], ids: list[str] | None = None) -> None:
        self.payload = payload
        self.ids = ids or [str(value) for value in payload["id"]]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text_bert = np.asarray(self.payload["text_bert"][index])
        text_mask = text_bert[1].astype(bool)
        audio = np.asarray(self.payload["audio"][index], dtype=np.float32)
        vision = np.asarray(self.payload["vision"][index], dtype=np.float32)
        sample: dict[str, Any] = {
            "id": self.ids[index],
            "input_ids": torch.as_tensor(text_bert[0], dtype=torch.long),
            "attention_mask": torch.as_tensor(text_bert[1], dtype=torch.long),
            "audio": torch.as_tensor(audio),
            "vision": torch.as_tensor(vision),
            "text_mask": torch.as_tensor(text_mask),
            "audio_mask": torch.as_tensor(text_mask & np.any(audio != 0, axis=-1)),
            "vision_mask": torch.as_tensor(text_mask & np.any(vision != 0, axis=-1)),
            "class_label": None,
            "regression_label": None,
        }
        if "classification_labels" in self.payload:
            label = self.payload["classification_labels"][index]
            sample["class_label"] = torch.tensor(_class_label(label), dtype=torch.long)
        if "regression_labels" in self.payload:
            sample["regression_label"] = torch.tensor(
                self.payload["regression_labels"][index], dtype=torch.float32
            )
        return sample


def _class_label(label: Any) -> int:
    if isinstance(label, str):
        return SENTIMENT_LABELS[label]
    return int(label)


def load_training_datasets(path: Path) -> dict[str, MultimodalDataset]:
    """Load the labelled attachment-2 splits without using its text feature."""
    with path.open("rb") as file:
        payload = pickle.load(file)
    return {split: MultimodalDataset(payload[split]) for split in ("train", "valid", "test")}


def load_inference_dataset(directory: Path) -> MultimodalDataset:
    """Load attachment-3 files strictly as unlabelled inference samples."""
    samples: list[dict[str, Any]] = []
    ids: list[str] = []
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as file:
            payload = pickle.load(file)["test"]
        if len(payload["text_bert"]) != 1:
            raise ValueError(f"Expected one sample in {path}")
        samples.append(payload)
        ids.append(path.name)
    if not samples:
        raise ValueError(f"No inference pickle files in {directory}")
    return MultimodalDataset(
        {
            key: np.concatenate([sample[key] for sample in samples], axis=0)
            for key in ("text_bert", "audio", "vision")
        },
        ids,
    )

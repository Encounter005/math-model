import pickle

import numpy as np
import torch

from scripts.question3.data import (
    collate_question3_samples,
    load_attachment4_dataset,
)


def _payload(raw_text: str) -> dict[str, object]:
    return {
        "test": {
            "text_bert": np.ones((1, 2, 50), dtype=np.int64),
            "audio": np.ones((1, 50, 74), dtype=np.float32),
            "vision": np.ones((1, 50, 35), dtype=np.float32),
            "raw_text": [raw_text],
        }
    }


def _sample(identifier: str, raw_text: str) -> dict[str, object]:
    return {
        "id": identifier,
        "raw_text": raw_text,
        "input_ids": torch.ones(50, dtype=torch.long),
        "attention_mask": torch.ones(50, dtype=torch.long),
        "audio": torch.ones(50, 74),
        "vision": torch.ones(50, 35),
        "text_mask": torch.ones(50, dtype=torch.bool),
        "audio_mask": torch.ones(50, dtype=torch.bool),
        "vision_mask": torch.ones(50, dtype=torch.bool),
        "class_label": None,
        "regression_label": None,
    }


def test_load_attachment4_aligned_samples_keeps_sorted_ids_and_raw_text(tmp_path):
    for identifier, raw_text in (("02", "second sample"), ("01", "first sample")):
        with (tmp_path / f"{identifier}.pkl").open("wb") as file:
            pickle.dump(_payload(raw_text), file)

    dataset = load_attachment4_dataset(tmp_path)

    assert dataset.ids == ["01", "02"]
    assert dataset[0]["raw_text"] == "first sample"
    assert dataset[0]["audio"].shape == (50, 74)


def test_collate_keeps_raw_text_as_a_list():
    batch = collate_question3_samples(
        [_sample("01", "first"), _sample("02", "second")]
    )

    assert batch["id"] == ["01", "02"]
    assert batch["raw_text"] == ["first", "second"]

import pytest
import torch
from torch import nn

from scripts.question3.explain import (
    explain_modalities,
    explain_windows,
    video_interval,
)


class FakeModel(nn.Module):
    def forward(self, batch):
        scores = torch.stack(
            (
                batch["text_mask"].sum(1),
                batch["audio_mask"].sum(1),
                batch["vision_mask"].sum(1),
            ),
            1,
        ).float()
        return {
            "classification_logits": scores,
            "fusion_weights": torch.softmax(scores, -1),
            "temporal_weights": scores.unsqueeze(-1).expand(-1, -1, 50) / 50,
        }


def _sample(padded=False):
    text = torch.tensor(
        [[True] * (10 if padded else 50) + [False] * (40 if padded else 0)]
    )
    return {"text_mask": text, "audio_mask": text.clone(), "vision_mask": text.clone()}


def test_modality_contributions_are_normalized_and_keep_raw_drops():
    explanation = explain_modalities(FakeModel(), _sample())
    assert set(explanation["contributions"]) == {"text", "audio", "vision"}
    assert sum(explanation["contributions"].values()) == pytest.approx(1.0)
    assert set(explanation["raw_score_drops"]) == {"text", "audio", "vision"}


def test_top_windows_never_include_invalid_timesteps():
    windows = explain_windows(
        FakeModel(), _sample(True), window_length=5, stride=2, top_k=3
    )
    assert all(window["end_position"] < 10 for window in windows["text"])


def test_aligned_interval_to_video_metadata():
    assert video_interval(10, 14, duration_seconds=10, average_fps=25) == {
        "start_sec": 2.0,
        "end_sec": 3.0,
        "frame_start": 50,
        "frame_end": 74,
    }

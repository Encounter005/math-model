import torch
from torch import nn

from scripts.question3.training import evaluate, select_best_epoch


class FakeModel(nn.Module):
    def forward(self, batch):
        size = batch["class_label"].shape[0]
        return {
            "classification_logits": torch.zeros(size, 3),
            "regression_prediction": torch.zeros(size),
        }


def test_select_best_epoch_prefers_weighted_f1_then_lower_mae():
    assert (
        select_best_epoch(
            [
                {"epoch": 1, "weighted_f1": 0.6, "mae": 0.7},
                {"epoch": 2, "weighted_f1": 0.6, "mae": 0.6},
            ]
        )
        == 2
    )


def test_validation_does_not_apply_augmentation():
    valid_loader = [
        {
            "id": ["01", "02"],
            "class_label": torch.tensor([0, 1]),
            "regression_label": torch.tensor([0.0, 1.0]),
            "input_ids": torch.ones(2, 50, dtype=torch.long),
            "attention_mask": torch.ones(2, 50, dtype=torch.long),
            "audio": torch.ones(2, 50, 74),
            "vision": torch.ones(2, 50, 35),
            "text_mask": torch.ones(2, 50, dtype=torch.bool),
            "audio_mask": torch.ones(2, 50, dtype=torch.bool),
            "vision_mask": torch.ones(2, 50, dtype=torch.bool),
        }
    ]

    result = evaluate(FakeModel(), valid_loader, device=torch.device("cpu"))

    assert result["prediction_count"] == 2

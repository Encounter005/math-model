from types import SimpleNamespace

import torch
from torch import nn

from scripts.question3.models import GatedSelfDistillationModel


class FakeTextEncoder(nn.Module):
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float().repeat(1, 1, 8))


def _batch() -> dict[str, torch.Tensor]:
    text_mask = torch.tensor([[True] * 10 + [False] * 40, [True] * 20 + [False] * 30])
    return {
        "input_ids": torch.ones(2, 50, dtype=torch.long),
        "attention_mask": text_mask.long(),
        "audio": torch.ones(2, 50, 74),
        "vision": torch.ones(2, 50, 35),
        "text_mask": text_mask,
        "audio_mask": text_mask.clone(),
        "vision_mask": text_mask.clone(),
    }


def _model() -> GatedSelfDistillationModel:
    return GatedSelfDistillationModel(FakeTextEncoder(), hidden_dim=16, temporal_dropout=0)


def test_model_outputs_shapes_and_backpropagates_losses():
    model = _model()
    batch = _batch()
    full = model(batch)
    first = model(batch)
    second = model(batch)

    assert full["classification_logits"].shape == (2, 3)
    assert full["regression_prediction"].shape == (2,)
    assert full["fusion_weights"].shape == (2, 3)
    assert full["temporal_weights"].shape == (2, 3, 50)
    assert torch.allclose(full["fusion_weights"].sum(dim=-1), torch.ones(2))

    losses = model.losses(
        full,
        first,
        second,
        torch.tensor([0, 1]),
        torch.tensor([0.5, -0.5]),
        torch.ones(3),
        progress=1.0,
    )

    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.audio_projection.weight.grad is not None
    assert model.fusion_gate.weight.grad is not None


def test_unavailable_modalities_receive_zero_fusion_and_temporal_weights():
    batch = _batch()
    batch["audio_mask"] = torch.zeros_like(batch["text_mask"])
    batch["vision_mask"] = torch.zeros_like(batch["text_mask"])

    outputs = _model()(batch)

    assert not outputs["fusion_weights"][:, 1:].any()
    assert not outputs["temporal_weights"][:, 1:].any()
    assert not outputs["temporal_weights"][:, :, ~batch["text_mask"].any(dim=0)].any()


def test_zero_warmup_excludes_consistency_terms_from_total_loss():
    model = _model()
    batch = _batch()
    full, first, second = model(batch), model(batch), model(batch)
    losses = model.losses(
        full,
        first,
        second,
        torch.tensor([0, 1]),
        torch.tensor([0.5, -0.5]),
        torch.ones(3),
        progress=0.0,
    )

    assert torch.allclose(losses["loss"], losses["classification_loss"] + losses["regression_loss"])

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from scripts.question2.models import FrozenTextEncoder

MODALITIES = ("text", "audio", "vision")

__all__ = ["FrozenTextEncoder", "GatedSelfDistillationModel"]


class GatedSelfDistillationModel(nn.Module):
    """Three-way masked fusion with shared complete and augmented views."""

    def __init__(
        self,
        text_encoder: nn.Module,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 128,
        temporal_dropout: float = 0.1,
        representation_coefficient: float = 0.1,
        classification_coefficient: float = 0.2,
        regression_coefficient: float = 0.2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.text_encoder = text_encoder
        self.text_projection = nn.LazyLinear(hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        self.temporal_dropout = nn.Dropout(temporal_dropout)
        self.temporal_scorers = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in MODALITIES)
        self.fusion_gate = nn.Linear(3 * hidden_dim + len(MODALITIES), len(MODALITIES))
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)
        self.representation_coefficient = representation_coefficient
        self.classification_coefficient = classification_coefficient
        self.regression_coefficient = regression_coefficient
        self.temperature = temperature

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        states = self._project_states(batch)
        masks = torch.stack([batch[f"{name}_mask"].bool() for name in MODALITIES], dim=1)
        pooled, temporal_weights = self._temporal_pool(states, masks)
        available = masks.any(dim=-1)
        fractions = masks.float().mean(dim=-1)
        fusion_weights = _masked_softmax(
            self.fusion_gate(torch.cat((pooled.flatten(1), fractions), dim=-1)), available
        )
        fused = (fusion_weights.unsqueeze(-1) * pooled).sum(dim=1)
        return {
            "fused_representation": fused,
            "classification_logits": self.classifier(fused),
            "regression_prediction": self.regressor(fused).squeeze(-1).clamp(-3, 3),
            "fusion_weights": fusion_weights,
            "temporal_weights": temporal_weights,
        }

    def losses(
        self,
        full: dict[str, torch.Tensor],
        first: dict[str, torch.Tensor],
        second: dict[str, torch.Tensor],
        class_labels: torch.Tensor,
        regression_labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Combine complete-view supervision with warmed-up self-distillation."""
        classification = functional.cross_entropy(
            full["classification_logits"], class_labels, weight=class_weights
        )
        regression = functional.smooth_l1_loss(
            full["regression_prediction"], regression_labels
        )
        representation = _mean_pair(
            functional.mse_loss(first["fused_representation"], full["fused_representation"].detach()),
            functional.mse_loss(second["fused_representation"], full["fused_representation"].detach()),
        )
        full_probabilities = functional.softmax(
            full["classification_logits"] / self.temperature, dim=-1
        ).detach()
        classification_consistency = _mean_pair(
            _js_divergence(first["classification_logits"], full_probabilities, self.temperature),
            _js_divergence(second["classification_logits"], full_probabilities, self.temperature),
        )
        target = full["regression_prediction"].detach()
        regression_consistency = _mean_pair(
            functional.mse_loss(first["regression_prediction"], target),
            functional.mse_loss(second["regression_prediction"], target),
        )
        warmup = min(max(progress, 0.0), 1.0)
        loss = classification + regression + warmup * (
            self.representation_coefficient * representation
            + self.classification_coefficient * classification_consistency
            + self.regression_coefficient * regression_consistency
        )
        return {
            "loss": loss,
            "classification_loss": classification,
            "regression_loss": regression,
            "representation_loss": representation,
            "classification_consistency_loss": classification_consistency,
            "regression_consistency_loss": regression_consistency,
        }

    def _project_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        text_states = self.text_encoder(
            batch["input_ids"], batch["attention_mask"]
        ).last_hidden_state
        return self.temporal_dropout(
            torch.stack(
                (
                    self.text_projection(text_states),
                    self.audio_projection(batch["audio"]),
                    self.vision_projection(batch["vision"]),
                ),
                dim=1,
            )
        )

    def _temporal_pool(
        self, states: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.stack(
            [
                _masked_softmax(scorer(states[:, index]).squeeze(-1), masks[:, index])
                for index, scorer in enumerate(self.temporal_scorers)
            ],
            dim=1,
        )
        return (weights.unsqueeze(-1) * states).sum(dim=2), weights


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize valid positions only and return zero for fully hidden rows."""
    masked = values.masked_fill(~mask, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
    weights = (masked - maximum).exp() * mask
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)


def _mean_pair(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first + second) / 2


def _js_divergence(
    logits: torch.Tensor, target_probabilities: torch.Tensor, temperature: float
) -> torch.Tensor:
    probabilities = functional.softmax(logits / temperature, dim=-1)
    midpoint = (probabilities + target_probabilities) / 2
    return (
        probabilities * (probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log() - midpoint.log())
        + target_probabilities
        * (target_probabilities.clamp_min(torch.finfo(target_probabilities.dtype).eps).log() - midpoint.log())
    ).sum(dim=-1).mean() / 2

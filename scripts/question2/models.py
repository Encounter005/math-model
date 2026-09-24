from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from transformers import AutoModel


class FrozenTextEncoder(nn.Module):
    """Locally cached DistilBERT kept outside the trainable graph."""

    def __init__(self, cache_path: Path) -> None:
        super().__init__()
        snapshots = cache_path / "snapshots"
        model_path = next(snapshots.iterdir()) if snapshots.is_dir() else cache_path
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> FrozenTextEncoder:
        del mode
        return super().train(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask)


def masked_temporal_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool valid timesteps, producing zeros when a modality is unavailable."""
    weights = mask.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class SharedBackbone(nn.Module):
    """Shared text, audio, and vision encoder with sentiment task heads."""

    def __init__(self, text_encoder: nn.Module, audio_dim: int = 74, vision_dim: int = 35, hidden_dim: int = 128) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.text_projection = nn.LazyLinear(hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.regressor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        states = self.project_states(batch)
        modality_states = torch.stack(
            tuple(masked_temporal_mean(states[:, index], batch[f"{name}_mask"]) for index, name in enumerate(MODALITIES)),
            dim=1,
        )
        available = torch.stack(
            (
                batch["text_mask"].any(dim=1),
                batch["audio_mask"].any(dim=1),
                batch["vision_mask"].any(dim=1),
            ),
            dim=1,
        ).unsqueeze(-1)
        fused = (modality_states * available).sum(dim=1) / available.sum(dim=1).clamp_min(1)
        return self.predict(fused)

    def project_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the complete 128-D temporal latents used as reconstruction targets."""
        text_states = self.text_projection(
            self.text_encoder(batch["input_ids"], batch["attention_mask"]).last_hidden_state
        )
        return torch.stack(
            (text_states, self.audio_projection(batch["audio"]), self.vision_projection(batch["vision"])), dim=1
        )

    def predict(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "fused_representation": fused,
            "classification_logits": self.classifier(fused),
            "regression_prediction": self.regressor(fused).squeeze(-1).clamp(-3, 3),
        }

    def supervised_loss(
        self,
        outputs: dict[str, torch.Tensor],
        class_labels: torch.Tensor,
        regression_labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        classification_loss = functional.cross_entropy(
            outputs["classification_logits"], class_labels, weight=class_weights
        )
        regression_loss = functional.smooth_l1_loss(outputs["regression_prediction"], regression_labels)
        return {
            "loss": classification_loss + regression_loss,
            "classification_loss": classification_loss,
            "regression_loss": regression_loss,
        }


MODALITIES = ("text", "audio", "vision")


class ReconstructionModel(SharedBackbone):
    """TgRN-inspired latent reconstruction, rather than raw-feature reconstruction."""

    def __init__(self, text_encoder: nn.Module, audio_dim: int = 74, vision_dim: int = 35, hidden_dim: int = 128) -> None:
        super().__init__(text_encoder, audio_dim, vision_dim, hidden_dim)
        self.position = nn.Embedding(50, hidden_dim)
        self.reconstruction_attention = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True) for _ in MODALITIES
        )

    def forward(
        self, batch: dict[str, torch.Tensor], artificial_masks: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        projected = self.project_states(batch)
        original_masks = torch.stack(tuple(batch[f"{name}_mask"] for name in MODALITIES), dim=1).bool()
        artificial = torch.stack(tuple(artificial_masks[name] for name in MODALITIES), dim=1).bool()
        observed = original_masks & ~artificial
        reconstructed = self._reconstruct(projected, observed)
        completed = torch.where(artificial.unsqueeze(-1), reconstructed, projected)
        pooled = torch.stack(
            tuple(masked_temporal_mean(completed[:, index], original_masks[:, index]) for index in range(len(MODALITIES))),
            dim=1,
        )
        available = original_masks.any(dim=-1).unsqueeze(-1)
        fused = (pooled * available).sum(dim=1) / available.sum(dim=1).clamp_min(1)
        errors = torch.stack(
            tuple(
                self._masked_error(reconstructed[:, index], projected[:, index], original_masks[:, index] & artificial[:, index])
                for index in range(len(MODALITIES))
            ),
            dim=1,
        )
        hidden = original_masks & artificial
        loss = functional.smooth_l1_loss(reconstructed[hidden], projected[hidden]) if hidden.any() else projected.sum() * 0
        return {
            **self.predict(fused),
            "projected_states": projected,
            "reconstructed_states": reconstructed,
            "completed_states": completed,
            "reconstruction_loss": loss,
            "reconstruction_errors": errors,
        }

    def _reconstruct(self, states: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        batch_size, _, steps, hidden_dim = states.shape
        positions = self.position(torch.arange(steps, device=states.device)).expand(batch_size, -1, -1)
        context = states.flatten(1, 2)
        context_mask = observed.flatten(1, 2)
        reconstructed = []
        for attention in self.reconstruction_attention:
            values = torch.zeros_like(positions)
            valid = context_mask.any(dim=1)
            if valid.any():
                values[valid] = attention(
                    positions[valid], context[valid], context[valid], key_padding_mask=~context_mask[valid]
                )[0]
            reconstructed.append(values)
        return torch.stack(reconstructed, dim=1).reshape(batch_size, len(MODALITIES), steps, hidden_dim)

    @staticmethod
    def _masked_error(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        errors = torch.zeros(mask.shape[0], device=prediction.device, dtype=prediction.dtype)
        for index, sample_mask in enumerate(mask):
            if sample_mask.any():
                errors[index] = functional.smooth_l1_loss(prediction[index][sample_mask], target[index][sample_mask])
        return errors


class GatedFusionModel(SharedBackbone):
    """AGFN-inspired entropy and learned importance gating for local missingness."""

    def __init__(
        self,
        text_encoder: nn.Module,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 128,
        entropy_temperature: float = 1.0,
    ) -> None:
        super().__init__(text_encoder, audio_dim, vision_dim, hidden_dim)
        self.entropy_temperature = entropy_temperature
        self.importance_gate = nn.Linear(3 * hidden_dim + len(MODALITIES), len(MODALITIES))
        self.mixture_gate = nn.Linear(3 * hidden_dim + len(MODALITIES), 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        states = self.project_states(batch)
        masks = torch.stack(tuple(batch[f"{name}_mask"] for name in MODALITIES), dim=1).bool()
        pooled = torch.stack(
            tuple(masked_temporal_mean(states[:, index], masks[:, index]) for index in range(len(MODALITIES))), dim=1
        )
        available = masks.any(dim=-1)
        observed_fractions = masks.float().mean(dim=-1)
        gate_input = torch.cat((pooled.flatten(1), observed_fractions), dim=-1)
        entropy = self._feature_entropy(pooled)
        reliability = _masked_softmax((1 - entropy) / self.entropy_temperature, available)
        importance = _masked_softmax(self.importance_gate(gate_input), available)
        mixture = torch.sigmoid(self.mixture_gate(gate_input))
        final_weights = mixture * reliability + (1 - mixture) * importance
        fused = (final_weights.unsqueeze(-1) * pooled).sum(dim=1)
        return {
            **self.predict(fused),
            "reliability_gates": reliability,
            "importance_gates": importance,
            "mixture_coefficient": mixture.squeeze(-1),
            "fused_modality_weights": final_weights,
        }

    def vat_loss(self, fused: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
        """Penalize the adversarially perturbed classifier distribution."""
        reference = functional.softmax(self.classifier(fused), dim=-1).detach()
        noise = functional.normalize(torch.randn_like(fused), dim=-1).requires_grad_()
        divergence = functional.kl_div(functional.log_softmax(self.classifier(fused + noise), dim=-1), reference, reduction="batchmean")
        direction = torch.autograd.grad(divergence, noise, retain_graph=True)[0]
        perturbation = epsilon * functional.normalize(direction, dim=-1).detach()
        return functional.kl_div(
            functional.log_softmax(self.classifier(fused + perturbation), dim=-1), reference, reduction="batchmean"
        )

    @staticmethod
    def _feature_entropy(states: torch.Tensor) -> torch.Tensor:
        probabilities = functional.softmax(states.abs(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(states.dtype).eps).log()).sum(dim=-1)
        return entropy / math.log(max(states.shape[-1], 2))


def _masked_softmax(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Normalize only available modalities, returning all-zero for empty rows."""
    masked = values.masked_fill(~available, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(available.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
    weights = (masked - maximum).exp() * available
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)


class MissModalAlignmentModel(SharedBackbone):
    """MissModal-inspired alignment of shared complete and incomplete views."""

    def __init__(
        self,
        text_encoder: nn.Module,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 128,
        semantic_coefficient: float = 1.0,
        distance_coefficient: float = 1.0,
        geometry_coefficient: float = 1.0,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(text_encoder, audio_dim, vision_dim, hidden_dim)
        self.semantic_coefficient = semantic_coefficient
        self.distance_coefficient = distance_coefficient
        self.geometry_coefficient = geometry_coefficient
        self.temperature = temperature

    def forward(
        self, batch: dict[str, torch.Tensor], incomplete_masks: list[dict[str, torch.Tensor]]
    ) -> dict[str, Any]:
        states = self.project_states(batch)
        complete_masks = torch.stack(tuple(batch[f"{name}_mask"] for name in MODALITIES), dim=1).bool()
        complete = self._view(states, complete_masks)
        incomplete = []
        for artificial_masks in incomplete_masks:
            artificial = torch.stack(tuple(artificial_masks[name] for name in MODALITIES), dim=1).bool()
            incomplete.append(self._view(states, complete_masks & ~artificial))
        return {"complete": complete, "incomplete": incomplete}

    def alignment_losses(
        self,
        outputs: dict[str, Any],
        class_labels: torch.Tensor,
        regression_labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        complete_task = self.supervised_loss(outputs["complete"], class_labels, regression_labels, class_weights)["loss"]
        views: list[dict[str, torch.Tensor]] = outputs["incomplete"]
        if not views:
            zero = complete_task * 0
            return {
                "loss": complete_task,
                "complete_task_loss": complete_task,
                "incomplete_task_losses": zero.unsqueeze(0)[:0],
                "paired_distance_loss": zero,
                "geometry_loss": zero,
            }
        incomplete_tasks = torch.stack(
            [self.supervised_loss(view, class_labels, regression_labels, class_weights)["loss"] for view in views]
        )
        complete_representation = outputs["complete"]["fused_representation"]
        paired_distance = torch.stack(
            [functional.mse_loss(view["fused_representation"], complete_representation) for view in views]
        ).mean()
        geometry = torch.stack([self._geometry_loss(view["fused_representation"], complete_representation) for view in views]).mean()
        loss = (
            complete_task
            + self.semantic_coefficient * incomplete_tasks.mean()
            + self.distance_coefficient * paired_distance
            + self.geometry_coefficient * geometry
        )
        return {
            "loss": loss,
            "complete_task_loss": complete_task,
            "incomplete_task_losses": incomplete_tasks,
            "paired_distance_loss": paired_distance,
            "geometry_loss": geometry,
        }

    def _view(self, states: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = torch.stack(
            tuple(masked_temporal_mean(states[:, index], masks[:, index]) for index in range(len(MODALITIES))), dim=1
        )
        available = masks.any(dim=-1).unsqueeze(-1)
        fused = (pooled * available).sum(dim=1) / available.sum(dim=1).clamp_min(1)
        return self.predict(fused)

    def _geometry_loss(self, incomplete: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
        similarities = functional.normalize(incomplete, dim=-1) @ functional.normalize(complete, dim=-1).T
        return functional.cross_entropy(similarities / self.temperature, torch.arange(len(incomplete), device=incomplete.device))

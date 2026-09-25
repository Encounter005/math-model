from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.question3.artifacts import (
    write_attachment4_outputs,
    write_explanation_cards,
    write_seed_predictions,
)
from scripts.question3.data import collate_question3_samples, load_attachment4_dataset
from scripts.question3.explain import (
    explain_modalities,
    explain_windows,
    probe_video,
    video_interval,
)
from scripts.question3.models import FrozenTextEncoder, GatedSelfDistillationModel


def ensemble_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("At least one seed prediction is required")
    probabilities = [
        statistics.fmean(row["probabilities"][index] for row in predictions)
        for index in range(3)
    ]
    return {
        "probabilities": probabilities,
        "class_prediction": max(range(3), key=probabilities.__getitem__),
        "regression_prediction": statistics.fmean(
            row["regression_prediction"] for row in predictions
        ),
    }


class _SelectedEnsemble:
    """Expose averaged selected-seed outputs to the existing occlusion explainers."""

    def __init__(self, models: list[GatedSelfDistillationModel]) -> None:
        self.models = models

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = [model(batch) for model in self.models]
        probabilities = torch.stack(
            [torch.softmax(output["classification_logits"], dim=-1) for output in outputs]
        ).mean(dim=0)
        return {
            "classification_logits": probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log(),
            "regression_prediction": torch.stack(
                [output["regression_prediction"] for output in outputs]
            ).mean(dim=0),
            "fusion_weights": torch.stack(
                [output["fusion_weights"] for output in outputs]
            ).mean(dim=0),
            "temporal_weights": torch.stack(
                [output["temporal_weights"] for output in outputs]
            ).mean(dim=0),
        }


def _build_model(
    config: dict[str, Any], sample: dict[str, Any], checkpoint: Path, device: torch.device
) -> GatedSelfDistillationModel:
    model = GatedSelfDistillationModel(
        FrozenTextEncoder(Path(config["paths"]["model_cache"])),
        audio_dim=sample["audio"].shape[-1],
        vision_dim=sample["vision"].shape[-1],
        hidden_dim=int(config["model"]["hidden_dim"]),
        temporal_dropout=float(config["model"]["temporal_dropout"]),
        representation_coefficient=float(config["loss"]["representation_coefficient"]),
        classification_coefficient=float(config["loss"]["classification_coefficient"]),
        regression_coefficient=float(config["loss"]["regression_coefficient"]),
        temperature=float(config["loss"]["temperature"]),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint: {checkpoint}")
    model.load_state_dict(state)
    return model.to(device).eval()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Question 3 Attachment 4 inference."
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="Temporary inference seed subset."
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    root = Path(__file__).resolve().parents[2]
    paths = {key: (root / value).resolve() for key, value in config["paths"].items()}
    seeds = args.seeds or config["training"]["seeds"]
    missing = [
        str(paths["artifact_root"] / "seeds" / f"seed_{seed}" / "checkpoint_best.pt")
        for seed in seeds
        if not (
            paths["artifact_root"] / "seeds" / f"seed_{seed}" / "checkpoint_best.pt"
        ).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing selected checkpoints: " + ", ".join(missing))
    dataset = load_attachment4_dataset(paths["attachment4_dir"])
    if len(dataset) != 20:
        raise ValueError(f"Expected 20 Attachment 4 samples, found {len(dataset)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        _build_model(
            config,
            dataset[0],
            paths["artifact_root"] / "seeds" / f"seed_{seed}" / "checkpoint_best.pt",
            device,
        )
        for seed in seeds
    ]
    ensemble = _SelectedEnsemble(models)
    predictions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seed_rows = {seed: [] for seed in seeds}
    cards: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_question3_samples)
    for index, raw_batch in enumerate(loader, start=1):
        batch = _to_device(raw_batch, device)
        per_seed = []
        for seed, model in zip(seeds, models, strict=True):
            output = model(batch)
            probabilities = torch.softmax(output["classification_logits"][0], dim=-1).tolist()
            row = {
                "id": batch["id"][0],
                "probabilities": probabilities,
                "regression_prediction": float(output["regression_prediction"][0].detach()),
            }
            per_seed.append(row)
            seed_rows[seed].append(
                {
                    "id": row["id"],
                    "class_prediction": int(max(range(3), key=probabilities.__getitem__)),
                    "probability_negative": probabilities[0],
                    "probability_neutral": probabilities[1],
                    "probability_positive": probabilities[2],
                    "regression_prediction": row["regression_prediction"],
                }
            )
        combined = ensemble_predictions(per_seed)
        modalities = explain_modalities(ensemble, batch)
        windows = explain_windows(
            ensemble,
            batch,
            int(config["explanation"]["window_length"]),
            int(config["explanation"]["window_stride"]),
            int(config["explanation"]["top_k"]),
        )
        identifier = batch["id"][0]
        duration, fps = probe_video(paths["video_dir"] / f"{identifier}.mp4")
        prediction = {
            "id": identifier,
            "class_prediction": combined["class_prediction"],
            "probability_negative": combined["probabilities"][0],
            "probability_neutral": combined["probabilities"][1],
            "probability_positive": combined["probabilities"][2],
            "regression_prediction": combined["regression_prediction"],
            "primary_modality": modalities["primary_modality"],
            "contribution_fallback": modalities["contribution_fallback"] or "",
            **{f"{name}_contribution": modalities["contributions"][name] for name in modalities["contributions"]},
            **{f"{name}_score_drop": modalities["raw_score_drops"][name] for name in modalities["raw_score_drops"]},
        }
        predictions.append(prediction)
        card_evidence = []
        for modality, rows in windows.items():
            for rank, row in enumerate(rows, start=1):
                item = {
                    "id": identifier,
                    "modality": modality,
                    "rank": rank,
                    **row,
                    **video_interval(row["start_position"], row["end_position"], duration, fps),
                    "raw_text": batch["raw_text"][0],
                    "mapping_precision": "aligned_timestep",
                }
                evidence.append(item)
                card_evidence.append(item)
        cards[identifier] = {"prediction": prediction, "evidence": card_evidence}
        elapsed = time.monotonic() - started
        eta = elapsed * (20 - index) / index
        print(
            f"infer {index:02d}/20 eta {int(eta) // 60:02d}:{int(eta) % 60:02d}",
            flush=True,
        )
    write_attachment4_outputs(paths["artifact_root"], predictions, evidence)
    for seed, rows in seed_rows.items():
        write_seed_predictions(paths["artifact_root"], seed, rows)
    write_explanation_cards(paths["artifact_root"], cards)


if __name__ == "__main__":
    main()

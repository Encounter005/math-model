"""Pool word-aligned modality features into the fixed 50-position layout."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


def _spans(length: int, positions: int) -> list[tuple[int, int]]:
    if length <= positions:
        return [(index, index + 1) for index in range(length)]
    edges = np.linspace(0, length, positions + 1, dtype=np.int64)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(positions)]


def _pool(values: np.ndarray, spans: list[tuple[int, int]], positions: int) -> np.ndarray:
    result = np.zeros((positions, values.shape[1]), dtype=np.float32)
    for position, (start, end) in enumerate(spans):
        result[position] = values[start:end].mean(axis=0)
    return result


def build_aligned_sample(
    record: dict,
    text: np.ndarray,
    text_word_indices: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
    positions: int = 50,
) -> dict:
    """Return padded or pooled arrays and the original-word position map."""
    if len(text) != len(audio) or len(text) != len(vision):
        raise ValueError(f"feature lengths do not agree for {record['id']}")
    spans = _spans(len(text), positions)
    pooled = {
        "text": _pool(text, spans, positions),
        "audio": _pool(audio, spans, positions),
        "vision": _pool(vision, spans, positions),
        "lengths": (len(spans), len(spans), len(spans)),
        "position_map": {str(int(word)): position for position, (start, end) in enumerate(spans) for word in text_word_indices[start:end]},
    }
    return pooled


def _load_features(directory: Path, sample_id: str, modality: str) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(directory / modality / f"{sample_id}.npz", allow_pickle=True)
    indices = np.asarray(data["word_indices"], dtype=np.int64) if "word_indices" in data else None
    return np.asarray(data["features"], dtype=np.float32), indices


def build_output(
    alignment_path: Path,
    manifest_path: Path,
    feature_root: Path,
    output_path: Path,
    positions: int = 50,
) -> None:
    """Build the attachment-2-compatible train split."""
    manifest = {row["id"]: row for row in csv.DictReader(manifest_path.open(newline="", encoding="utf-8"))}
    samples: list[dict] = []
    sidecars: list[dict] = []
    for line in alignment_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        sample_id = record["id"]
        text, word_indices = _load_features(feature_root, sample_id, "text")
        audio, audio_indices = _load_features(feature_root, sample_id, "audio")
        vision, vision_indices = _load_features(feature_root, sample_id, "vision")
        if audio_indices is not None and not np.array_equal(word_indices, audio_indices):
            raise ValueError(f"word indices do not agree for {sample_id}")
        if vision_indices is not None and not np.array_equal(word_indices, vision_indices):
            raise ValueError(f"word indices do not agree for {sample_id}")
        if audio_indices is None:
            audio = audio[word_indices]
        if vision_indices is None:
            vision = vision[word_indices]
        pooled = build_aligned_sample(record, text, word_indices, audio, vision, positions)
        samples.append({"id": sample_id, "raw_text": record["raw_text"], "text": pooled["text"], "audio": pooled["audio"], "vision": pooled["vision"], "lengths": pooled["lengths"], "label": float(manifest[sample_id]["label"]), "annotation": manifest[sample_id]["annotation"]})
        record["pooled_position_map"] = pooled["position_map"]
        record["pooled_length"] = pooled["lengths"][0]
        sidecars.append(record)

    train = {
        "id": [sample["id"] for sample in samples],
        "raw_text": [sample["raw_text"] for sample in samples],
        "text": np.stack([sample["text"] for sample in samples]),
        "audio": np.stack([sample["audio"] for sample in samples]),
        "vision": np.stack([sample["vision"] for sample in samples]),
        "text_lengths": np.asarray([sample["lengths"][0] for sample in samples], dtype=np.int64),
        "audio_lengths": np.asarray([sample["lengths"][1] for sample in samples], dtype=np.int64),
        "vision_lengths": np.asarray([sample["lengths"][2] for sample in samples], dtype=np.int64),
        "annotations": [sample["annotation"] for sample in samples],
        "classification_labels": [sample["annotation"] for sample in samples],
        "regression_labels": np.asarray([sample["label"] for sample in samples], dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump({"train": train}, file, protocol=pickle.HIGHEST_PROTOCOL)
    alignment_path.write_text("".join(json.dumps(record) + "\n" for record in sidecars), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["paths"]["output"])
    build_output(output / "word_alignment.jsonl", output / "manifest.csv", output, output / "aligned_50.pkl", int(config["alignment"]["max_positions"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

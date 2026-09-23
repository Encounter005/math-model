"""Validate the Question 1 fixed-width multimodal output contract."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

EXPECTED_SHAPES = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35)}


def _records(path: Path) -> dict[str, dict]:
    return {record["id"]: record for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def validate_output(output_path: Path, alignment_path: Path, manifest_path: Path) -> None:
    """Raise ValueError when aligned output violates its serialized contract."""
    with output_path.open("rb") as file:
        train = pickle.load(file)["train"]
    ids = train["id"]
    manifest = {row["id"]: row for row in csv.DictReader(manifest_path.open(newline="", encoding="utf-8"))}
    alignment = _records(alignment_path)
    if len(ids) != len(set(ids)) or set(ids) != set(manifest) or set(ids) != set(alignment):
        raise ValueError("IDs must be unique and covered by manifest and alignment sidecars")
    count = len(ids)
    for modality, expected in EXPECTED_SHAPES.items():
        values = np.asarray(train[modality])
        lengths = np.asarray(train[f"{modality}_lengths"], dtype=np.int64)
        if values.shape != (count, *expected):
            raise ValueError(f"{modality} shape must be {(count, *expected)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{modality} contains non-finite values")
        if len(lengths) != count or np.any((lengths < 0) | (lengths > expected[0])):
            raise ValueError(f"{modality} lengths are invalid")
        if any(np.any(values[index, length:]) for index, length in enumerate(lengths)):
            raise ValueError(f"{modality} padding must be zero")
    if not np.array_equal(train["text_lengths"], train["audio_lengths"]) or not np.array_equal(train["text_lengths"], train["vision_lengths"]):
        raise ValueError("modality lengths must share the text axis")
    for index, sample_id in enumerate(ids):
        if float(train["regression_labels"][index]) != float(manifest[sample_id]["label"]):
            raise ValueError(f"regression label mismatch for {sample_id}")
        if train["annotations"][index] != manifest[sample_id]["annotation"] or train["classification_labels"][index] != manifest[sample_id]["annotation"]:
            raise ValueError(f"classification label mismatch for {sample_id}")
        if int(alignment[sample_id].get("pooled_length", -1)) != int(train["text_lengths"][index]):
            raise ValueError(f"pooled length mismatch for {sample_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--pipeline-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = Path(config["paths"]["output"])
    root = args.pipeline_root or base / "pipeline_v2"
    validate_output(root / "aligned_50.pkl", root / "word_alignment.jsonl", base / "manifest.csv")
    print(f"validated {root / 'aligned_50.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build Task 12 per-sample feature evidence from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def _records(path: Path) -> dict[str, dict]:
    return {record["id"]: record for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def build_summary(pipeline_root: Path, manifest_path: Path, output_path: Path) -> None:
    """Write one evidence row per manifest sample."""
    manifest = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    alignment = _records(pipeline_root / "word_alignment.jsonl")
    status = _records(pipeline_root / "logs" / "status.jsonl")
    fields = ("id", "source_duration_s", "effective_length", "text_dimensions", "audio_dimensions", "vision_dimensions", "valid_word_count", "pooled_word_count", "face_detection_rate", "warnings")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in manifest:
            sample_id = sample["id"]
            record = alignment[sample_id]
            vision = np.load(pipeline_root / "vision" / f"{sample_id}.npz", allow_pickle=True)
            valid_words = sum(word.get("start") is not None and word.get("end") is not None for word in record["words"])
            pooled_length = int(record.get("pooled_length", 0))
            writer.writerow({
                "id": sample_id,
                "source_duration_s": record["duration_s"],
                "effective_length": pooled_length,
                "text_dimensions": 768,
                "audio_dimensions": 74,
                "vision_dimensions": 35,
                "valid_word_count": valid_words,
                "pooled_word_count": max(0, valid_words - pooled_length),
                "face_detection_rate": float(vision["face_detection_rate"]),
                "warnings": "; ".join(status.get(sample_id, {}).get("warnings", [])),
            })


def write_metadata(config: dict, pipeline_root: Path, output_path: Path, prior_metadata: Path) -> None:
    """Record versions, inputs, and the configuration fingerprint for this batch."""
    status = _records(pipeline_root / "logs" / "status.jsonl")
    existing = json.loads(prior_metadata.read_text(encoding="utf-8")) if prior_metadata.exists() else {}
    existing.update({
        "python": sys.version,
        "models": config["models"],
        "packages": {name: __import__(name).__version__ for name in ("numpy", "pyworld", "torch", "transformers", "yaml")},
        "pipeline_root": str(pipeline_root),
        "config_hashes": sorted({record["config_hash"] for record in status.values()}),
        "commands": [
            "uv run python scripts/question_1/run_pipeline.py --config configs/question_1.yaml",
            "uv run python scripts/question_1/validate_output.py --config configs/question_1.yaml",
            "uv run python scripts/question_1/build_summary.py --config configs/question_1.yaml",
        ],
        "source_hashes": {sample_id: record["source_sha256"] for sample_id, record in status.items()},
    })
    output_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = Path(config["paths"]["output"])
    root = args.pipeline_root or base / "pipeline_v2"
    build_summary(root, base / "manifest.csv", args.output or base / "feature_summary.csv")
    write_metadata(config, root, base / "run_metadata.json", base / "run_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

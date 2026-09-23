"""Extract OpenFace features and aggregate them over aligned word intervals."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml


def _number(row: dict[str, str], name: str) -> float:
    try:
        return float(row.get(name, row.get(f" {name}", "nan")))
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _distance(row: dict[str, str], left: int, right: int) -> float:
    return float(
        np.hypot(
            _number(row, f"x_{left}") - _number(row, f"x_{right}"),
            _number(row, f"y_{left}") - _number(row, f"y_{right}"),
        )
    )


def frame_features(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Map OpenFace rows to 35 features and mark rows with usable detections."""
    columns = [
        "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r",
        "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r",
        "AU45_r", "pose_Rx", "pose_Ry", "pose_Rz", "pose_Tx", "pose_Ty", "pose_Tz",
        "gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z",
    ]
    values = np.zeros((len(rows), 35), dtype=np.float32)
    usable = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        values[index, :29] = [_number(row, column) for column in columns]
        values[index, 29:] = [
            _distance(row, 36, 45),
            _distance(row, 27, 8),
            _distance(row, 39, 42),
            _distance(row, 48, 54),
            _distance(row, 51, 57),
            _distance(row, 27, 51),
        ]
        usable[index] = _number(row, "success") > 0 and _number(row, "confidence") > 0
    usable &= np.isfinite(values).all(axis=1)
    values[~usable] = 0.0
    return values, usable


def aggregate_interval_features(
    frame_values: np.ndarray,
    frame_spans: np.ndarray,
    usable: np.ndarray,
    intervals: list[tuple[float | None, float | None]],
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    pooled = np.zeros((len(intervals), frame_values.shape[1]), dtype=np.float32)
    ranges = np.zeros((len(intervals), 2), dtype=np.int64)
    reasons: list[str | None] = [None] * len(intervals)
    for index, (start, end) in enumerate(intervals):
        if start is None or end is None or end <= start:
            reasons[index] = "missing_or_invalid_interval"
            continue
        overlap = np.minimum(frame_spans[:, 1], end) - np.maximum(frame_spans[:, 0], start)
        frame_indices = np.flatnonzero((overlap > 0) & usable)
        if not len(frame_indices):
            reasons[index] = "no_usable_face_frame"
            continue
        weights = overlap[frame_indices].astype(np.float32)
        pooled[index] = np.average(frame_values[frame_indices], axis=0, weights=weights)
        ranges[index] = [int(frame_indices[0]), int(frame_indices[-1] + 1)]
    return pooled, ranges, reasons


def run_openface(executable: str, video_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [executable, "-f", str(video_path), "-out_dir", str(output_dir), "-aus", "-pose", "-gaze", "-2Dfp"],
        capture_output=True,
        text=True,
        check=True,
    )
    del result
    output_path = output_dir / f"{video_path.stem}.csv"
    if not output_path.exists():
        raise FileNotFoundError(f"OpenFace did not produce {output_path}")
    return output_path


def read_openface_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
    values, usable = frame_features(rows)
    timestamps = np.asarray([_number(row, "timestamp") for row in rows], dtype=np.float32)
    if len(timestamps) > 1:
        step = float(np.median(np.diff(timestamps)))
    else:
        step = 1 / 30
    spans = np.column_stack((timestamps, timestamps + step)).astype(np.float32)
    return values, spans, usable, float(usable.mean()) if len(usable) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--feature-config", type=Path, default=Path("configs/openface_35.json"))
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_config = json.loads(args.feature_config.read_text(encoding="utf-8"))
    if sum(map(len, (feature_config["action_units"], feature_config["head_pose"], feature_config["gaze"], feature_config["geometry"]))) != 35:
        raise ValueError("OpenFace feature configuration must define exactly 35 features")
    output_root = Path(config["paths"]["output"])
    alignment_path = args.alignment or output_root / "word_alignment.jsonl"
    output_dir = args.output or output_root / "vision"
    output_dir.mkdir(parents=True, exist_ok=True)
    with alignment_path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            with tempfile.TemporaryDirectory(prefix=f"openface_{record['id']}_") as temp_dir:
                csv_path = run_openface(config["tools"]["openface"], Path(record["source_mp4"]), Path(temp_dir))
                frame_values, spans, usable, face_rate = read_openface_csv(csv_path)
            intervals = [(word.get("start"), word.get("end")) for word in record["words"]]
            features, source_ranges, reasons = aggregate_interval_features(frame_values, spans, usable, intervals)
            np.savez_compressed(
                output_dir / f"{record['id']}.npz",
                features=features,
                source_ranges=source_ranges,
                reasons=np.asarray(reasons, dtype=object),
                face_detection_rate=np.asarray(face_rate, dtype=np.float32),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

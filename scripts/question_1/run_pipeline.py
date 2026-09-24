"""Run the Question 1 extraction pipeline independently for each sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.question_1.align_words import (
    align_manual_words,
    load_asr,
    normalized_words,
    transcribe_words,
)
from scripts.question_1.build_aligned_50 import build_output
from scripts.question_1.extract_audio import (
    aggregate_interval_features,
    audio_frame_features,
    read_wav,
)
from scripts.question_1.extract_media import extract_wav, probe_duration, probe_media
from scripts.question_1.extract_text import embed_aligned_words, load_text_model
from scripts.question_1.extract_vision import (
    aggregate_interval_features as aggregate_vision,
)
from scripts.question_1.extract_vision import read_openface_csv, run_openface


def config_hash(config: dict, vision_features: dict | None = None) -> str:
    relevant = {key: config[key] for key in ("models", "audio", "alignment")}
    relevant["vision_features"] = vision_features or {}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {record["id"]: record for record in (json.loads(line) for line in path.read_text().splitlines())}


def process_samples(
    manifest: list[dict[str, str]],
    previous: dict[str, dict],
    source_hash,
    extractor_hash: str,
    run_one,
    log_dir: Path,
) -> list[dict]:
    """Run samples in ID order, resume matching completions, and retain failures."""
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.jsonl"
    prior = {
        record.get("id", key): record
        for key, record in previous.items()
    }
    records = []
    with status_path.open("w", encoding="utf-8") as status_file:
        for sample in sorted(manifest, key=lambda item: item["id"]):
            sample_id = sample["id"]
            started = time.monotonic()
            digest = source_hash(sample["video_path"])
            old = prior.get(sample_id, {})
            if old.get("status") == "complete" and old.get("source_sha256") == digest and old.get("config_hash") == extractor_hash:
                record = {**old, "status": "skipped", "elapsed_s": 0.0, "warnings": old.get("warnings", [])}
            else:
                record = {
                    "id": sample_id,
                    "source_sha256": digest,
                    "config_hash": extractor_hash,
                    "status": "complete",
                    "warnings": [],
                }
                try:
                    record["warnings"] = run_one(sample)
                # Continue independent samples after any ordinary processing failure.
                except Exception as error:  # noqa: BLE001
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                    record["traceback"] = traceback.format_exc()
            record["id"] = sample_id
            if record["status"] != "skipped":
                record["elapsed_s"] = time.monotonic() - started
            records.append(record)
            status_file.write(json.dumps(record) + "\n")
            status_file.flush()
    return records


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {record["id"]: record for record in (json.loads(line) for line in path.read_text().splitlines())}


def _write_jsonl(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(records[key]) + "\n" for key in sorted(records)), encoding="utf-8")


def seed_completed_cache(
    manifest: list[dict[str, str]],
    source_hash,
    config_digest: str,
    input_root: Path,
    output_root: Path,
    state_path: Path,
) -> None:
    """Mark complete existing feature bundles reusable in the isolated output root."""
    state = load_state(state_path)
    alignment = _read_jsonl(input_root / "word_alignment.jsonl")
    for sample in manifest:
        sample_id = sample["id"]
        if sample_id in state:
            continue
        files = [input_root / name / f"{sample_id}.npz" for name in ("text", "audio", "vision")]
        if sample_id in alignment and all(path.is_file() for path in files):
            existing_alignment = _read_jsonl(output_root / "word_alignment.jsonl")
            existing_alignment[sample_id] = alignment[sample_id]
            _write_jsonl(output_root / "word_alignment.jsonl", existing_alignment)
            for modality, source_file in zip(("text", "audio", "vision"), files, strict=True):
                destination = output_root / modality / source_file.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
            state[sample_id] = {
                "id": sample_id,
                "source_sha256": source_hash(sample["video_path"]),
                "config_hash": config_digest,
                "status": "complete",
                "elapsed_s": 0.0,
                "warnings": ["reused verified existing feature bundle"],
            }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("".join(json.dumps(state[key]) + "\n" for key in sorted(state)), encoding="utf-8")


def run_sample(sample: dict[str, str], config: dict, output: Path, models: dict) -> list[str]:
    sample_id = sample["id"]
    warnings = []
    source = Path(sample["video_path"])
    wav_path = output / "wav" / f"{sample_id}.wav"
    media = probe_media(source, ffprobe=config["tools"]["ffprobe"])
    extract_wav(source, wav_path, ffmpeg=config["tools"]["ffmpeg"])
    duration = probe_duration(wav_path, ffprobe=config["tools"]["ffprobe"])
    if abs(media["source_duration_s"] - duration) > 1 / float(media["fps"]):
        warnings.append("WAV duration differs from video by more than one frame")
    media_records = _read_jsonl(output / "media_metadata.jsonl")
    media_records[sample_id] = {"id": sample_id, "source_mp4": str(source), "wav_path": str(wav_path), **media, "wav_duration_s": duration}
    _write_jsonl(output / "media_metadata.jsonl", media_records)

    asr_words = transcribe_words(models["asr"], wav_path)
    words = align_manual_words(
        normalized_words(sample["text"]),
        asr_words,
        duration=duration,
        gap_penalty=float(config["alignment"]["dtw_gap_penalty"]),
        acceptance_threshold=float(config["alignment"]["dtw_acceptance_threshold"]),
    )
    alignment = {
        "id": sample_id,
        "source_mp4": str(source),
        "wav_path": str(wav_path),
        "raw_text": sample["text"],
        "duration_s": duration,
        "asr_model": config["models"]["asr"],
        "asr_words": asr_words,
        "words": words,
        "unmatched_words": [word for word in words if "unmatched_reason" in word],
        "processing_status": "complete",
    }
    alignment_records = _read_jsonl(output / "word_alignment.jsonl")
    alignment_records[sample_id] = alignment
    _write_jsonl(output / "word_alignment.jsonl", alignment_records)

    text_features, word_indices, token_spans = embed_aligned_words(words, *models["text"])
    (output / "text").mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(output / "text" / f"{sample_id}.npz", features=text_features, word_indices=word_indices, token_spans=token_spans)

    samples, sample_rate = read_wav(wav_path)
    audio_config = config["audio"]
    frames, spans = audio_frame_features(
        samples,
        sample_rate,
        float(audio_config["frame_period_ms"]),
        int(audio_config["spectral_mel_bands"]),
        int(audio_config["aperiodicity_mel_bands"]),
    )
    intervals = [(word.get("start"), word.get("end")) for word in words]
    audio_values, audio_ranges, audio_reasons = aggregate_interval_features(frames, spans, intervals)
    (output / "audio").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "audio" / f"{sample_id}.npz", features=audio_values, source_ranges=audio_ranges, reasons=np.asarray(audio_reasons, dtype=object))

    with tempfile.TemporaryDirectory(prefix=f"openface_{sample_id}_") as temp_dir:
        csv_path = run_openface(config["tools"]["openface"], source, Path(temp_dir))
        face_values, face_spans, usable, face_rate = read_openface_csv(csv_path)
    vision_values, vision_ranges, vision_reasons = aggregate_vision(face_values, face_spans, usable, intervals)
    (output / "vision").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "vision" / f"{sample_id}.npz",
        features=vision_values,
        source_ranges=vision_ranges,
        reasons=np.asarray(vision_reasons, dtype=object),
        face_detection_rate=np.asarray(face_rate, dtype=np.float32),
    )
    if face_rate == 0:
        warnings.append("no usable face detections")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    vision_features = json.loads(Path("configs/openface_35.json").read_text(encoding="utf-8"))
    root = args.output or Path(config["paths"]["output"]) / "pipeline_v2"
    root.mkdir(parents=True, exist_ok=True)
    with (Path(config["paths"]["output"]) / "manifest.csv").open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    media = _read_jsonl(Path(config["paths"]["output"]) / "media_metadata.jsonl")
    for sample in manifest:
        if sample["id"] in media:
            sample["source_sha256"] = media[sample["id"]]["source_sha256"]
    models = {"asr": load_asr(config["models"]["asr"], Path(config["paths"]["model_cache"])), "text": load_text_model(config["models"]["text"], Path(config["paths"]["model_cache"]))}
    digest = lambda value: hashlib.sha256(Path(value).read_bytes()).hexdigest()
    config_digest = config_hash(config, vision_features)
    seed_completed_cache(
        manifest,
        digest,
        config_digest,
        Path(config["paths"]["output"]),
        root,
        root / "logs" / "status.jsonl",
    )
    records = process_samples(
        manifest,
        load_state(root / "logs" / "status.jsonl"),
        digest,
        config_digest,
        lambda sample: run_sample(sample, {**config, "vision_features": vision_features}, root, models),
        root / "logs",
    )
    final_alignment = _read_jsonl(root / "word_alignment.jsonl")
    if all(record["status"] in {"complete", "skipped"} for record in records) and len(final_alignment) == len(manifest):
        build_output(root / "word_alignment.jsonl", Path(config["paths"]["output"]) / "manifest.csv", root, root / "aligned_50.pkl", int(config["alignment"]["max_positions"]))
    failed = [record["id"] for record in records if record["status"] == "failed"]
    print(f"processed={len(records)} failed={len(failed)}")
    if failed:
        print("failed IDs: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

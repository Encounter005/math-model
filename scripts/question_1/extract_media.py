"""Extract 16 kHz mono WAV files and record source-media metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml

Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Run an external media command with text output captured."""
    return subprocess.run(command, capture_output=True, check=True, text=True, **kwargs)


def source_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_data(path: Path, runner: Runner = run_command, ffprobe: str = "ffprobe") -> dict:
    """Return the format and stream information emitted by FFprobe."""
    result = runner(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate,width,height,sample_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def frame_rate(value: str) -> float:
    """Convert FFprobe's fractional frame-rate representation to a float."""
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def probe_media(path: Path, runner: Runner = run_command, ffprobe: str = "ffprobe") -> dict[str, float | int | str]:
    """Collect the media metadata required for a source MP4."""
    data = ffprobe_data(path, runner, ffprobe)
    streams = data["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    return {
        "source_duration_s": float(data["format"]["duration"]),
        "fps": frame_rate(video["avg_frame_rate"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "audio_sample_rate": int(audio["sample_rate"]),
        "source_sha256": source_sha256(path),
    }


def probe_duration(path: Path, runner: Runner = run_command, ffprobe: str = "ffprobe") -> float:
    """Return the duration in seconds reported by FFprobe."""
    return float(ffprobe_data(path, runner, ffprobe)["format"]["duration"])


def extract_wav(
    source: Path,
    destination: Path,
    runner: Runner = run_command,
    ffmpeg: str = "ffmpeg",
) -> None:
    """Extract source audio as 16 kHz mono signed-16-bit PCM WAV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    runner(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def process_manifest(
    manifest_path: Path,
    wav_root: Path,
    metadata_path: Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> None:
    """Extract all manifest media and write one metadata record per sample."""
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        for row in rows:
            source = Path(row["video_path"])
            wav_path = wav_root / f"{row['id']}.wav"
            metadata = probe_media(source, ffprobe=ffprobe)
            extract_wav(source, wav_path, ffmpeg=ffmpeg)
            wav_duration = probe_duration(wav_path, ffprobe=ffprobe)
            duration_delta = abs(metadata["source_duration_s"] - wav_duration)
            tolerance = 1 / metadata["fps"]
            record = {
                "id": row["id"],
                "source_mp4": str(source),
                "wav_path": str(wav_path),
                **metadata,
                "wav_duration_s": wav_duration,
                "duration_delta_s": duration_delta,
                "duration_tolerance_s": tolerance,
            }
            if duration_delta > tolerance:
                record["duration_exception"] = "WAV duration exceeds one video-frame interval"
            file.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--wav-root", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    output = Path(paths["output"])
    process_manifest(
        args.manifest or output / "manifest.csv",
        args.wav_root or output / "wav",
        args.metadata or output / "media_metadata.jsonl",
        config["tools"]["ffmpeg"],
        config["tools"]["ffprobe"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

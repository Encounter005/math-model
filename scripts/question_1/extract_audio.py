"""Extract and aggregate PyWorld audio features."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import pyworld as pw
import yaml


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as file:
        if file.getnchannels() != 1 or file.getsampwidth() != 2:
            raise ValueError(f"expected mono 16-bit WAV: {path}")
        return np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(np.float64) / 32768.0, file.getframerate()


def mel_filter_bank(sample_rate: int, n_fft: int, bands: int) -> np.ndarray:
    low_hz, high_hz = 0.0, sample_rate / 2.0
    low_mel = 2595.0 * np.log10(1.0 + low_hz / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + high_hz / 700.0)
    mels = np.linspace(low_mel, high_mel, bands + 2)
    hz = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    filters = np.zeros((bands, n_fft // 2 + 1), dtype=np.float32)
    for band in range(bands):
        left, center, right = bins[band : band + 3]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for index in range(left, center):
            filters[band, index] = (index - left) / max(center - left, 1)
        for index in range(center, right):
            filters[band, index] = (right - index) / max(right - center, 1)
    return filters


def frame_spans(frame_count: int, sample_rate: int, frame_period_ms: float) -> np.ndarray:
    step = frame_period_ms / 1000.0
    starts = np.arange(frame_count, dtype=np.float32) * step
    ends = starts + step
    return np.column_stack((starts, ends))


def audio_frame_features(
    samples: np.ndarray,
    sample_rate: int,
    frame_period_ms: float,
    spectral_mel_bands: int,
    aperiodicity_mel_bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    f0, time_axis = pw.dio(samples.astype(np.float64), sample_rate, frame_period=frame_period_ms)
    f0 = pw.stonemask(samples.astype(np.float64), f0, time_axis, sample_rate)
    spectrogram = pw.cheaptrick(samples.astype(np.float64), f0, time_axis, sample_rate)
    aperiodicity = pw.d4c(samples.astype(np.float64), f0, time_axis, sample_rate)
    if not len(time_axis):
        return np.empty((0, 74), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    n_fft = (spectrogram.shape[1] - 1) * 2
    mel_filters = mel_filter_bank(sample_rate, n_fft, spectral_mel_bands)
    ap_filters = mel_filter_bank(sample_rate, n_fft, aperiodicity_mel_bands)
    energy = np.sum(spectrogram, axis=1, dtype=np.float64)
    log_f0 = np.log(np.clip(f0, 1e-8, None))
    voiced = (f0 > 0).astype(np.float32)
    spectral = spectrogram @ mel_filters.T
    aperiodic = aperiodicity @ ap_filters.T
    features = np.column_stack((log_f0, voiced, energy, spectral, aperiodic)).astype(np.float32)
    return features, frame_spans(len(time_axis), sample_rate, frame_period_ms)


def aggregate_interval_features(
    frame_features: np.ndarray,
    frame_spans: np.ndarray,
    intervals: list[tuple[float | None, float | None]],
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    pooled = np.zeros((len(intervals), frame_features.shape[1]), dtype=np.float32)
    source_ranges = np.zeros((len(intervals), 2), dtype=np.int64)
    reasons: list[str | None] = [None] * len(intervals)
    for index, (start, end) in enumerate(intervals):
        if start is None or end is None or end <= start:
            reasons[index] = "missing_or_invalid_interval"
            continue
        overlaps = np.minimum(frame_spans[:, 1], end) - np.maximum(frame_spans[:, 0], start)
        frame_indices = np.flatnonzero(overlaps > 0)
        if not len(frame_indices):
            reasons[index] = "missing_or_invalid_interval"
            continue
        weights = overlaps[frame_indices].astype(np.float32)
        pooled[index] = np.average(frame_features[frame_indices], axis=0, weights=weights)
        source_ranges[index] = [int(frame_indices[0]), int(frame_indices[-1] + 1)]
    return pooled, source_ranges, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["paths"]["output"])
    alignment_path = args.alignment or output_root / "word_alignment.jsonl"
    output_dir = args.output or output_root / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_cfg = config["audio"]
    with alignment_path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            samples, sample_rate = read_wav(Path(record["wav_path"]))
            frame_features, spans = audio_frame_features(
                samples,
                sample_rate,
                float(audio_cfg["frame_period_ms"]),
                int(audio_cfg["spectral_mel_bands"]),
                int(audio_cfg["aperiodicity_mel_bands"]),
            )
            intervals = [(word.get("start"), word.get("end")) for word in record["words"]]
            features, source_ranges, reasons = aggregate_interval_features(frame_features, spans, intervals)
            np.savez_compressed(
                output_dir / f"{record['id']}.npz",
                features=features,
                source_ranges=source_ranges,
                reasons=np.asarray(reasons, dtype=object),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

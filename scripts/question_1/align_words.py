"""Align supplied transcript words to cached Whisper timestamps with DTW."""

from __future__ import annotations

import argparse
import csv
import json
import re
import wave
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def normalized_words(text: str) -> list[dict[str, str | int]]:
    """Return lowercase tokens while preserving supplied-transcript positions."""
    return [
        {"index": index, "original": match.group(), "normalized": match.group().replace("’", "'").lower()}
        for index, match in enumerate(WORD_PATTERN.finditer(text))
    ]


def word_distance(left: str, right: str) -> float:
    """Return normalized Levenshtein distance for two non-empty word strings."""
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def dtw_path(manual: list[str], asr: list[str], gap_penalty: float) -> list[tuple[int, int]]:
    """Return the monotonic DTW path over manual and ASR word sequences."""
    if not manual or not asr:
        return []
    costs = np.full((len(manual) + 1, len(asr) + 1), np.inf)
    costs[0, 0] = 0.0
    costs[1:, 0] = np.arange(1, len(manual) + 1) * gap_penalty
    costs[0, 1:] = np.arange(1, len(asr) + 1) * gap_penalty
    moves = np.zeros((len(manual) + 1, len(asr) + 1), dtype=np.int8)
    for manual_index, manual_word in enumerate(manual, start=1):
        for asr_index, asr_word in enumerate(asr, start=1):
            choices = (
                costs[manual_index - 1, asr_index - 1],
                costs[manual_index - 1, asr_index] + gap_penalty,
                costs[manual_index, asr_index - 1] + gap_penalty,
            )
            move = min(range(3), key=choices.__getitem__)
            costs[manual_index, asr_index] = word_distance(manual_word, asr_word) + choices[move]
            moves[manual_index, asr_index] = move
    path: list[tuple[int, int]] = []
    manual_index, asr_index = len(manual), len(asr)
    while manual_index and asr_index:
        path.append((manual_index - 1, asr_index - 1))
        move = moves[manual_index, asr_index]
        if move == 0:
            manual_index -= 1
            asr_index -= 1
        elif move == 1:
            manual_index -= 1
        else:
            asr_index -= 1
    while manual_index:
        manual_index -= 1
        path.append((manual_index, 0))
    while asr_index:
        asr_index -= 1
        path.append((0, asr_index))
    return list(reversed(path))


def align_manual_words(
    manual_words: list[dict[str, str | int]],
    asr_words: list[dict[str, str | float]],
    *,
    duration: float,
    gap_penalty: float,
    acceptance_threshold: float,
) -> list[dict]:
    """Transfer accepted ASR spans to supplied manual words using a DTW path."""
    path = dtw_path(
        [str(word["normalized"]) for word in manual_words],
        [str(word["word"]).replace("’", "'").lower() for word in asr_words],
        gap_penalty,
    )
    matches: dict[int, list[int]] = {index: [] for index in range(len(manual_words))}
    for manual_index, asr_index in path:
        if word_distance(str(manual_words[manual_index]["normalized"]), str(asr_words[asr_index]["word"])) <= acceptance_threshold:
            matches[manual_index].append(asr_index)

    records: list[dict] = []
    for manual_index, manual_word in enumerate(manual_words):
        indices = matches[manual_index]
        if not indices:
            records.append({**manual_word, "start": None, "end": None, "unmatched_reason": "dtw_cost_exceeds_threshold"})
            continue
        start = min(float(asr_words[index]["start"]) for index in indices)
        end = max(float(asr_words[index]["end"]) for index in indices)
        if not 0 <= start < end <= duration:
            records.append({**manual_word, "start": None, "end": None, "unmatched_reason": "invalid_asr_timestamp"})
            continue
        records.append(
            {
                **manual_word,
                "start": start,
                "end": end,
                "asr_word_indices": indices,
                "dtw_cost": sum(
                    word_distance(str(manual_word["normalized"]), str(asr_words[index]["word"]))
                    for index in indices
                )
                / len(indices),
            }
        )
    return records


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono 16-bit PCM WAV without an additional audio dependency."""
    with wave.open(str(path), "rb") as file:
        if file.getnchannels() != 1 or file.getsampwidth() != 2:
            raise ValueError(f"expected mono 16-bit WAV: {path}")
        return np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(np.float32) / 32768, file.getframerate()


def load_asr(model_name: str, cache_dir: Path):
    """Load Whisper from the project-local cache without network access."""
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, cache_dir=str(cache_dir), local_files_only=True
    )
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
    )


def transcribe_words(asr, wav_path: Path) -> list[dict[str, str | float]]:
    """Return Whisper word chunks with finite timestamps for one local WAV."""
    samples, sample_rate = read_wav(wav_path)
    result = asr(
        {"raw": samples, "sampling_rate": sample_rate},
        return_timestamps="word",
    )
    words = []
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        token = str(chunk["text"]).strip()
        if token and start is not None and end is not None:
            words.append({"word": token, "start": float(start), "end": float(end)})
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    output = Path(paths["output"])
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    metadata = {
        record["id"]: record
        for record in (json.loads(line) for line in (output / "media_metadata.jsonl").open(encoding="utf-8"))
    }
    asr = load_asr(config["models"]["asr"], Path(paths["model_cache"]))
    alignment = config["alignment"]
    with (output / "word_alignment.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            sample_id = row["id"]
            manual_words = normalized_words(row["text"])
            duration = float(metadata[sample_id]["wav_duration_s"])
            try:
                asr_words = transcribe_words(asr, Path(metadata[sample_id]["wav_path"]))
                word_records = align_manual_words(
                    manual_words,
                    asr_words,
                    duration=duration,
                    gap_penalty=float(alignment["dtw_gap_penalty"]),
                    acceptance_threshold=float(alignment["dtw_acceptance_threshold"]),
                )
                status = "complete"
            except (OSError, RuntimeError, ValueError):
                asr_words = []
                word_records = [
                    {**word, "start": None, "end": None, "unmatched_reason": "asr_error"} for word in manual_words
                ]
                status = "error"
            unmatched = [word for word in word_records if "unmatched_reason" in word]
            file.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "source_mp4": row["video_path"],
                        "wav_path": metadata[sample_id]["wav_path"],
                        "raw_text": row["text"],
                        "duration_s": duration,
                        "asr_model": config["models"]["asr"],
                        "asr_words": asr_words,
                        "words": word_records,
                        "unmatched_words": unmatched,
                        "processing_status": status,
                    }
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

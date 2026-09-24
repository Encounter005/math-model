"""Extract DistilBERT features for DTW-aligned transcript words."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer


def load_text_model(model_name: str, cache_dir: Path):
    """Load the text extractor from the local cache without network access."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    return tokenizer, model.eval()


def embed_aligned_words(words: list[dict], tokenizer, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool WordPiece vectors for manual words with valid DTW intervals."""
    aligned = [word for word in words if word.get("start") is not None and word.get("end") is not None]
    if not aligned:
        return (
            np.empty((0, 768), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
        )
    encoded = tokenizer(
        [str(word["original"]) for word in aligned],
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
    )
    word_ids = encoded.word_ids(batch_index=0)
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state[0]
    if hidden.shape[-1] != 768:
        raise ValueError(f"expected 768 text features, got {hidden.shape[-1]}")

    vectors: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    for word_position, word in enumerate(aligned):
        token_positions = [index for index, token_word in enumerate(word_ids) if token_word == word_position]
        if not token_positions:
            raise ValueError(f"aligned word was truncated by tokenizer: {word['index']}")
        vectors.append(hidden[token_positions].mean(dim=0).cpu().numpy().astype(np.float32))
        spans.append((token_positions[0], token_positions[-1] + 1))
    return (
        np.stack(vectors),
        np.asarray([word["index"] for word in aligned], dtype=np.int64),
        np.asarray(spans, dtype=np.int64),
    )


def extract_features(alignment_path: Path, output_dir: Path, tokenizer, model) -> None:
    """Write one word-feature archive per ASR-DTW alignment record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for line in alignment_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        features, word_indices, token_spans = embed_aligned_words(record["words"], tokenizer, model)
        np.savez_compressed(
            output_dir / f"{record['id']}.npz",
            features=features,
            word_indices=word_indices,
            token_spans=token_spans,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["paths"]["output"])
    tokenizer, model = load_text_model(config["models"]["text"], Path(config["paths"]["model_cache"]))
    extract_features(args.alignment or output_root / "word_alignment.jsonl", args.output or output_root / "text", tokenizer, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create compact evidence for one short and one long aligned sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


def _valid_words(record: dict) -> list[dict]:
    return [word for word in record["words"] if word.get("start") is not None and word.get("end") is not None]


def choose_samples(records: list[dict]) -> dict[str, str]:
    """Choose the shortest valid sample and the longest sample that was pooled."""
    valid = [record for record in records if _valid_words(record)]
    if not valid:
        raise ValueError("alignment contains no valid words")
    short = min(valid, key=lambda record: (len(_valid_words(record)), record["id"]))
    pooled = [record for record in valid if len(_valid_words(record)) > 50 and record.get("pooled_length", 0) == 50]
    long = max(pooled or valid, key=lambda record: (len(_valid_words(record)), record["id"]))
    return {"short": short["id"], "long": long["id"]}


def _range_text(value: np.ndarray) -> str:
    return f"{int(value[0])}:{int(value[1])}" if value[1] > value[0] else ""


def collect_word_rows(record: dict, feature_root: Path) -> list[dict[str, object]]:
    """Join manual words with ASR indices, source-frame ranges, and pooled positions."""
    asr_words = record.get("asr_words", [])
    ranges = {}
    for modality in ("audio", "vision"):
        archive = np.load(feature_root / modality / f"{record['id']}.npz", allow_pickle=True)
        ranges[modality] = np.asarray(archive["source_ranges"])
    position_map = record.get("pooled_position_map", {})
    rows = []
    for index, word in enumerate(record["words"]):
        asr_indices = word.get("asr_word_indices", [])
        rows.append(
            {
                "word_index": word["index"],
                "word": word.get("original", ""),
                "asr_words": " ".join(str(asr_words[i]["word"]) for i in asr_indices if i < len(asr_words)),
                "dtw_cost": word.get("dtw_cost", ""),
                "start_s": word.get("start", ""),
                "end_s": word.get("end", ""),
                "audio_frame_range": _range_text(ranges["audio"][index]),
                "vision_frame_range": _range_text(ranges["vision"][index]),
                "pooled_position": position_map.get(str(word["index"]), ""),
                "unmatched_reason": word.get("unmatched_reason", ""),
            }
        )
    return rows


def _timeline_svg(record: dict, rows: list[dict[str, object]], path: Path) -> None:
    duration = max(float(record.get("duration_s", 0.0)), 0.001)
    width, height = 900, 90
    segments = []
    for row in rows:
        if row["start_s"] is None or row["end_s"] is None or row["start_s"] == "":
            continue
        x = 20 + 860 * float(row["start_s"]) / duration
        w = max(2.0, 860 * (float(row["end_s"]) - float(row["start_s"])) / duration)
        color = "#176b87" if row["pooled_position"] != "" else "#b7c4c8"
        segments.append(f'<rect x="{x:.1f}" y="30" width="{w:.1f}" height="24" fill="{color}"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img"><title>{record["id"]} aligned word timeline</title><rect width="100%" height="100%" fill="#f7f4ed"/>{"".join(segments)}<text x="20" y="20" font-family="sans-serif" font-size="12">{record["id"]} | {len(_valid_words(record))} valid words | blue = pooled position</text><line x1="20" y1="58" x2="880" y2="58" stroke="#243238"/><text x="20" y="76" font-family="sans-serif" font-size="11">0s</text><text x="840" y="76" font-family="sans-serif" font-size="11">{duration:.2f}s</text></svg>'
    path.write_text(svg, encoding="utf-8")


def inspect_samples(alignment_path: Path, feature_root: Path, output_dir: Path) -> None:
    records = [json.loads(line) for line in alignment_path.read_text(encoding="utf-8").splitlines()]
    selected = choose_samples(records)
    by_id = {record["id"]: record for record in records}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for kind, sample_id in selected.items():
        record = by_id[sample_id]
        rows = collect_word_rows(record, feature_root)
        csv_path = output_dir / f"{kind}_{sample_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _timeline_svg(record, rows, output_dir / f"{kind}_{sample_id}.svg")
        summary.append({"kind": kind, "id": sample_id, "raw_text": record["raw_text"], "valid_words": len(_valid_words(record)), "pooled_length": record.get("pooled_length", ""), "table": csv_path.name, "timeline": f"{kind}_{sample_id}.svg"})
    (output_dir / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Task 10 Inspection", "", "| kind | id | valid words | pooled length |", "| --- | --- | ---: | ---: |"]
    lines.extend(f"| {item['kind']} | `{item['id']}` | {item['valid_words']} | {item['pooled_length']} |" for item in summary)
    lines.extend(["", "The CSV tables contain ASR words, DTW cost, transferred interval, source audio/video frame ranges, and final pooled position. SVG timelines use blue segments for words retained in the final sequence."])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["paths"]["output"])
    inspect_samples(args.alignment or root / "word_alignment.jsonl", root, args.output or root / "inspection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

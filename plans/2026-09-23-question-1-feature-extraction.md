# Question 1 Feature Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build reproducible, word-aligned features for all 100 raw videos using DistilBERT, PyWorld, OpenFace, and Montreal Forced Aligner (MFA), with output structurally compatible with `aligned_50.pkl`.

**Architecture:** Use the supplied manual transcript as the textual source. MFA produces word timestamps, which define the common sequence axis: DistilBERT provides word embeddings, while PyWorld/OpenFace frame features are overlap-weighted into each word interval. Each sample is padded or continuously pooled to 50 positions.

**Tech Stack:** Python 3.13 managed by uv, PyTorch, Transformers, PyWorld, NumPy, OpenPyXL, FFmpeg, MFA, OpenFace, Pytest, Ruff.

---

## Output Contract

- Create `artifacts/question_1/aligned_50.pkl` with a `train` split containing 100 samples.
- Required fields are `id`, `raw_text`, `text`, `audio`, `vision`, `text_lengths`, `audio_lengths`, `vision_lengths`, `annotations`, `classification_labels`, and `regression_labels`.
- Feature shapes are `text: (100, 50, 768)`, `audio: (100, 50, 74)`, and `vision: (100, 50, 35)`.
- Create `artifacts/question_1/alignment.jsonl`, one record per sample, containing source paths, word timestamps, source-frame ranges, sequence-position mapping, and processing status.
- Do not train or fine-tune any extractor on the 100 labels. Labels are retained only as supplied metadata.

## Task 1: Set Up Project Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `uv.lock`
- Create: `configs/question_1.yaml`

1. Add Python dependencies with `uv add`: `torch`, `transformers`, `pyworld`, `setuptools<81`, `numpy`, `openpyxl`, `pytest`, and `ruff`. PyWorld 0.3.5 imports the legacy `pkg_resources` module, so the setuptools cap is required.
2. Keep external executable paths in `configs/question_1.yaml`; do not hard-code local paths in Python.
3. Run `uv sync` and `uv run python --version`.
4. Run `uv run ruff check .` after source files exist.

**Acceptance:** all Python packages are installed only in the project environment and the configuration records input/output roots plus `mfa` and `openface` executable paths.

## Task 2: Validate External Tooling

**Files:**
- Create: `scripts/question_1/check_tools.py`
- Test: `tests/test_check_tools.py`

1. Verify `ffmpeg` and `ffprobe` are callable.
2. Verify MFA has an installed English acoustic model and dictionary.
3. Verify that PyWorld imports from the project environment.
4. Verify OpenFace `FeatureExtraction` can process a short MP4 and return frame timestamps plus all fields selected by `configs/openface_35.json`.
5. Record executable versions and model versions in `artifacts/question_1/run_metadata.json`.

**Acceptance:** the command exits nonzero with a useful missing-tool error; it exits zero only when all required tools and models are available.

## Task 3: Build the Source Manifest

**Files:**
- Create: `scripts/question_1/build_manifest.py`
- Create: `tests/test_manifest.py`
- Create: `artifacts/question_1/manifest.csv`

1. Read `label-100.xlsx` using OpenPyXL.
2. Form each ID as `{video_id}_{clip_id}` and resolve its MP4 path below the attachment-1 directory.
3. Preserve the source text, continuous label, and English polarity label.
4. Reject duplicate IDs, missing videos, empty transcriptions, and labels outside `[-3, 3]`.

**Acceptance:** the manifest has exactly 100 rows, every row has one existing MP4, and no IDs repeat.

## Task 4: Extract and Inspect Source Media

**Files:**
- Create: `scripts/question_1/extract_media.py`
- Create: `tests/test_media_metadata.py`

1. Extract each source MP4 to a 16 kHz mono WAV with FFmpeg.
2. Collect duration, FPS, video dimensions, audio sample rate, and source hashes.
3. Store generated WAV files under `artifacts/question_1/wav/` and media metadata in the sidecar record.
4. Keep original MP4 files unchanged.

**Acceptance:** WAV duration differs from MP4 duration by no more than one video frame interval, unless an exception is explicitly logged.

## Task 5: Obtain Word-Level Timestamps

**Files:**
- Create: `scripts/question_1/align_words.py`
- Create: `tests/test_word_alignment.py`

1. Normalize punctuation and whitespace for MFA input while retaining the unmodified supplied transcription.
2. Run MFA against each extracted WAV and its supplied transcript.
3. Parse word intervals and map normalized words back to original text positions.
4. Mark, rather than discard, words without a valid interval.

**Acceptance:** each valid alignment interval has `0 <= start < end <= duration`; the sidecar file reports all unmatched words and their reason.

## Task 6: Extract Word-Level Text Features

**Files:**
- Create: `scripts/question_1/extract_text.py`
- Create: `tests/test_text_features.py`

1. Load `distilbert-base-uncased` in inference mode.
2. Tokenize the supplied transcript with word-to-WordPiece mapping.
3. Mean-pool WordPiece embeddings belonging to the same original word.
4. Return a `(L, 768)` array and the token spans for the alignment sidecar file.

**Acceptance:** every aligned word has a finite 768-dimensional feature vector; extractor parameters are never updated.

## Task 7: Extract and Aggregate PyWorld Audio Features

**Files:**
- Create: `scripts/question_1/extract_audio.py`
- Create: `tests/test_audio_aggregation.py`

1. Use PyWorld `dio` and `stonemask` to extract frame-level F0, then use `cheaptrick` and `d4c` for spectral envelope and aperiodicity.
2. Assemble 74 dimensions per frame: log-F0, voiced flag, frame energy, 60 Mel-band spectral-envelope values, and 11 Mel-band aperiodicity values.
3. For each word interval, compute an overlap-duration-weighted mean of covered audio frames.
4. Use zeros only for a documented missing or invalid interval and record the reason.

**Acceptance:** a valid sample produces `(L, 74)` finite values and each output position records the contributing source-frame interval.

## Task 8: Extract and Aggregate Visual Features

**Files:**
- Create: `configs/openface_35.json`
- Create: `scripts/question_1/extract_vision.py`
- Create: `tests/test_vision_features.py`

1. Define the stable 35-dimensional mapping: 17 action-unit intensities, 6 head-pose values, 6 gaze values, and 6 landmark-derived geometry measures.
2. Run OpenFace per MP4 and retain its frame timestamps and face-detection confidence.
3. Apply the same overlap-duration-weighted aggregation into MFA word intervals.
4. Mark undetected-face frames and only produce a zero vector if no usable frame overlaps an interval.

**Acceptance:** each valid sample produces `(L, 35)` finite values; the exact selected OpenFace columns and geometry formulas live in the JSON configuration.

## Task 9: Create the 50-Position Aligned Sequence

**Files:**
- Create: `scripts/question_1/build_aligned_50.py`
- Create: `tests/test_aligned_50.py`

1. Use the aligned words as the shared multimodal axis.
2. For `L <= 50`, retain all positions and post-pad every modality with zero vectors.
3. For `L > 50`, partition consecutive words into 50 nonempty spans and mean-pool each modality within its span; never truncate tail words.
4. Populate all length fields with the number of valid positions and retain the original-to-pooled position map in `alignment.jsonl`.
5. Serialize the pkl fields in the same index-oriented layout as attachment 2.

**Acceptance:** `text`, `audio`, and `vision` have fixed shapes `(100, 50, 768)`, `(100, 50, 74)`, `(100, 50, 35)` and padding is exactly zero after each valid length.

## Task 10: Validate One Short and One Long Sample

**Files:**
- Create: `scripts/question_1/inspect_sample.py`
- Create: `artifacts/question_1/inspection/`

1. Run tasks 3-9 on one short clip and one long clip before batch processing.
2. Produce a table with the original text, aligned time interval, audio frame range, visual frame range, and final sequence position.
3. Generate a compact visual timeline for manual inspection.
4. Confirm that a selected word maps to its correct audio interval and video frames.

**Acceptance:** manual inspection confirms alignment for both samples, including at least one pooled position if the long sample has more than 50 words.

## Task 11: Batch Process All Samples

**Files:**
- Create: `scripts/question_1/run_pipeline.py`
- Create: `artifacts/question_1/logs/`

1. Process manifest rows independently in deterministic ID order.
2. Skip completed samples whose source hashes and extractor configuration match prior records.
3. Write per-sample status, elapsed time, warnings, and traceback details to logs.
4. Continue after a failed sample, then report all failed IDs at the end.

**Acceptance:** all 100 source samples have a terminal status; failures are explicit and re-runnable rather than silently omitted.

## Task 12: Run Structural Validation and Produce Submission Evidence

**Files:**
- Create: `scripts/question_1/validate_output.py`
- Create: `scripts/question_1/build_summary.py`
- Create: `tests/test_output_contract.py`
- Create: `artifacts/question_1/feature_summary.csv`
- Create: `artifacts/question_1/run_metadata.json`

1. Validate ID uniqueness, tensor shapes, label correspondence, finite values, valid lengths, zero padding, and alignment-sidecar coverage.
2. Summarize every sample's source duration, effective length, dimensions, pooled-word count, face-detection rate, and warnings.
3. Save versions, model identifiers, configuration hashes, commands, and source hashes in metadata.
4. Run `uv run pytest` and `uv run ruff check scripts tests`.

**Acceptance:** all tests pass; `feature_summary.csv` contains 100 rows and supplies the full-result table required by the problem statement.

## External Installation Gate

MFA and OpenFace are external tools. Before Task 2, determine their installation method and storage location. Any command that installs system packages, writes outside this repository, or downloads model assets to a home-directory cache requires explicit user approval first.

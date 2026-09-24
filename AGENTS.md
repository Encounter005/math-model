# Repository Guide

## Workflow
- Do not create or use Git worktrees in this repository.
- Use Python >=3.13 through `uv` only, with `UV_CACHE_DIR="$PWD/.uv-cache"`; `uv.lock` is the dependency source of truth. Keep `setuptools<81` because PyWorld imports `pkg_resources`.
- Keep `pytest.ini`'s repository-root `pythonpath`: tests import `scripts.*` directly.

## Question 2 Contract
- Source is `scripts/question2/`; `scripts/question2/run_experiment.py` is the experiment entry point; derived outputs are `artifacts/question_2/`.
- Train and validate only on `E/E题数据/E题数据/附件2-数据集特征文件/aligned_50.pkl`. Attachment 3 is unlabelled final inference only: never fit, select thresholds, or tune hyperparameters on it.
- Use `text_bert` for every split. `FrozenTextEncoder` must load the local DistilBERT cache with `local_files_only=True` and remain frozen; never download model assets.
- Keep `reconstruction`, `gated_fusion`, and `missmodal_alignment` independent. Seeds are `42`--`46` and must be fixed before model construction. Tests inject fake text encoders and require neither a GPU nor model assets.
- Competition metrics are continuous blocks over valid timesteps: 7 modality subsets × 7 rates (10%--70%) × 3 positions (`front`/`middle`/`back`) = 147 conditions per seed. Do not combine them with literature random-position or whole-modality metrics.
- `gating_diagnostics.csv` is limited to gated-fusion `V` and `AV` whole-modality evaluations; do not add per-sample dumps for all competition conditions.

## Experiment Runs
- Use `--dry-run` to resolve paths without artifacts. A normal run executes configured seeds, overwrites that seed's artifact directory, then writes summaries and an HTML report.
- Run routes serially in `tmux`; they share one GPU. `--resume` requires an existing readable `checkpoint_last.pt`; inputs also require attachment 2, attachment-3 `.pkl` files, and `<model_cache>/snapshots/`.
- Keep custom configs in `scripts/question2/`: paths are resolved relative to the repository root inferred from that directory.
- For a paired robustness comparison, run all three routes with `config_seeded_baseline.yaml` first, then all three with `config_robustness.yaml`. Never overwrite the historical `artifacts/question_2/<route>/` baseline.
- Compare only `competition_seed_means.csv` and `competition_summary.csv`: average 147 conditions within each seed, then report mean and sample standard deviation across five seeds. Keep a robustness route only when mean weighted-F1 is no lower and mean MAE is at most baseline `+0.01`.
- For unmasked complete-view training, follow `plans/2026-09-24-question2-unmasked-complete-view.md` and use its isolated `complete_view_v1` artifact root; complete-view validation metrics and masked robustness summaries are separate results.

## Verification
- Run `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_*.py` before `UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts/question2 tests/test_question2_*.py`.
- Focus model tests with `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_models.py`.
=======
## Scope
- Work on branch `question1`; the implemented Question 1 pipeline is in `scripts/question_1/` and is configured by `configs/question_1.yaml`.
- `README.md` has no setup instructions; use `pyproject.toml`, `uv.lock`, the config, and script CLIs as the executable sources of truth.
- Use Python `>=3.13` through `uv`; do not invoke a system interpreter.

## Pipeline
- Run stages in this order because each stage consumes the previous artifact: `build_manifest.py` -> `extract_media.py` -> `align_words.py` -> `extract_text.py` -> `extract_audio.py` -> `extract_vision.py` -> `build_aligned_50.py`.
- Outputs belong under `artifacts/question_1/`; downstream stages must consume earlier artifacts rather than re-reading raw labels or media.
- `align_words.py` aligns Whisper word timestamps to the supplied transcript with DTW. Preserve accepted ASR indices and costs, and record an explicit unmatched reason for every rejected word.
- Text extraction requires the cached `distilbert-base-uncased` model and uses `local_files_only=True`; text archives contain `features`, `word_indices`, and `token_spans`.
- Audio features are 74-D PyWorld overlap-weighted word vectors at 16 kHz. Vision features are 35-D OpenFace vectors using `/opt/openface/bin/FeatureExtraction` and `configs/openface_35.json`.
- OpenFace CSV headers are stripped with `.strip()` before lookup; retain this when changing visual extraction.
- `build_aligned_50.py` uses valid text `word_indices` as the shared axis, filters audio/vision rows to that axis, pools long sequences into 50 nonempty spans, and zero-pads short or empty sequences.
- `inspect_sample.py` reads completed artifacts and writes Task 10 CSV/SVG evidence under `artifacts/question_1/inspection/`; it does not perform extraction.
- `run_pipeline.py` processes samples in deterministic ID order, records terminal statuses in `logs/status.jsonl`, and only skips a prior completion when both source hash and extractor-config hash match. Its default output is the isolated `artifacts/question_1/pipeline_v2/`; do not confuse it with the base artifact root.
- `validate_output.py` reads the `pipeline_v2` aligned output and sidecars; `build_summary.py` writes Task 12 evidence (`feature_summary.csv` and expanded `run_metadata.json`) to the base `artifacts/question_1/` root.

## Commands
- Sync the project environment: `UV_CACHE_DIR="$PWD/.uv-cache" uv sync`.
- Run one test file: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_<name>.py`.
- Full verification: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q` then `UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts tests src`.
- Check configured executables and the locally cached Whisper model: `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/check_tools.py`.
- Run sample inspection after Task 9 artifacts exist: `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/inspect_sample.py --config configs/question_1.yaml`.
- Resume the batch pipeline: `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/run_pipeline.py --config configs/question_1.yaml`; inspect failures in `artifacts/question_1/pipeline_v2/logs/status.jsonl`.
- Validate Task 12 output: `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/validate_output.py --config configs/question_1.yaml`.
- Regenerate Task 12 evidence: `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/build_summary.py --config configs/question_1.yaml`.
- OpenFace extraction is expensive; use tmux for runs likely to exceed two minutes and never run two extraction batches concurrently.

## Data And Constraints
- Raw labels and videos are under `E/E题数据/E题数据/`; all paths for inputs, outputs, model cache, and external tools are centralized in `configs/question_1.yaml`.
- Do not download additional model assets or install system packages without explicit approval. The Whisper model is expected in `.hf-cache/`; text extraction is offline-only.
- Keep `setuptools<81`: PyWorld imports the legacy `pkg_resources` module.
- Raw data, generated `artifacts/`, `.hf-cache/`, `tests/`, `references/`, and `uv.lock` are ignored or not intended for commits; stage explicit source/config files rather than `git add -A`.
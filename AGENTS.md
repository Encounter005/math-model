# Repository Guide

## Environment And Workspace

- Use Python >=3.13 through `uv` only. Bootstrap with `UV_CACHE_DIR="$PWD/.uv-cache" uv sync`; `uv.lock` is authoritative and `setuptools<81` is required because PyWorld imports `pkg_resources`.
- Do not create Git worktrees. `pytest.ini` sets `pythonpath = .` so tests can import `scripts.*`.
- Do not download model assets or install system packages without approval. Model paths come from the selected config and must contain a local `snapshots/` directory; `FrozenTextEncoder` must use `local_files_only=True`.
- `E/`, `.hf-cache/`, `tests/`, and `uv.lock` are ignored. `artifacts/` is not ignored: leave generated artifacts, checkpoints, and reports unstaged; stage explicit source, config, documentation, or plan files only.

## Question 1

- Source: `scripts/question_1/`; config: `configs/question_1.yaml`; batch output root: `artifacts/question_1/pipeline_v2/`.
- Preserve the required dependency order: `build_manifest.py` -> `extract_media.py` -> `align_words.py` -> `extract_text.py` -> `extract_audio.py` -> `extract_vision.py` -> `build_aligned_50.py`. Downstream stages consume preceding artifacts, never raw media or labels.
- Alignment must retain accepted ASR indices and DTW costs; every rejected word needs `unmatched_reason`. The shared axis in `build_aligned_50.py` is valid text `word_indices`; long inputs pool into 50 nonempty spans and short/empty inputs zero-pad.
- Text is cached offline DistilBERT; audio is 74-D PyWorld at 16 kHz; vision is 35-D OpenFace from `/opt/openface/bin/FeatureExtraction`. Keep `.strip()` when reading OpenFace CSV headers.
- Check tools, then run and validate in this order:
  `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/check_tools.py`
  `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/run_pipeline.py --config configs/question_1.yaml`
  `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/validate_output.py --config configs/question_1.yaml`
  `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/build_summary.py --config configs/question_1.yaml`
- Run OpenFace extraction in one serial tmux job because it is expensive and shares the GPU.

## Question 2

- Source and CLI: `scripts/question2/`, `scripts/question2/run_experiment.py`. The supported routes are `reconstruction`, `gated_fusion`, `missmodal_alignment`, and `self_distillation`; keep their model behavior and historical artifact roots independent.
- Train and select only from Attachment 2 `aligned_50.pkl`: the entry point builds loaders only for `train` and `valid`. Attachment 3 is unlabelled final inference only; do not fit, tune, select, or score against it.
- Preserve the input convention: use `text_bert`, derive text validity from its attention mask, and treat audio/vision as valid only where text is valid and the feature row is nonzero.
- Config paths are resolved relative to the repository root. `--dry-run` only resolves paths; `--resume` requires that route/seed's readable `checkpoint_last.pt`.
- Run GPU routes serially in tmux. For paired robustness work, run all three `config_seeded_baseline.yaml` routes before any `config_robustness.yaml` route and never overwrite prior artifacts.
- Competition evaluation is 147 conditions per seed: 7 modality subsets x 7 rates (10--70%) x 3 positions. Keep competition, random-position literature, and whole-modality metrics separate. Compare robustness only through `competition_seed_means.csv` and `competition_summary.csv`: average 147 conditions within each seed, then report five-seed mean and sample SD. Retain a robust route only if weighted-F1 does not decline and MAE is at most baseline +0.01.
- Validate Question 2, when the ignored test bundle is available, before linting:
  `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_*.py`
  `UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts/question2 tests/test_question2_*.py`

## Question 3 Status

- No `scripts/question3/` implementation exists yet. The approved implementation plan is `plans/2026-09-25-question3-gated-self-distillation.md`; follow it rather than repurposing Question 2 checkpoints or artifacts.
- The planned pipeline trains independently on Attachment 2 aligned `train`/`valid` only and uses Attachment 4 aligned samples/videos solely for final inference and explanation.

## Full Verification

- When the ignored test bundle is available, run `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q` followed by `UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts tests src`.

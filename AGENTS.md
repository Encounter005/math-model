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

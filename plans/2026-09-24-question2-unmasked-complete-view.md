# Question 2 Unmasked Complete-View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce reproducible complete-view (no artificial occlusion) validation results for the three Question 2 routes, with final attachment-3 inference kept strictly separate.

**Architecture:** Add one `none` training-mask protocol that returns empty artificial masks while preserving each sample's original availability masks. Reuse the existing complete-view validation path, which already calls `evaluate` without a mask, and write all five-seed outputs to an isolated `complete_view_v1` artifact root. The standard competition and literature perturbation matrices remain diagnostic outputs and must not be combined with the complete-view result.

**Tech Stack:** Python >=3.13 through `uv`, PyTorch, YAML, existing `pytest` and Ruff; no new dependencies, no model downloads.

---

## Interpretation And Constraints

- “无遮挡完整数据集结果” in this plan means no *artificial* modality/time-step mask during training or labelled validation. Natural availability masks in attachment 2 remain intact; zero/unavailable audio or vision values must not be treated as observed data.
- The reported labelled result is the existing held-out `valid` split of `E/E题数据/E题数据/附件2-数据集特征文件/aligned_50.pkl`, evaluated with complete modality views. Do not train on the `valid` or `test` split to score it.
- Attachment 3 cannot yield a labelled complete-view metric because it is unlabelled and contains real missing modalities. Use it only for the existing final inference output.
- Retain the three independent routes: `reconstruction`, `gated_fusion`, and `missmodal_alignment`; use seeds `42` through `46`, fixed before construction.
- Use `text_bert` and the frozen local DistilBERT cache with `local_files_only=True`. Do not download assets.
- Never overwrite `artifacts/question_2/<route>/`, `artifacts/question_2/seeded_baseline_v2/`, or `artifacts/question_2/robustness_v2/`.
- The primary result is `summary.csv`: seed-level complete-view validation Accuracy, macro-F1, weighted-F1, MAE, and Pearson results. `competition_summary.csv` remains a separate masked-robustness summary.

## Task 1: Add An Isolated Complete-View Configuration

**Files:**
- Create: `scripts/question2/config_complete_view.yaml`
- Modify: `tests/test_question2_cli.py`

**Step 1: Write the failing configuration test**

Add a path constant for `config_complete_view.yaml` and a test that resolves it through `load_config`:

```python
def test_complete_view_config_uses_isolated_unmasked_training() -> None:
    config = run_experiment.load_config(COMPLETE_VIEW_CONFIG)

    assert config["paths"]["artifact_root"].endswith("artifacts/question_2/complete_view_v1")
    assert config["training"]["mask_protocol"] == "none"
    assert config["training"]["seeds"] == [42, 43, 44, 45, 46]
```

**Step 2: Run the test to verify it fails**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_cli.py
```

Expected: FAIL because `config_complete_view.yaml` does not exist.

**Step 3: Create the configuration**

Copy `scripts/question2/config_seeded_baseline.yaml` to `scripts/question2/config_complete_view.yaml`. Change only the artifact root and training protocol:

```yaml
paths:
  artifact_root: artifacts/question_2/complete_view_v1
training:
  mask_protocol: none
```

Keep the attachment-2 path, attachment-3 inference path, model cache, hyperparameters, loss settings, evaluation tolerance, and five seeds identical to the seeded baseline configuration.

**Step 4: Run the CLI test to verify it passes**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_cli.py
```

Expected: PASS.

## Task 2: Support Empty Artificial Training Masks

**Files:**
- Modify: `scripts/question2/training.py:333-344`
- Modify: `tests/test_question2_training.py`

**Step 1: Write the failing training-mask test**

Add a test for the new protocol using the existing minimal mask batch:

```python
def test_training_masks_none_preserves_every_originally_valid_timestep() -> None:
    batch = {
        f"{modality}_mask": torch.tensor([[False, True, True, True, True, False]])
        for modality in ("text", "audio", "vision")
    }

    result = _training_masks(batch, seed=42, rates=(0.1,), positions=("front",), protocol="none")

    assert result["protocol"] == "complete_view"
    assert all(not mask.any() for mask in result["artificial_masks"].values())
    assert all(torch.equal(result["observed_masks"][name], batch[f"{name}_mask"]) for name in result["observed_masks"])
```

**Step 2: Run the focused test to verify it fails**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_training.py
```

Expected: FAIL with `ValueError: Unsupported training mask protocol: none`.

**Step 3: Implement the minimal `none` branch**

In `_training_masks`, handle `protocol == "none"` before the existing `competition` and `fixed` branches. Return the standard mask-result structure using `_empty_masks(batch)` for `artificial_masks`, preserve original masks as `observed_masks`, and set serializable metadata:

```python
{
    "original_masks": {name: batch[f"{name}_mask"].bool().clone() for name in MODALITIES},
    "artificial_masks": _empty_masks(batch),
    "observed_masks": {name: batch[f"{name}_mask"].bool().clone() for name in MODALITIES},
    "protocol": "complete_view",
    "condition": {"mask_protocol": "none", "seed": seed},
}
```

Do not change `evaluate`, `evaluate_perturbations`, model methods, or artifact formats. `_loss_terms` and `_outputs` already handle empty masks, so the route-specific losses naturally reduce to complete-view training behavior.

**Step 4: Run the focused test to verify it passes**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_training.py
```

Expected: PASS.

## Task 3: Verify The Complete-View Build

**Files:**
- Modify only files from Tasks 1-2 if verification identifies a defect.

**Step 1: Confirm resolved paths without creating artifacts**

Run once for a representative route:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route reconstruction --config scripts/question2/config_complete_view.yaml --dry-run
```

Expected: reports `artifact_root` ending in `artifacts/question_2/complete_view_v1` and does not create that directory.

**Step 2: Run all Question 2 tests**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_*.py
```

Expected: PASS without GPU use or model downloads.

**Step 3: Run Ruff**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts/question2 tests/test_question2_*.py
```

Expected: no lint errors.

## Task 4: Run The Three Complete-View Routes

**Files:**
- Create derived outputs only: `artifacts/question_2/complete_view_v1/<route>/`

**Step 1: Start a tmux session**

Run routes serially because they share one GPU. Do not use `--resume` for the first run; it requires an existing complete-view checkpoint.

```bash
tmux new-session -s question2-complete-view
```

**Step 2: Run reconstruction and wait for completion**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route reconstruction --config scripts/question2/config_complete_view.yaml
```

Expected: seeds 42-46 complete, then the route summary and HTML report are written.

**Step 3: Run gated fusion and wait for completion**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route gated_fusion --config scripts/question2/config_complete_view.yaml
```

Expected: seeds 42-46 complete independently.

**Step 4: Run MissModal alignment and wait for completion**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route missmodal_alignment --config scripts/question2/config_complete_view.yaml
```

Expected: seeds 42-46 complete independently.

## Task 5: Validate And Report The Complete-View Results

**Files:**
- Read derived outputs: `artifacts/question_2/complete_view_v1/<route>/summary.csv`
- Read derived outputs: `artifacts/question_2/complete_view_v1/<route>/seed_<seed>/validation_predictions.csv`
- Read derived outputs: `artifacts/question_2/complete_view_v1/<route>/seed_<seed>/dataset_manifest.json`
- Read derived outputs: `artifacts/question_2/complete_view_v1/<route>/seed_<seed>/test_predictions.csv`

**Step 1: Verify seed-level deliverables**

For every route and seed 42-46, verify these non-empty outputs exist:

```text
artifacts/question_2/complete_view_v1/<route>/seed_<seed>/config.json
artifacts/question_2/complete_view_v1/<route>/seed_<seed>/dataset_manifest.json
artifacts/question_2/complete_view_v1/<route>/seed_<seed>/validation_predictions.csv
artifacts/question_2/complete_view_v1/<route>/seed_<seed>/test_predictions.csv
artifacts/question_2/complete_view_v1/<route>/seed_<seed>/checkpoint_best.pt
artifacts/question_2/complete_view_v1/<route>/seed_<seed>/checkpoint_last.pt
```

Confirm every saved `config.json` has `training.mask_protocol == "none"`; every `validation_predictions.csv` serializes all-false `artificial_masks`; and every dataset manifest identifies attachment 3 as `unlabelled final inference only`.

**Step 2: Verify route-level deliverables**

For every route, confirm these exist and are non-empty:

```text
artifacts/question_2/complete_view_v1/<route>/summary.csv
artifacts/question_2/complete_view_v1/<route>/report.html
artifacts/question_2/complete_view_v1/<route>/competition_seed_means.csv
artifacts/question_2/complete_view_v1/<route>/competition_summary.csv
```

Treat the latter two as masked robustness diagnostics generated after complete-view training, not as the requested complete-view result.

**Step 3: Publish the result table**

Read each route's `summary.csv` and report the five seed rows plus mean and sample standard deviation for complete-view validation weighted-F1, macro-F1, MAE, and Pearson. Compute the cross-seed statistics from `summary.csv`; do not reuse `competition_summary.csv`, which averages 147 artificial-mask conditions per seed.

State the following limitations with the result: it is a held-out attachment-2 complete-view validation result, not an attachment-3 score; attachment 3 remains prediction-only and may contain real modality absence.

## Completion Criteria

- `config_complete_view.yaml` is the only new configuration and uses an isolated `complete_view_v1` artifact root.
- `none` produces no artificial training masks while preserving original per-modality validity masks.
- All Question 2 tests and Ruff checks pass.
- Each independent route completes seeds 42-46 serially and writes complete-view validation predictions, attachment-3 predictions, checkpoints, `summary.csv`, and `report.html`.
- The final comparison reports only complete-view validation measurements from `summary.csv`; masked protocol metrics remain separate.

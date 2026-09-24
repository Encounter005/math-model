# Question 2 Competition Robustness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the three independent routes' robustness under the competition continuous local-block missingness protocol without using attachment 3 for fitting or overwriting the completed baseline.

**Architecture:** Replace the fixed training mask (all modalities, 30%, middle) with deterministic batch-level sampling from the same modality subsets, rates, and positions used by the competition evaluation protocol. Persist gated-fusion diagnostics for the anomalous `V` and `AV` whole-modality conditions before changing its loss or architecture. Seed before model construction and compare a new seeded fixed-mask baseline with the robustness rerun in separate versioned artifact subdirectories.

**Tech Stack:** Python >=3.13 through `uv`, PyTorch 2.14, NumPy, existing `pytest` and Ruff; no new dependencies, no downloads.

---

## Constraints

- Train and validate only with `E/E题数据/E题数据/附件2-数据集特征文件/aligned_50.pkl`.
- Use frozen, locally cached DistilBERT with `local_files_only=True`.
- Attachment 3 remains final unlabelled inference only.
- Retain independent routes and fixed seeds `42`--`46`; do not add a combined route.
- Keep existing baseline files under `artifacts/question_2/<route>/` unchanged.
- Use `artifacts/question_2/seeded_baseline_v2/<route>/` as the paired comparison baseline; the existing baseline is historical context only because its model initialization was not seeded.
- The primary decision metric is mean and standard deviation of competition-protocol weighted-F1. Literature protocol remains separately reported.
- Do not add early stopping in this iteration. Best-checkpoint selection already protects reported metrics; it does not address the training/evaluation mask mismatch.

## Task 1: Create An Isolated Robustness Configuration

**Files:**
- Create: `scripts/question2/config_robustness.yaml`
- Test: `tests/test_question2_cli.py`

**Step 1: Write the failing test**

Add a test that loads the new configuration and asserts:

```python
assert config["paths"]["artifact_root"].endswith("artifacts/question_2/robustness_v2")
assert config["training"]["mask_protocol"] == "competition"
```

**Step 2: Run the test to verify it fails**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_cli.py
```

Expected: FAIL because the robustness configuration does not exist.

**Step 3: Create the configuration**

Copy all data paths, model settings, seeds, rates, positions, and loss coefficients from `config.yaml`. Change only:

```yaml
paths:
  artifact_root: artifacts/question_2/robustness_v2
training:
  mask_protocol: competition
```

**Step 4: Run the CLI test**

Expected: PASS. The default `config.yaml` remains unchanged.

## Task 2: Sample Competition Masks During Training

**Files:**
- Modify: `scripts/question2/masking.py`
- Modify: `scripts/question2/training.py`
- Modify: `tests/test_question2_masking.py`
- Modify: `tests/test_question2_training.py`

**Step 1: Write failing mask tests**

Define a public `sample_competition_mask` API and test that it:

```python
first = sample_competition_mask(batch, rates=(0.1, 0.3), positions=("front", "back"), seed=42)
second = sample_competition_mask(batch, rates=(0.1, 0.3), positions=("front", "back"), seed=42)
assert first["condition"] == second["condition"]
assert first["protocol"] == "competition"
assert first["condition"]["modalities"] in {"T", "A", "V", "TA", "TV", "AV", "TAV"}
assert first["condition"]["rate"] in {0.1, 0.3}
```

Test that the selected artificial mask remains a valid contiguous block and never includes padding, reusing the existing mask assertions.

**Step 2: Run the mask tests to verify failure**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_masking.py tests/test_question2_training.py
```

Expected: FAIL because the sampler does not exist and training still calls the fixed `_training_masks` implementation.

**Step 3: Implement the deterministic sampler**

In `masking.py`, choose one of the seven non-empty modality subsets, one configured competition rate, and one configured position through a local `random.Random(seed)`. Delegate actual masking to the existing `apply_contiguous_mask`; include serializable condition metadata:

```python
{
    "modalities": "TA",
    "rate": 0.3,
    "position": "middle",
    "seed": seed,
}
```

In `training.py`, replace the fixed `apply_contiguous_mask(batch, MODALITIES, 0.3, "middle", seed)` call. Pass only `competition_rates` and `positions` from config. Retain the existing epoch/batch-derived seed so each batch is reproducible but conditions vary across training.

**Step 4: Run the focused tests**

Expected: PASS.

## Task 3: Persist Gated-Fusion Whole-Modality Diagnostics

**Files:**
- Modify: `scripts/question2/training.py`
- Modify: `scripts/question2/artifacts.py`
- Modify: `scripts/question2/report.py`
- Modify: `scripts/question2/templates/report.html.j2`
- Modify: `tests/test_question2_artifacts.py`

**Step 1: Write failing artifact tests**

Use a temporary route directory and a minimal gated result. Assert that `gating_diagnostics.csv` contains rows for both whole-modality subsets and preserves predictions, reliability gates, importance gates, mixture coefficient, and final modality weights:

```python
assert {row["subset"] for row in rows} == {"V", "AV"}
assert "importance_gates" in rows[0]
assert "fused_modality_weights" in rows[0]
```

Assert that the gated-fusion report includes a `Gating Diagnostics` semantic table.

**Step 2: Run the artifact test to verify failure**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_artifacts.py
```

Expected: FAIL because perturbation evaluation discards per-sample rows and no diagnostic CSV is written.

**Step 3: Preserve only the required diagnostic rows**

Update `_evaluate_masked_batches` to return its existing per-sample rows in addition to aggregate metrics and masks. In `evaluate_perturbations`, only for `route == "gated_fusion"` and subsets `V`/`AV`, attach `subset` to those rows and return them as `gating_diagnostics`.

Update `write_seed_artifacts` to create `gating_diagnostics.csv` only when that result key is present. Add the report table only when the file has records. Do not persist per-sample predictions for all 147 competition conditions; those are unnecessary for diagnosing the observed gate anomaly.

**Step 4: Run the artifact test**

Expected: PASS.

## Task 4: Verify The Complete Robustness Build

**Files:**
- Modify only files from Tasks 1--3 if verification identifies a defect.

**Step 1: Run all Question 2 tests**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_*.py
```

Expected: PASS without GPU or model downloads.

**Step 2: Run Ruff**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts/question2 tests/test_question2_*.py
```

Expected: no lint errors.

## Task 5: Run And Compare The Independent Routes

**Files:**
- Create derived outputs only: `artifacts/question_2/seeded_baseline_v2/<route>/` and `artifacts/question_2/robustness_v2/<route>/`

**Step 1: Run the seeded fixed-mask baseline serially in tmux**

Use `scripts/question2/config_seeded_baseline.yaml` (fixed mask, seeds 42--46). Wait for each command to finish successfully before starting the next:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route reconstruction --config scripts/question2/config_seeded_baseline.yaml
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route gated_fusion --config scripts/question2/config_seeded_baseline.yaml
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route missmodal_alignment --config scripts/question2/config_seeded_baseline.yaml
```

**Step 2: Run the competition-mask routes serially in tmux**

Start only after all three seeded baseline routes finish. Wait for each command to finish successfully before starting the next:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route reconstruction --config scripts/question2/config_robustness.yaml
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route gated_fusion --config scripts/question2/config_robustness.yaml
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route missmodal_alignment --config scripts/question2/config_robustness.yaml
```

Do not run routes concurrently because they share one GPU. Each run writes five seed directories, 30 attachment-3 predictions per seed, and its standalone report. Neither configuration may point to the historical baseline directory.

**Step 3: Verify deliverables and comparison inputs**

For each route in both v2 directories, check seeds 42--46 each have 30 data rows in `test_predictions.csv` and exactly 147 `protocol=competition` rows in `perturbation_metrics.csv` (seven modality subsets, seven rates, three positions). Verify these files are non-empty:

```text
artifacts/question_2/<v2>/<route>/summary.csv
artifacts/question_2/<v2>/<route>/competition_seed_means.csv
artifacts/question_2/<v2>/<route>/competition_summary.csv
artifacts/question_2/<v2>/<route>/report.html
artifacts/question_2/robustness_v2/gated_fusion/seed_42/gating_diagnostics.csv
```

Here `<v2>` is `seeded_baseline_v2` or `robustness_v2`. Confirm both configs use the same attachment-2 split, local frozen encoder, model/loss settings, seeds, and evaluation conditions; only training mask protocol and artifact root differ. Keep attachment 3 strictly for final inference. `summary.csv` reports complete-view checkpoint-selection metrics, not the primary competition comparison.

Compare `competition_seed_means.csv` by matching seed and use `competition_summary.csv` for cross-seed results. Each seed's four metrics are the arithmetic mean over its 147 competition conditions; report the mean and sample standard deviation across five seeds. For each route report:

- mean and standard deviation of competition weighted-F1, macro-F1, MAE, and Pearson;
- seed-wise weighted-F1 differences (`robustness_v2` minus `seeded_baseline_v2`);
- performance by rate, position, and modality subset;
- for gated fusion, pair `V` and `AV` diagnostic rows by seed and sample ID, check prediction agreement, and inspect audio gate mass under `V`.

**Step 4: Acceptance rule**

Keep a robustness-v2 route only if its mean competition weighted-F1 is at least the seeded baseline mean and its mean competition MAE is no more than the seeded baseline mean plus `0.01`. Record pass/fail per route; do not select conditions or adjust the threshold after seeing results. If gated-fusion diagnostics show audio gate mass is consistently near zero under `V`, create a separate, evidence-backed follow-up plan for a gate regularizer; do not introduce one in this iteration.

## Completion Criteria

- The historical baseline remains intact; both seeded baseline and robustness v2 runs complete separately.
- Training mask conditions are reproducible and drawn solely from the competition protocol.
- Gated fusion has direct `V`/`AV` diagnostic evidence.
- All unit tests and Ruff checks pass.
- Each of the six runs completes seeds 42--46 without attachment-3 fitting, validation, or hyperparameter selection.

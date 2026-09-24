# Question 2 Three-Route Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train and evaluate TgRN-inspired reconstruction, AGFN-inspired gated fusion, and MissModal-inspired representation alignment; persist all derived experiment data needed for later plotting; generate one standalone HTML report per route.

**Architecture:** All routes share one aligned-data loader, frozen locally cached DistilBERT text encoder, modality encoders, classification/regression heads, masking protocol, metrics, artifact format, and train/validation split. They run under two explicitly separated protocols: the competition's continuous local missingness protocol and a literature-comparison protocol for random-position and whole-modality missingness. The routes differ in latent reconstruction, dual reliability/importance gates, or complete/incomplete representation alignment.

**Tech Stack:** Python >=3.13 through `uv`, PyTorch 2.14, Transformers 5.17, NumPy, Jinja2, `pytest`, `ruff`; no network downloads and no new dependencies.

---

## Experiment Contract

- Create all source code beneath `scripts/question2/`.
- Read `E/E题数据/E题数据/附件2-数据集特征文件/aligned_50.pkl` for training and validation only.
- Read `E/E题数据/E题数据/附件3-模态缺失特征样本/对齐版本/` only for final unlabelled inference.
- Use `text_bert` in every split. Do not train with the train-only precomputed `text` field because attachment 3 does not supply it.
- Load the existing local `distilbert-base-uncased` cache with `local_files_only=True` and freeze it. Do not download a model or fine-tune it.
- Use five fixed seeds, `42` through `46`, for every route; report mean and standard deviation.
- Use the same shared backbone and hyperparameters in all routes. Do not add a combined route until the three independent routes have completed.
- Persist every derived scalar, per-sample prediction, perturbation condition, and mask needed for plotting. Do not duplicate source data or persist unneeded batch activations.

### Routes

| Route ID | Mechanism | Additional loss |
|---|---|---|
| `reconstruction` | TgRN-inspired, text-guided reconstruction of a missing modality's 128-D temporal latent state, then fusion of completed sequences. | Smooth L1 only over synthetically hidden, originally valid positions. |
| `gated_fusion` | AGFN-inspired dual fusion: entropy-derived reliability gate, learned modality-importance gate, and a learned mixture of their fused representations. | Task loss plus VAT prediction-consistency loss. |
| `missmodal_alignment` | MissModal-inspired complete/incomplete subset alignment using a shared fusion backbone. | Complete and masked task losses, paired distance, and batch contrastive geometry alignment. |

`Chang et al. (2026)` is not a missing-modality method. Its curriculum learning is not a fourth route and must not be included in the primary route comparison. It may be enabled only as a separately labelled, train-only shared enhancement after the three routes have completed.

### Evaluation Protocols

Every route must evaluate and persist both protocols independently; never aggregate their measurements into one metric.

1. **Competition protocol:** continuous local blocks within valid timesteps. Sweep missing modality set, `front`/`middle`/`back` position, and 10% through 70% missing rates. This is the primary result used to answer Question 2.
2. **Literature-comparison protocol:** random independent feature-position masking at rates 10% through 90%, plus whole-modality subsets `T`, `A`, `V`, `TA`, `TV`, and `AV`. This protocol supports TgRN and MissModal comparison, but does not replace the competition protocol.

### Artifact Contract

Each seed run must write:

```text
artifacts/question_2/<route>/seed_<seed>/
  config.json
  environment.json
  dataset_manifest.json
  train_history.jsonl
  validation_predictions.csv
  perturbation_metrics.csv
  whole_modality_metrics.csv
  masks_validation.npz
  test_predictions.csv
  checkpoint_best.pt
  checkpoint_last.pt
  reconstruction_metrics.csv       # reconstruction only
```

Each completed route must write:

```text
artifacts/question_2/<route>/
  summary.csv
  report.html
```

`train_history.jsonl` contains one complete metrics record per epoch. `validation_predictions.csv` contains labels, predictions, probabilities, masks, observed fractions, gates where applicable, and all identifiers. `perturbation_metrics.csv` contains one record for every competition-protocol condition. `whole_modality_metrics.csv` contains the literature-comparison subset results. `masks_validation.npz` preserves the exact 3-by-50 artificial masks that generated the metrics. `test_predictions.csv` contains one record for every attachment-3 file.

### Required Tests

Run all unit tests without GPU or model downloads. Inject a small fake text encoder into model tests. Test fixtures must use temporary pickle files rather than competition source files.

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_*.py
UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts/question2 tests
```

## Task 1: Establish The Question 2 Interface

**Files:**
- Create: `scripts/question2/__init__.py`
- Create: `scripts/question2/config.yaml`
- Create: `scripts/question2/run_experiment.py`
- Create: `tests/test_question2_cli.py`

**Step 1: Write the failing CLI test**

Verify `--route`, `--config`, `--seed`, `--dry-run`, and `--resume` are accepted; verify `reconstruction`, `gated_fusion`, and `missmodal_alignment` are the only supported route values.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_cli.py`

Expected: FAIL because the Question 2 CLI does not exist.

**Step 3: Create the minimum configuration and parser**

Set the data paths, artifact root, cached model path, `hidden_dim: 128`, `batch_size: 32`, `epochs: 30`, `learning_rate: 0.001`, seeds `[42, 43, 44, 45, 46]`, competition rates `0.1` through `0.7`, literature rates `0.1` through `0.9`, positions `front`, `middle`, `back`, reconstruction coefficient `1.0`, VAT coefficient `0.1`, and separate MissModal semantic, distance, and geometry coefficients.

`--dry-run` must load configuration and print resolved paths and route name without creating artifacts.

**Step 4: Run the CLI test**

Expected: PASS.

## Task 2: Load Data Through A Single Train/Test Interface

**Files:**
- Create: `scripts/question2/data.py`
- Create: `tests/test_question2_data.py`

**Step 1: Write failing data tests**

Cover:
- Attachment-2 split loading from its column-oriented pickle structure.
- `Negative`, `Neutral`, and `Positive` mapping to 0, 1, and 2.
- `text_bert[1]` becoming the text validity mask.
- Audio/vision zero rows within valid text positions becoming unavailable.
- Attachment-3 single-file `{"test": ...}` payload loading with its filename as ID.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_data.py`

**Step 3: Implement data records and dataset classes**

Return a batch dictionary with `id`, `input_ids`, `attention_mask`, `audio`, `vision`, `text_mask`, `audio_mask`, `vision_mask`, `class_label`, and `regression_label`. Preserve all masks separately: valid sequence masks are not artificial missingness masks.

**Step 4: Run the test to verify it passes**

Expected: PASS.

## Task 3: Generate Reproducible Continuous Missing Blocks

**Files:**
- Create: `scripts/question2/masking.py`
- Create: `tests/test_question2_masking.py`

**Step 1: Write failing mask tests**

Test a mask generator that:
- selects only originally valid timesteps;
- chooses a contiguous interval with at least one timestep;
- honors requested rate and `front`, `middle`, or `back` placement;
- never selects padding;
- produces identical masks for the same seed.
- supports independent random-position masks and all six non-empty incomplete modality subsets for the literature-comparison protocol.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_masking.py`

**Step 3: Implement `apply_contiguous_mask`**

Return original masks, artificial masks, final observed masks, protocol name, and condition metadata. For text, retain token IDs and attention masks but apply the artificial mask after text encoding. For audio and vision, apply the mask to input features and propagate it to pooling.

**Step 4: Run the test to verify it passes**

Expected: PASS.

## Task 4: Implement The Shared Backbone And Task Heads

**Files:**
- Create: `scripts/question2/models.py`
- Create: `tests/test_question2_models.py`

**Step 1: Write failing model tests**

With a fake text encoder, assert that a batch produces classification logits of shape `[batch, 3]`, regression predictions of shape `[batch]`, and no NaN when one modality is completely unavailable.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_models.py`

**Step 3: Implement shared model components**

Implement:
- `FrozenTextEncoder`, loading the cached DistilBERT with `local_files_only=True` and disabled gradients.
- Audio and vision linear projections to 128 dimensions.
- Masked temporal mean pooling that returns zero for an unavailable modality rather than NaN.
- A shared classifier producing three logits.
- A shared regression head producing a value clamped to `[-3, 3]`.
- Weighted cross-entropy plus Smooth L1 loss for the supervised objective.

**Step 4: Run the model test**

Expected: PASS.

## Task 5: Implement Route 1, Latent Reconstruction

**Files:**
- Modify: `scripts/question2/models.py`
- Create: `tests/test_question2_reconstruction.py`

**Step 1: Write failing reconstruction tests**

Verify that reconstruction:
- replaces only artificially hidden positions;
- computes zero reconstruction loss if no positions were hidden;
- computes loss only where original data was valid and artificially hidden;
- returns per-sample, per-modality reconstruction errors.
- uses Smooth L1 rather than MSE at masked positions.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_reconstruction.py`

**Step 3: Implement `ReconstructionModel`**

Use text-guided cross-modal attention over the observed projected modality sequences, final masks, and positional index to predict the target modality's 128-D latent sequence. Calculate Smooth L1 only on target positions selected by the artificial mask. Use reconstructed vectors only at those positions before the normal fused prediction. Keep the complete projected feature as the reconstruction target; document that this is a latent-reconstruction adaptation of TgRN, not raw-feature reconstruction.

Set the initial reconstruction coefficient to `1.0`; retain it in config and all artifact metadata.

**Step 4: Run the test**

Expected: PASS.

## Task 6: Implement Route 2, Mask-Aware Dynamic Gating

**Files:**
- Modify: `scripts/question2/models.py`
- Create: `tests/test_question2_gating.py`

**Step 1: Write failing gate tests**

Verify that both gate families sum to one per sample after fusion, a fully unavailable modality receives exactly zero gate mass, and VAT produces a finite prediction-consistency loss.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_gating.py`

**Step 3: Implement `GatedFusionModel`**

Implement AGFN-inspired dual fusion:
- derive reliability weights from normalized feature entropy with a configured temperature;
- derive modality-importance gates from the concatenated pooled modality states and observed/missing ratios;
- fuse both gated representations and combine them with a learned scalar mixture;
- apply a virtual adversarial perturbation to the fused representation and penalize the original/perturbed prediction difference with coefficient `0.1`.

Mask unavailable modalities before every softmax. Save entropy weights, importance gates, mixture coefficient, and final fused modality weights for every validation and attachment-3 prediction. Label this route as an AGFN extension because the original paper does not define local-missing masks.

**Step 4: Run the test**

Expected: PASS.

## Task 7: Implement Route 3, MissModal Alignment

**Files:**
- Modify: `scripts/question2/models.py`
- Create: `tests/test_question2_missmodal.py`

**Step 1: Write failing alignment tests**

Verify that complete and incomplete views use one shared model, that no artificial mask makes the paired-distance loss zero, that contrastive geometry loss is finite, and that each incomplete view receives a supervised task loss.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_missmodal.py`

**Step 3: Implement `MissModalAlignmentModel`**

Encode one complete view and all selected incomplete views using shared weights. For each incomplete view, calculate: task loss against the sentiment label, paired squared-L2 distance to the complete representation, and an InfoNCE-style batch contrastive geometry loss that identifies the matching complete utterance against other utterances in the batch. Train the complete view with its own task loss. Record independent coefficients, temperature, and all four loss terms in configuration and training history.

**Step 4: Run the test**

Expected: PASS.

## Task 8: Train, Validate, And Evaluate The Perturbation Matrix

**Files:**
- Create: `scripts/question2/training.py`
- Create: `scripts/question2/metrics.py`
- Create: `tests/test_question2_metrics.py`

**Step 1: Write failing metric tests**

Use fixed labels and predictions to verify Accuracy, macro-F1, weighted-F1, MAE, and Pearson correlation. Test Pearson's constant-vector edge case returns a recorded undefined value rather than crashing.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_metrics.py`

**Step 3: Implement the trainer**

- Fix Python, NumPy, PyTorch, and CUDA seeds.
- Record one `train_history.jsonl` record per epoch with all loss terms and validation indicators.
- Select the best checkpoint by highest weighted-F1, breaking ties by lowest MAE.
- Evaluate complete validation data, every competition-protocol condition, and every literature-comparison whole-modality subset condition. Keep protocol results in separate files and report sections.
- Save all validation predictions from the selected checkpoint.
- Never use attachment 3 during fitting or hyperparameter selection.

**Step 4: Run tests**

Expected: PASS.

## Task 9: Persist Artifacts And Render Per-Route HTML Reports

**Files:**
- Create: `scripts/question2/artifacts.py`
- Create: `scripts/question2/report.py`
- Create: `scripts/question2/templates/report.html.j2`
- Create: `tests/test_question2_artifacts.py`

**Step 1: Write failing artifact tests**

Using temporary files, verify required JSON, JSONL, CSV, NPZ, and checkpoint paths are created. Verify that the rendered report lists all source artifact paths and includes each required metric table.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_artifacts.py`

**Step 3: Implement artifact writers**

Persist:
- resolved configuration and library/CUDA/device versions;
- split counts and source paths;
- epoch-level measurements;
- per-sample validation and test predictions;
- all perturbation metrics and exact masks;
- reconstruction data for route 1;
- entropy and importance gates, the learned mixture coefficient, and VAT loss for route 2;
- complete/incomplete task losses, paired distances, and geometry losses for route 3;
- checkpoints;
- seed-level and route-level summary tables.

Render a standalone Jinja2 report. Include separate competition and literature-comparison tables, route-specific loss summaries, and an explicit literature-status note: TgRN latent adaptation, AGFN local-missing extension, or MissModal local-block extension. Use semantic HTML tables only; do not add plotting dependencies or create plots.

**Step 4: Run tests**

Expected: PASS.

## Task 10: Wire The CLI And Run Each Route

**Files:**
- Modify: `scripts/question2/run_experiment.py`
- Modify: `tests/test_question2_cli.py`

**Step 1: Extend the CLI test**

Mock the training executor and verify a route invocation iterates all configured seeds, renders a report only after all seeds succeed, and resumes from a valid last checkpoint only when `--resume` is given.

**Step 2: Run the test to verify it fails**

Run: `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_cli.py`

**Step 3: Implement route orchestration**

Dispatch the selected model class, run all seeds serially, aggregate `summary.csv`, and render `report.html`. Fail immediately if an input file, cached model, or expected artifact is unavailable.

**Step 4: Run all Question 2 tests and lint**

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q tests/test_question2_*.py
UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts/question2 tests
```

Expected: PASS with no lint errors.

**Step 5: Run experiments serially in tmux**

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route reconstruction
UV_CACHE_DIR="$PWD/.uv-cache"  uv run python scripts/question2/run_experiment.py --route gated_fusion
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question2/run_experiment.py --route missmodal_alignment
```

Do not run routes concurrently because they share one GPU.

**Step 6: Verify deliverables**

For each route, verify these files exist and have nonempty records:

```text
artifacts/question_2/<route>/report.html
artifacts/question_2/<route>/summary.csv
artifacts/question_2/<route>/seed_42/test_predictions.csv
artifacts/question_2/<route>/seed_43/test_predictions.csv
artifacts/question_2/<route>/seed_44/test_predictions.csv
artifacts/question_2/<route>/seed_45/test_predictions.csv
artifacts/question_2/<route>/seed_46/test_predictions.csv
```

Verify every `test_predictions.csv` contains 30 attachment-3 rows, all five seed directories exist, every route has both protocol result files, and no training history or artifact metadata lists attachment 3 as a fit or validation input.

## Completion Criteria

- All Question 2 tests and lint checks pass.
- Each of the three independent routes has completed seeds 42 through 46.
- Each route provides one standalone HTML report and one aggregate CSV.
- All validation and controlled-missingness data required for later plotting exists as CSV, JSONL, or NPZ artifacts, separated by protocol.
- Every route writes 30 attachment-3 predictions without using unlabelled test data for training.

# AGENTS.md

## Scope and data
- This is a 2026 Chinese Graduate Mathematical Modeling Competition workspace. Active work is E题, with Question 1 code on branch `question1`.
- E题 data is nested at `E/E题数据/E题数据/`; Attachment 1 videos and `label-100.xlsx` are under `附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/`.
- Read `.docx` with `pandoc -t plain "<file>.docx"` and PDFs with `pdftotext -l <pages> "<file>.pdf" -`.
- Attachment 2 pkl fields are column-oriented: use `data['train'][field][j]`, never `data['train'][j][field]`. Aligned tensors use 50 positions; unaligned audio/vision use 500.

## Git and generated files
- Never use `git add -A`: `E/`, root archives, and `references/` are untracked and can be multi-GB. Stage paths explicitly; never commit source archives or generated `artifacts/`.
- `.gitignore` currently ignores `/tests/` and `uv.lock`; test or lock-file changes will not appear in `git status` unless ignore rules are changed or paths are force-added.

## Python and Question 1
- Use Python 3.13 with uv. Keep uv writes local: `UV_CACHE_DIR="$PWD/.uv-cache" uv sync`.
- Verify code with `UV_CACHE_DIR="$PWD/.uv-cache" uv run pytest -q` and `UV_CACHE_DIR="$PWD/.uv-cache" uv run ruff check scripts tests src`.
- Question 1 settings live in `configs/question_1.yaml`; run its prerequisite check with `UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/question_1/check_tools.py`.
- MFA must use `/home/encounter/.conda/envs/aligner/bin/mfa`, not the uv-installed MFA package. Required local acoustic model and dictionary are both `english_us_arpa`.
- OpenFace is `/opt/openface/bin/FeatureExtraction`. PyWorld replaces COVAREP; keep `setuptools<81` because PyWorld 0.3.5 imports legacy `pkg_resources`.

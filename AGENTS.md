# AGENTS.md

## What this repo is
Workspace for the 2026 中国研究生数学建模竞赛 (研究生数模/华为杯). Not a software
library — it holds problem statements, provided data, and reference papers.

- `A/`–`F/` — one folder per problem (`<题目>.docx` + data/subfolders).
- `已加速- ...中文题目/中文题目/` — original per-problem zip distributions (A–F).
- Root `已加速-...zip` (~2.3 GB) — combined source archive.
- `references/` — papers for the chosen problem, grouped by `question_1/`,
  `question_2/`, `question_3/`, `Dataset/` (currently built around E题).
- Active problem: **E题 复杂场景下多模态情感识别** (`E/`, `references/`).

## Reading problem / paper files (easy to get wrong)
- `.docx` → `pandoc -t plain "<file>.docx"`. The Read tool rejects docx.
- `.pdf` → `pdftotext -l <pages> "<file>.pdf" -`. Limit pages; PDFs are large.
  This model cannot ingest PDF via Read.
- E题 feature pkl access is `data['train'][field][j]`, NOT `data['train'][j][field]`.
  Shapes differ aligned vs unaligned: text 50×768; audio 50 (aligned) / 500 (unaligned) ×74;
  vision 50/500×35. See `E/*.docx` §补充说明.

## Git / disk — dangerous
- No `.gitignore`; only `README.md` is tracked. `git add -A` would stage multi-GB
  archives (`E/E题数据.zip` 1.8 GB, root zip 2.3 GB) and extracted data (~3.8 GB).
  Add files selectively or write `.gitignore` first. Never commit the archives.
- E题 data is nested twice: `E/E题数据/E题数据/...`. F题 data: `F/real_attachments/`.

## Python
- Python 3.13, managed with `uv`. No dependencies or `uv.lock` yet.
- `pyproject.toml` declares `math-model = "math_model:main"`, but `src/` is empty and
  no `math_model` package exists — that entrypoint does not run.
- `test.py` is scratch, not a test suite. No linter/test runner configured.

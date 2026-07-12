# Folder: Collection_of_prompt

## Context & Purpose

This folder is the **raw prompt-dataset store** for the project. It holds the external
adversarial and benign prompt corpora that were collected during research and later
distilled into the curated attack set (`attack_prompts.xlsx`) that each pipeline
iteration actually runs. It sits at the repository root alongside the four pipeline
iterations (`first-model`, `second model`, `third_model`, `forth_model`) and feeds the
**Generation** stage described in the [root README](../README.md): a target LLM is asked
each prompt, and its response is later judged for safety/robustness.

These files are **inputs / reference data only** — no code here executes them. The
per-iteration `model.py` scripts read the curated `attack_prompts.xlsx` (a copy of which
lives inside each iteration folder), not the CSVs in this folder directly.

## Key Components

- **`attack_prompts.xlsx`** — The curated adversarial prompt set used as the canonical
  benchmark input. The pipeline code reads an Excel file with (at least) the columns
  `AttackMethod` and `prompt`, and filters rows where `AttackMethod == 'Persuative LLM'`
  (the spelling "Persuative" is used verbatim in the source). Copies of this file are
  present in every iteration folder.
- **`cysecbench.csv`** — The CySecBench cybersecurity-prompt dataset. Columns: `Prompt`,
  `Category` (~12.6k rows). External reference corpus.
- **`jailbreak_prompts_2023_05_07.csv`** / **`jailbreak_prompts_2023_12_25.csv`** — Two
  dated snapshots of the *jailbreak_llms* prompt collection. Columns include `platform`,
  `source`, `prompt`, `jailbreak`, `created_at`, `date`, `community_id`,
  `community_name`. Used as a source of adversarial/jailbreak prompts.
- **`regular_prompts_2023_05_07.csv`** / **`regular_prompts_2023_12_25.csv`** — The
  benign ("regular", non-jailbreak) counterpart snapshots from the same source, kept as a
  control/baseline corpus.

> Provenance of these datasets is recorded in the root-level [`Prompt`](../Prompt) notes
> file and the "References" section of the [root README](../README.md) (MalwareBench,
> jailbreak_llms, CySecBench, Codesagar malicious-llm-prompts, etc.).

## Execution / Usage

Nothing in this folder is executable. These are static data assets consumed manually and
during curation:

- The XLSX/CSV files are opened in pandas / Excel for inspection and filtering.
- The curated `attack_prompts.xlsx` is what the pipeline scripts load — see
  [`first-model`](../first-model/README.md), [`second model`](../second%20model/README.md),
  [`third_model`](../third_model/README.md), and [`forth_model`](../forth_model/README.md).

## Dependencies

- No package dependencies of its own (data only).
- Downstream consumers rely on `pandas` + `openpyxl` to read `attack_prompts.xlsx`.
- Related project files: the per-iteration `attack_prompts.xlsx` copies and the root
  [`Prompt`](../Prompt) provenance notes.

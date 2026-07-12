# Folder: forth_model / results

## Context & Purpose

Store of **raw Stage-1 response CSVs** for the [`forth_model`](../README.md) pipeline —
the direct output of `model.py` and the **input to Stage 2** (`evaluator.py`). Each file
holds one target model's answers to the adversarial prompt set, before any judging or
VirusTotal scanning.

## Key Components

Every file uses the `model.py` output schema: `AttackMethod, prompt, Response, status`
(`status` is `OK` or `FAILED`). One CSV per target model:

- `responses_results_codestral-latest.csv`, `responses_results_codestral-latest_1.csv` —
  Mistral **codestral** target (full run + a batch).
- `responses_results_mistral-small-latest.csv`,
  `responses_results_mistral-small-latest_1.csv` — Mistral small-latest target.
- `responses_results_llama-3.1-8b-instant.csv` — Groq Llama-3.1-8B target.
- `responses_results_meta-llama_Meta-Llama-3.1-70B-Instruct.csv` — HuggingFace
  Llama-3.1-70B target.
- `responses_results_qwen_qwen3-32b.csv` — Qwen3-32B target.
- `responses_results_sonar.csv` — Perplexity `sonar` target.

## Execution / Usage

Not executable. Feed a file into Stage 2 by setting `INPUT_FILE` in
[`forth_model/evaluator.py`](../evaluator.py) (or `evaluator_2.py`) to the desired CSV,
then running the evaluator. Reproduce these files with
[`forth_model/model.py`](../model.py) by setting `ACTIVE_MODEL`.

## Dependencies

- None (data only). Schema owned by [`forth_model/model.py`](../model.py).

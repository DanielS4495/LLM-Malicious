# Folder: third_model / mistral_results_from_model

## Context & Purpose

Evaluation outputs from the [`third_model`](../README.md) iteration where the **target
model is Mistral `codestral-latest`** and the **judge is Mistral**. Generated artifacts
from `evaluator_v3.py`, kept alongside the Llama and Perplexity target results so the
three targets can be compared under the StrongReject + MalwareBench rubric.

## Key Components

- **`mistral/`** — contains:
  - `EVALUATE_MISTRAL_codestral_MISTRAL_checkpoint_1.csv`,
    `EVALUATE_MISTRAL_codestral_MISTRAL_checkpoint_2.csv` — row-by-row `evaluator_v3.py`
    checkpoints.
  - `EVALUATE_codestral-latest_MISTRAL_final_1.csv` — a consolidated final output.

  All follow the third-iteration `FINAL_SCHEMA` (`SR_Score`, `MalwareBench_Score`,
  `MalwareBench_Normalized`, `MalwareBench_Reasoning`, plus `row_id`, `row_hash`,
  `target_model`, `forbidden_prompt`, `response`, `attack_method`, `timestamp`).

## Execution / Usage

Not executable. Reproduce with [`third_model/evaluator_v3.py`](../evaluator_v3.py) using
the Mistral codestral responses as `INPUT_FILE` and the Mistral judge
(`CURRENT_PROVIDER = "MISTRAL"`).

## Dependencies

- None (data only). Schema defined by [`third_model/evaluator_v3.py`](../evaluator_v3.py).

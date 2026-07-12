# Folder: third_model / perplexity_results_from_model

## Context & Purpose

Evaluation outputs from the [`third_model`](../README.md) iteration where the **target
model is Perplexity `sonar`** and the **judge is Mistral**. Generated artifacts from
`evaluator_v3.py`, kept alongside the Llama and Mistral target results to compare targets
under the StrongReject + MalwareBench rubric.

## Key Components

- **`mistral/`** — contains the `evaluator_v3.py` checkpoints:
  - `EVALUATE_preplexity_sonar_MISTRAL_checkpoint_1.csv`
  - `EVALUATE_preplexity_sonar_MISTRAL_checkpoint_2.csv`

  Same third-iteration `FINAL_SCHEMA` as the sibling result folders (`SR_Score`,
  `MalwareBench_Score`, `MalwareBench_Normalized`, `MalwareBench_Reasoning`, etc.).

## Execution / Usage

Not executable. Reproduce with [`third_model/evaluator_v3.py`](../evaluator_v3.py) using
the Perplexity `sonar` responses as `INPUT_FILE` and the Mistral judge.

## Dependencies

- None (data only). Schema defined by [`third_model/evaluator_v3.py`](../evaluator_v3.py).

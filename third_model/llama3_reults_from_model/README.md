# Folder: third_model / llama3_reults_from_model

## Context & Purpose

Evaluation outputs from the [`third_model`](../README.md) iteration where the **target
model is Llama-3.1-8B** (`llama-3.1-8b-instant`). Results are grouped by which **judge**
scored the responses. Generated artifacts from `evaluator_v3.py`; used to compare judge
agreement on the same target. (Folder name contains the original "reults" spelling.)

## Key Components

- **`huggingface_llama3_1/`** — Llama-3.1-8B responses judged via the HuggingFace judge.
  Files: `EVALUATE_llama-3.1-8b-instant_huggingface_checkpoint_1.csv`,
  `..._checkpoint_2.csv`. These are `evaluator_v3.py` checkpoints with the
  `SR_Score` + `MalwareBench_*` `FINAL_SCHEMA`
  (`row_id, row_hash, target_model, forbidden_prompt, response, attack_method, SR_Score,
  MalwareBench_Score, MalwareBench_Normalized, MalwareBench_Reasoning, timestamp`).
- **`mistral/`** — the same Llama-3.1-8B responses judged via the Mistral judge. File:
  `EVALUATE_llama-3.1-8b-instant_MISTRAL_checkpoint.csv`.

## Execution / Usage

Not executable. Reproduce with [`third_model/evaluator_v3.py`](../evaluator_v3.py) using
the Llama-3.1-8B responses as `INPUT_FILE` and the matching `CURRENT_PROVIDER` judge.

## Dependencies

- None (data only). Schema defined by [`third_model/evaluator_v3.py`](../evaluator_v3.py).

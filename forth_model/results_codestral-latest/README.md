# Folder: forth_model / results_codestral-latest

## Context & Purpose

Per-model workspace for evaluating the Mistral **`codestral-latest`** target in the
[`forth_model`](../README.md) pipeline, kept isolated so cross-model comparisons stay
clean (per the [root README](../../README.md) convention). It holds both the raw response
**batches** (Stage-1 input) and the corresponding **evaluated** outputs (Stage-2), split
into numbered batches (1–5).

## Key Components

- **`responses/`** — raw Stage-1 response batches from
  [`forth_model/model.py`](../model.py), schema `AttackMethod, prompt, Response, status`:
  `responses_results_codestral-latest_1.csv` … `_5.csv` (note the batch-1 filename has a
  leading space). These are the `INPUT_FILE`s for the evaluator.
- **`EVALUATE_codestral-latest_1_groq_llama-3.3_final.csv` … `_5_groq_llama-3.3_final.csv`**
  — Stage-2 evaluated outputs (one per batch), judged by Groq Llama-3.3, in the
  `forth_model` `FINAL_SCHEMA` (`MB_Status`, `MalwareBench_Score`,
  `MalwareBench_Normalized`, VT columns, etc.; see [`CLAUDE.md`](../CLAUDE.md)).
- **`EVALUATE_MISTRAL_codestral_groq_llama-3.1_final.csv`** — an additional evaluated run
  judged by Groq Llama-3.1.

## Execution / Usage

Not executable. Typical use:

```bash
cd forth_model
# Stage 2: point evaluator.py INPUT_FILE at one responses/ batch, then:
python evaluator.py
# Stage 3: analyze all evaluated batches in this folder together
python statistics.py --dir results_codestral-latest
```

## Dependencies

- None (data only). Schemas owned by [`forth_model/model.py`](../model.py) (responses) and
  [`forth_model/evaluator.py`](../evaluator.py) (EVALUATE files); consumed by
  [`forth_model/statistics.py`](../statistics.py).

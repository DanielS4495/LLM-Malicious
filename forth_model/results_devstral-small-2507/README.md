# Folder: forth_model / results_devstral-small-2507

## Context & Purpose

Per-model workspace for evaluating the Mistral **`devstral-small-2507`** target in the
[`forth_model`](../README.md) pipeline, isolated from other models to keep cross-model
comparisons clean (per the [root README](../../README.md) convention). It mirrors the
layout of the sibling [`results_codestral-latest`](../results_codestral-latest/README.md)
folder: raw response **batches** (Stage-1) plus their **evaluated** outputs (Stage-2) in
numbered batches (1–5).

## Key Components

- **`responses/`** — raw Stage-1 response batches from
  [`forth_model/model.py`](../model.py), schema `AttackMethod, prompt, Response, status`:
  `responses_results_devstral-small-2507_1.csv` … `_5.csv`. These are the evaluator
  `INPUT_FILE`s.
- **`EVALUATE_devstral-small-2507_1_groq_llama-3.3_final.csv` … `_5_groq_llama-3.3_final.csv`**
  — Stage-2 evaluated outputs (one per batch), judged by Groq Llama-3.3, in the
  `forth_model` `FINAL_SCHEMA` (`MB_Status`, `MalwareBench_Score`,
  `MalwareBench_Normalized`, VT columns, etc.; see [`CLAUDE.md`](../CLAUDE.md)).

## Execution / Usage

Not executable. Typical use:

```bash
cd forth_model
# Stage 2: point evaluator.py INPUT_FILE at one responses/ batch, then:
python evaluator.py
# Stage 3: analyze all evaluated batches in this folder together
python statistics.py --dir results_devstral-small-2507
```

## Dependencies

- None (data only). Schemas owned by [`forth_model/model.py`](../model.py) (responses) and
  [`forth_model/evaluator.py`](../evaluator.py) (EVALUATE files); consumed by
  [`forth_model/statistics.py`](../statistics.py).

# Folder: first-model

## Context & Purpose

`first-model` is the **first experimental iteration** of the project's evaluation
pipeline. It establishes the core two-stage pattern that every later iteration refines:
**(1) generate** target-model responses to the adversarial prompt set, then **(2) judge**
those responses for refusal/safety. Here the judge is the **StrongReject** rubric run
through Groq-hosted Llama, and the statistics are a simple jailbreak-rate summary.

It connects to the wider **LLM - Malicious** project as the historical baseline: the
generation loop, the `attack_prompts.xlsx` → responses-CSV → evaluated-CSV flow, and the
"jailbreak rate / average safety score" reporting all originate here and are carried
forward (and eventually replaced by the MalwareBench + VirusTotal design) in
[`second model`](../second%20model/README.md), [`third_model`](../third_model/README.md),
and [`forth_model`](../forth_model/README.md). See the [root README](../README.md) for the
overall architecture.

## Key Components

- **`test_llama3_hf.py`** — Earliest generation script. Loads `attack_prompts.xlsx`,
  filters to `AttackMethod == 'Persuative LLM'`, then processes only the first few rows
  (`df_filtered.head()`) through HuggingFace `InferenceClient` with
  `meta-llama/Meta-Llama-3-8B-Instruct`. Writes responses to `responses_results.csv` and
  also writes them back into the source Excel. Prompts the user for `HF_TOKEN` if not in
  the environment. Effectively a smoke test of the generation idea.
- **`model_GPT.py`** — More complete generation script. Loads the same prompt set and
  filter, then sends each prompt to a chat model. Three client back-ends appear (the
  active one is Perplexity `sonar`; HuggingFace Llama-3-70B and Groq `llama3-70b-8192`
  are present but commented out). Cleans each prompt (strips brackets/quotes/escapes),
  appends successful responses to `responses_results.csv` with columns
  `AttackMethod, prompt, Response`, and sleeps 2s between calls. Only successful rows are
  saved (errors are printed, not stored).
- **`evaluator.py`** — The **judge** stage. Renames the response CSV columns to
  StrongReject's expected schema (`forbidden_prompt`, `response`, `attack_method`), builds
  a HuggingFace `datasets.Dataset`, and calls `strong_reject.evaluate.evaluate_dataset`
  with the `strongreject_rubric` using judge model `openai/llama-3.1-8b-instant` pointed
  at Groq's OpenAI-compatible endpoint. Processes one row at a time with resume support
  via `eval_checkpoint.csv` (keyed on `row_id`), then merges results into
  `responses_results_evaluated_preplexity_sonar.csv`. `run_statistic()` computes the
  overall **jailbreak rate** (rows where `refusal == 0`), the mean safety `score`, and
  per-`attack_method` effectiveness.

### Data files (generated artifacts)

- `responses_results.csv` — raw target-model responses.
- `responses_results_evaluated.csv`, `responses_results_evaluated_5.csv`,
  `responses_results_evaluated.xlsx` — StrongReject-scored outputs.
- `responses_results_preplexity_sonar.csv` — Perplexity `sonar` responses.
- `attack_prompts.xlsx` — local copy of the curated prompt set (see
  [`Collection_of_prompt`](../Collection_of_prompt/README.md)).

## Execution / Usage

Run directly from inside this folder (scripts use relative filenames):

```bash
# 1. Generate responses (edit the active client in model_GPT.py first)
python model_GPT.py

# 2. Judge the responses with StrongReject and print jailbreak stats
python evaluator.py
```

`evaluator.py`'s `__main__` runs `run_evaluator()` then `run_statistic()`. Requires a
`.env` / environment with `GROQ_API_KEY` (judge) and `PPLX_API_KEY` / `HF_TOKEN`
(generation, depending on the active provider).

## Dependencies

- **Python packages:** `pandas`, `openpyxl`, `python-dotenv`, `openai`, `groq`,
  `huggingface_hub`, `datasets`, and `strong_reject`
  (`git+https://github.com/dsbowen/strong_reject.git`).
- **Project dependencies:** reads the local `attack_prompts.xlsx` (curated from
  [`Collection_of_prompt`](../Collection_of_prompt/README.md)). Superseded by later
  iterations; not imported by any other folder.

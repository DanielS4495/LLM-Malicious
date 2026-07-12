# Folder: forth_model / evaluation

## Context & Purpose

Curated gallery of **final evaluated result files** for the [`forth_model`](../README.md)
pipeline — one `EVALUATE_*_final.csv` per target model. These are the Stage-2 outputs of
`evaluator.py`/`evaluator_2.py` and the typical **input directory for Stage 3**: running
`python statistics.py --dir evaluation` loads every CSV here (one file = one model) and
produces the cross-model comparison report. This is the folder that backs the project's
headline "compare models" analysis.

## Key Components

Each file is a completed evaluation in the `forth_model` `FINAL_SCHEMA` (see
[`CLAUDE.md`](../CLAUDE.md)): `row_id, row_hash, target_model, forbidden_prompt, response,
attack_method, MB_Status, MalwareBench_Score, MalwareBench_Normalized,
MalwareBench_Reasoning, VT_Status, Web_Link, VT_Verdict, Malicious_Count, File_Type, Tags,
Sigma_Hits, MITRE_Techniques, YARA_Rules, Threat_Category, Threat_Label, Engines_List,
timestamp`. `statistics.py` derives each model's name from the filename after the
`EVALUATE_` prefix.

- `EVALUATE_codestral-latest_groq_llama-3.3_final.csv` — target Mistral **codestral**,
  judged by Groq Llama-3.3.
- `EVALUATE_MISTRAL_small-latest_groq_llama-3.1_final.csv` — target Mistral small-latest,
  judged by Groq Llama-3.1.
- `EVALUATE_mistral-small-latest_groq_llama-3.3_final.csv` — target Mistral small-latest,
  judged by Groq Llama-3.3.
- `EVALUATE_llama-3.1-8b-instant_groq_llama-3.3_final.csv` — target Groq Llama-3.1-8B.
- `EVALUATE_meta-llama_Meta-Llama-3.1-70B-Instruct_groq_llama-3.3_final.csv` — target
  HuggingFace Llama-3.1-70B.
- `EVALUATE_qwen_qwen3-32b_groq_llama-3.3_final.csv` — target Qwen3-32B.

## Execution / Usage

Not executable. Consumed by Stage 3:

```bash
cd forth_model
python statistics.py --dir evaluation
```

Reproduce individual files by running [`forth_model/evaluator.py`](../evaluator.py) with
the matching `INPUT_FILE` (target responses) and `CURRENT_PROVIDER` (judge).

## Dependencies

- None of its own (data). Schema owned by [`forth_model/evaluator.py`](../evaluator.py);
  consumed by [`forth_model/statistics.py`](../statistics.py).

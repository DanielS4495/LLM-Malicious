# Folder: second model

## Context & Purpose

`second model` is the **second experimental iteration**. It generalizes the generation
stage into a provider **registry** (so any of several LLMs can be the target with a
one-line change), keeps **StrongReject** as the judge (now with multiple selectable judge
back-ends and a failed-row retry mode), and — most importantly — **introduces malware
scanning of the generated code** via VirusTotal and OPSWAT MetaDefender. That scanning
idea becomes the central feature of [`third_model`](../third_model/README.md) and the
mature idempotent VirusTotal state machine in [`forth_model`](../forth_model/README.md).

Within the overall **LLM - Malicious** project (see [root README](../README.md)) this
iteration covers all three stages — generation, judging, and a first attempt at
"dynamic" (AV-based) evaluation of whether the produced code is actually detected as
malicious.

## Key Components

- **`model.py`** — Generation stage. Defines a `MODEL_REGISTRY` mapping keys
  (`perplexity`, `groq`, `huggingface`, `chatgpt`) to a model name + a provider function,
  and selects one via `ACTIVE_MODEL`. A `SYSTEM_INSTRUCTION` forces raw, executable,
  single-language code output (natural-language notes must be code comments). Loads
  `attack_prompts.xlsx`, filters to `AttackMethod == 'Persuative LLM'`, and appends each
  response to `responses_results_<active>_<model>.csv` (`AttackMethod, prompt, Response`),
  sleeping 2s between calls.
- **`evaluator.py`** — Judge stage built on **StrongReject** (`strongreject_rubric`) via
  LiteLLM/`datasets`. `configure_judge_environment()` selects the judge back-end
  (`ACTIVE_JUDGE`): perplexity, groq, gemini, huggingface, openrouter, or **poe**
  (`fastapi_poe`). Processes one row at a time with `eval_checkpoint.csv` resume, merges
  into an evaluated CSV, and `run_statistic()` reports jailbreak rate (`refusal == 0`),
  mean `score`, and per-attack-method effectiveness. A `--retry` mode
  (`retry_failed_rows()`) reprocesses checkpoint rows with missing `score`/`refusal`.
- **`scan_virustotal.py`** — **Dynamic (AV) evaluation.** For each generated response it
  extracts the code, computes its SHA256, and queries VirusTotal `GET /v3/files/{hash}`;
  if unknown, uploads via `POST /v3/files` and polls `/v3/analyses/{id}` until complete.
  `parse_vt_response()` extracts verdict, malicious engine count, file type, tags, Sigma
  hits, MITRE ATT&CK techniques, YARA rules, reputation, threat category/label, and the
  detecting engines. Refusals or code shorter than 15 chars are skipped. Resumes by row
  count and writes `virustotal_scan_results_*.csv`. Requires `VT_API_KEY`.
- **`scan_metadefender.py`** — Alternative AV scanner using OPSWAT MetaDefender
  (`api.metadefender.com/v4`). `detect_extension()` guesses the language to name the
  uploaded file, uploads with `rule=multiscan`, polls `/v4/file/{data_id}` to 100%
  progress, and `analyze_opswat_response()` derives verdict, threat name, and detected
  engine count. Same skip/resume logic. Requires `OPSWAT_API_KEY`.

### Data files

- `attack_prompts.xlsx` — local copy of the curated prompt set.
- `responses_results_perplexity_sonar.csv` — raw target responses (evaluator input).
- `eval_checkpoint.csv` — StrongReject per-row checkpoint.
- `results/` — collected evaluated and scan output CSVs (see
  [`results/README.md`](results/README.md)).

## Execution / Usage

Run each stage in order from inside this folder:

```bash
# 1. Generate target-model responses (set ACTIVE_MODEL in model.py)
python model.py

# 2. Judge responses with StrongReject (set ACTIVE_JUDGE in evaluator.py)
python evaluator.py            # normal mode
python evaluator.py --retry    # reprocess only failed/missing rows

# 3. (Optional) Scan the generated code with an AV back-end
python scan_virustotal.py      # needs VT_API_KEY
python scan_metadefender.py    # needs OPSWAT_API_KEY
```

`INPUT_FILE`/output filenames are hard-coded constants near the top of each script — edit
them to match the model you generated for.

## Dependencies

- **Python packages:** `pandas`, `openpyxl`, `python-dotenv`, `openai`, `groq`,
  `huggingface_hub`, `datasets`, `strong_reject`, `fastapi_poe`, `requests`.
- **Environment variables:** `HF_TOKEN`, `GROQ_API_KEY`, `PPLX_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `POE_API_KEY`, `VT_API_KEY`,
  `OPSWAT_API_KEY` (only those for the providers you actually use).
- **Project dependencies:** reads `attack_prompts.xlsx` (from
  [`Collection_of_prompt`](../Collection_of_prompt/README.md)); the scan scripts are
  copied verbatim into [`third_model`](../third_model/README.md).

# Folder: forth_model

## Context & Purpose

`forth_model` is the **current, consolidated pipeline** — the production version of the
project that merges the successful pieces of the three earlier iterations into one clean
three-stage workflow. Relative to its predecessors it: replaces the 1–5 judge with a
richer **7-category MalwareBench-style rubric** producing a weighted composite risk score
(0–10); integrates **VirusTotal natively** as an idempotent, resumable state machine
(no separate scanner scripts, no MetaDefender); **retires StrongReject** entirely; and
adds a **multi-layer statistical analyzer** that emits CSV tables, plots, and an HTML
report.

This folder is what the [root README](../README.md) points to as the canonical pipeline.
Its own design contract and invariants are documented in detail in
[`CLAUDE.md`](CLAUDE.md), which should be treated as the authoritative spec for the schema
and the evaluator's run modes. The three stages implement the project-wide vocabulary:
**Generation → Evaluation/Judging → Statistics**.

## Key Components

- **`model.py`** — **Stage 1, Generation.** `MODEL_REGISTRY` maps keys to (model name,
  provider function) for Perplexity, Groq, HuggingFace, OpenRouter, OpenAI, and Mistral;
  `ACTIVE_MODEL` selects one target. A strict `SYSTEM_INSTRUCTION` forces raw, executable,
  comment-free single-file code. Reads `attack_prompts.xlsx`, filters
  `AttackMethod == 'Persuative LLM'`, and appends rows `(AttackMethod, prompt, Response,
  status)` to `responses_results_<safe_model>.csv`. Skips already-completed prompts
  (resume) and marks transport failures `status=FAILED` (re-processed next run).
  `openrouter_provider` has a placeholder `messages=[...]`.
- **`evaluator.py`** — **Stage 2, Judging (primary).** LiteLLM-based. `CURRENT_PROVIDER`
  selects the judge (Groq Llama-3.1/3.3/4-Scout, Gemini, Perplexity, HuggingFace, GPT,
  OpenRouter, Ollama, Mistral). `MalwareBenchEvaluator` sends the response to the judge
  with the 7-category rubric and parses the `COMPOSITE RISK SCORE x/10` and executive
  summary. VirusTotal is handled inline: code is extracted, SHA256-hashed (used only as a
  lookup key), checked via `GET /v3/files/{hash}`, uploaded via `POST /v3/files` if
  unknown, and results parsed into VT columns. An idempotent **state machine**
  (`classify_row` → COMPLETE / PENDING_VT / INCOMPLETE) with **Mode 1 (fresh)** and
  **Mode 2 (resume/repair)** guarantees safe re-runs; every row is saved immediately, HTTP
  429 stops VT for the run, and a 3-attempt limit applies. Output:
  `EVALUATE_<model>_<provider>_final.csv` in `FINAL_SCHEMA` order. Full rules in
  [`CLAUDE.md`](CLAUDE.md).
- **`evaluator_2.py`** — **Stage 2, secondary/advanced pipeline.** Same schema and state
  machine as `evaluator.py` plus: **Groq and VT API-key rotation** on 429
  (`GROQ_API_KEYS` / `VT_API_KEYS`, comma-separated), a **local VT cache** that copies
  results between rows sharing a code hash (no duplicate API calls), MB `"error"` treated
  as **retryable** (not terminal), and VT **skipped** for `refusal`/`error` rows. The
  documented FIX 1–FIX 8 in its `run_pipeline()` docstring enumerate these behaviors.
- **`statistics.py`** — **Stage 3, Analysis.** CLI tool (`--dir`, `--benchmark`,
  `--threshold` default 0.5, `--output` default `stats_output`) that auto-discovers all
  `*.csv` in a directory (one file = one model, name derived from the `EVALUATE_` stem),
  loads them through `DataLoader` (normalizes columns, 85%-response completeness filter,
  derived token/label columns), then runs layered analytics. Layers are implemented as
  `layer1..layer15` functions (descriptive stats, binary/refusal rates, MB-vs-VT
  agreement, distributions, segmentation by attack method, token-vs-score, stability,
  correlation, drift, entropy, error taxonomy, max risk) plus model/benchmark comparison,
  a summary dashboard, and an HTML report. The active `main()` runs a curated subset
  (L1, L3, L4 histogram, L6, L8, L15, model comparison, dashboard, report); other layers
  remain defined but pruned from the run. Refusal rows (`MB_Status="refusal"`, score 0)
  are always included. Token counts use a `len/4` character approximation.
- **`CLAUDE.md`** — Authoritative design spec: data flow, run modes, `FINAL_SCHEMA`, and
  the 10 mandatory project rules (row completeness, score independence, VT integration,
  idempotency, retired features such as `AV_Scan_ID`/`SR_Score`/MetaDefender, etc.).
- **`requirements.txt`** — Pinned dependency list for the whole project (UTF-16 encoded).
- **`attack_prompts.xlsx`** — local copy of the curated prompt set (Stage-1 input).
- **`non_ascii_chars.txt`** — supporting text asset (non-ASCII character reference).

### Result / output subfolders

- **`evaluation/`** — a curated gallery of final per-model `EVALUATE_*_final.csv` files
  (the inputs typically fed to `statistics.py`). See [`evaluation/README.md`](evaluation/README.md).
- **`results/`** — raw Stage-1 response CSVs (evaluator inputs). See
  [`results/README.md`](results/README.md).
- **`results_codestral-latest/`** — per-batch evaluation of the Mistral **codestral**
  target, with a `responses/` subfolder of raw batches. See
  [`results_codestral-latest/README.md`](results_codestral-latest/README.md).
- **`results_devstral-small-2507/`** — per-batch evaluation of the Mistral **devstral**
  target, with a `responses/` subfolder. See
  [`results_devstral-small-2507/README.md`](results_devstral-small-2507/README.md).
- **`stats_output/`** — referenced in [`CLAUDE.md`](CLAUDE.md) as the generated
  statistics destination (CSV tables, `plots/`, `report.html`). Do not edit by hand;
  regenerate via `statistics.py`.

## Execution / Usage

Run from inside `forth_model` (paths are relative). Full flow:

```bash
# Stage 1 — generate target responses (set ACTIVE_MODEL in model.py)
python model.py

# Stage 2 — judge responses + VirusTotal (set CURRENT_PROVIDER + INPUT_FILE in evaluator.py)
python evaluator.py
# re-run until every row shows VT_Status="complete" (VT is asynchronous)
# or use the key-rotating / cached variant:
# python evaluator_2.py

# Stage 3 — statistics + HTML report over a folder of EVALUATE_*.csv files
python statistics.py --dir evaluation
python statistics.py --dir results_codestral-latest --threshold 0.4 --output my_report
```

## Dependencies

- **Python packages** (see [`requirements.txt`](requirements.txt)): `pandas`, `numpy`,
  `openpyxl`, `python-dotenv`, `requests`, `litellm`, `openai`, `groq`,
  `huggingface_hub`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`. `statistics.py`
  degrades gracefully if matplotlib/scipy/sklearn are missing. (Note: `strong_reject` is
  still pinned in requirements but is **retired** in this iteration per
  [`CLAUDE.md`](CLAUDE.md).)
- **Environment variables** (`.env`): `HF_TOKEN`, `GROQ_API_KEY` (or `GROQ_API_KEYS`),
  `PPLX_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`,
  `GEMINI_API_KEY`, `VT_API_KEY` (or `VT_API_KEYS`).
- **Project dependencies:** reads `attack_prompts.xlsx` (from
  [`Collection_of_prompt`](../Collection_of_prompt/README.md)). Self-contained otherwise;
  it supersedes [`first-model`](../first-model/README.md),
  [`second model`](../second%20model/README.md), and
  [`third_model`](../third_model/README.md).

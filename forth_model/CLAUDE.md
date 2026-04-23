# CLUADE.MD

## Project Overview
This project evaluates how a target code-generation model responds to adversarial prompt-injection attack prompts, then analyzes risk statistically.

The workflow is split into 3 main stages:
1. Generate model responses (`model.py`)
2. Evaluate responses with safety/risk evaluators (`evaluator.py`)
3. Produce multi-layer statistical analysis and report (`statistics.py`)

---

## Repository Structure
- `model.py`: Sends prompts to a selected provider/model and writes raw responses to CSV.
- `evaluator.py`: Scores each response using a MalwareBench-style rubric and VirusTotal via a two-mode idempotent pipeline (Mode 1: first run, Mode 2: resume/repair).
- `statistics.py`: Runs layered analytics and generates CSV summaries, plots, and HTML report.
- `EVALUATE_MISTRAL_codestral_MISTRAL_final.csv`: Evaluated dataset output (input for statistics).
- `stats_output/`: Generated analysis artifacts (CSV tables, figures, report).

---

## End-to-End Data Flow
1. Input prompts are read from `attack_prompts.xlsx` inside `model.py`.
2. Responses are generated and saved to `responses_results_<model>.csv`.
3. `evaluator.py` loads those responses and populates:
   - `MB_Status`, `MalwareBench_Score`, `MalwareBench_Normalized`, `MalwareBench_Reasoning`
   - VirusTotal columns (`VT_Status`, `Web_Link`, `VT_Verdict`, `Malicious_Count`, `File_Type`, `Tags`, `Sigma_Hits`, `MITRE_Techniques`, `YARA_Rules`, `Threat_Category`, `Threat_Label`, `Engines_List`)
4. Evaluated rows are saved to:
   - final CSV (`EVALUATE_<...>_final.csv`)
5. `statistics.py` reads one or more evaluation CSVs and generates:
   - layer outputs (`L1`, `L2`, ..., `L15`)
   - comparison files (`MC_*`, `BENCH_*`)
   - plots in `stats_output/plots/`
   - `stats_output/report.html`

---

## Stage Details

### 1) Response Generation (`model.py`)
- Loads API keys from `.env`.
- Uses a strict system instruction forcing raw executable code output.
- Supports multiple providers via `MODEL_REGISTRY`.
- Uses `ACTIVE_MODEL` to pick exactly one provider/model pair.
- Filters prompts by `AttackMethod == 'Persuative LLM'`.
- Writes each row to `responses_results_<safe_model_name>.csv`.
- Skips prompts already present in output CSV (resume behavior).

Important note:
- `openrouter_provider` currently uses `messages=[...]` as a placeholder in the code.

### 2) Evaluation (`evaluator.py`)
- Selects an evaluator provider using `CURRENT_PROVIDER`.
- Configures environment variables dynamically in `configure_environment()`.
- **MalwareBench failure after 3 attempts** produces `MalwareBench_Score=0`, `MalwareBench_Normalized=0`, `MB_Status="refusal"`. This is a valid, meaningful result — not an error. These rows are included in all statistical calculations. A refusal row with VT complete is classified as COMPLETE and never retried.
- **VT polling** on pending rows re-derives the SHA256 hash from the response text and queries `GET /v3/files/{hash}` directly. `AV_Scan_ID` is fully retired and removed from the schema.
- Detects run mode at startup by checking whether the output CSV exists and contains rows:
  - **Mode 1 — First Run** (output absent or empty): For each input row, run MB evaluation → extract code → SHA256 hash-check VT → upload if not found. Does not wait for VT analysis to complete; saves row immediately.
  - **Mode 2 — Resume Run** (output has rows): Scans existing rows in order and repairs only what is missing:
    - INCOMPLETE rows: re-run MB evaluation using the existing response. Never calls the target LLM again.
    - PENDING_VT rows: re-derive SHA256 from response and poll `GET /v3/files/{hash}`. HTTP 429 stops all VT polling for the run (sleep 60 s, save, exit polling loop).
    - After repairing existing rows, processes any input rows not yet in the output (same logic as Mode 1).
- Uses row hashing (`row_hash`) and `classify_row()` for state-based resume. `classify_row()` treats `MB_Status="refusal"` as a valid completed MB state, equivalent to having numeric scores.
- Writes output immediately after every row in both modes.

### 3) Statistics (`statistics.py`)
- CLI-style script with arguments:
  - `--files` (required)
  - `--benchmark` (optional)
  - `--threshold` (default 0.5)
  - `--output` (default `stats_output`)
- Uses `VT_Status` (not `AV_Status`) and `Malicious_Count` (not `AV_Score`) throughout.
- Refusal rows (`MB_Status="refusal"`, `MalwareBench_Score=0`, `MalwareBench_Normalized=0`) are included in all statistical calculations — never filtered out.
- Performs layered analysis:
  - L1: Descriptive statistics
  - L2: Binary success/failure rates — includes **Refusal Rate** metric (count of `MB_Status="refusal"` / total rows)
  - L3: Agreement analysis (MalwareBench vs VT `Malicious_Count`)
  - L4: Score distributions — histograms include a visible "Refusals" annotation at x=0
  - L5: Segmentation by attack method
  - L6: Token counts vs score
  - L7: Stability / robustness
  - L8: Correlation matrices
  - L12: Drift over time
  - L13: Entropy / uncertainty
  - L14: Error taxonomy — **Refusal** is an explicit category (rows where `MB_Status="refusal"`). Rows with missing scores are "Missing/Error".
  - L15: Max risk + top worst cases
- Also supports model-to-model and benchmark comparisons.
- Generates dashboard image and HTML report.

---

## Environment Variables
Expected in `.env` (depending on provider choice):
- `HF_TOKEN`
- `GROQ_API_KEY`
- `PPLX_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY`
- `VT_API_KEY`

---

## Critical Project Rules
1. Do not manually edit files under `stats_output/`.
   - These are generated artifacts and should be recreated by rerunning code.
2. Do not change model identifier strings in code unless intentionally switching experiments.
   - Keep exact naming/format for model keys and provider model names.
3. If output format needs to change, change generator code (in `evaluator.py` / `statistics.py`), then regenerate outputs.

---

## Typical Run Order
1. Run `model.py` to generate raw responses.
2. Run `evaluator.py` to score responses into final evaluation CSV.
3. Run `statistics.py --files <final_csv>` to generate full statistical report.

---

## Practical Notes
- Evaluator uses delays (`time.sleep`) to reduce rate-limit errors.
- Resume/checkpoint logic exists in both response generation and evaluation stages.
- Statistical pipeline is resilient to missing optional libraries (plots/advanced metrics are skipped if package missing).

---

## Current Schema

`FINAL_SCHEMA` column order (as defined in `evaluator.py`):

```
row_id, row_hash, target_model, forbidden_prompt, response,
attack_method, MB_Status, MalwareBench_Score, MalwareBench_Normalized,
MalwareBench_Reasoning, VT_Status, Web_Link,
VT_Verdict, Malicious_Count, File_Type, Tags,
Sigma_Hits, MITRE_Techniques, YARA_Rules,
Threat_Category, Threat_Label, Engines_List, timestamp
```

Key schema notes:
- `MB_Status` appears **before** `MalwareBench_Score`. Valid values: `"ok"`, `"refusal"`.
- `VT_Status` replaces the retired `AV_Status`. Valid values: `"complete"`, `"pending"`, `"error"`.
- `AV_Scan_ID`, `Reputation`, and `Saferpickle` are removed entirely.

---

## Mandatory Project Rules

### Rule 1 — Row Completeness
A row is complete when `MalwareBench_Score`, `MalwareBench_Normalized`, and `MalwareBench_Reasoning` are all populated (non-null, non-NA, non-empty-string), **or** when `MB_Status="refusal"` (which always sets Score=0, Normalized=0, and a fixed reasoning string). VT columns (`VT_Status`, `VT_Verdict`, etc.) may be pending or skipped — this does not make the row incomplete. A row with complete MalwareBench scores and `VT_Status="pending"` is valid and will have its VT columns resolved on a subsequent run.

### Rule 2 — Score Independence
`MalwareBench_*` scores and VT columns (`Malicious_Count`, `VT_Verdict`) come from independent evaluation systems with different methodologies. Never derive, map, substitute, or infer one from the other. Correlation analysis between them is permitted as informational only (see Layer 8). Layer 3 must use agreement analysis, not a confusion matrix framing.

### Rule 3 — VirusTotal Integration
VirusTotal is handled natively inside `evaluator.py` via an idempotent state machine (COMPLETE / PENDING_VT / INCOMPLETE). `av_poller.py` and `scan_virustotal.py` are retired and must not be used or recreated. PENDING_VT rows are polled by re-deriving the SHA256 hash from the response text and querying `GET /v3/files/{hash}` directly. New rows are submitted via `POST /v3/files`. VT columns (`VT_Status`, `Web_Link`, `VT_Verdict`, `Malicious_Count`, `File_Type`, `Tags`, `Sigma_Hits`, `MITRE_Techniques`, `YARA_Rules`, `Threat_Category`, `Threat_Label`, `Engines_List`) are updated in-place. The primary VT metric is `Malicious_Count` (integer 0–70+).

### Rule 4 — Token Count Approximation
Token counts in Layer 6 use a character-based approximation (`len(text) / 4`) suited for code-heavy content. The column is named `response_char_tokens_approx` to make this explicit. Do not substitute word-count proxies or NLP tokenizers for this column.

### Rule 5 — SR_Score Retired
`SR_Score` (StrongReject) is fully retired. It must not appear in any schema, column reference, or analysis. The `StrongRejectEvaluator` class and all imports from `strong_reject` have been removed from `evaluator.py`.

### Rule 6 — Idempotent Execution
`evaluator.py` can be run multiple times safely. Mode 1 processes all input rows from scratch. Mode 2 repairs INCOMPLETE rows, polls PENDING_VT rows, then appends any new input rows not yet in the checkpoint. No row is re-processed unnecessarily. COMPLETE rows are never touched.

### Rule 7 — In-Place Update Rule
When repairing or updating an existing checkpoint row (Mode 2), updates must be made strictly in-place using `df.at[idx, col] = val` for each changed column, followed immediately by `save_checkpoint(df)` to overwrite the full file. Never remove and re-append a repaired row. The row count in the checkpoint must never increase during a repair pass — only during the new-row pass at the end of Mode 2.

### Rule 8 — 3-Attempt Limit
Every API call in the evaluator (MalwareBench evaluation or VT polling) has a strict maximum of 3 attempts per row per run. If all 3 MB attempts fail, the row receives `MalwareBench_Score=0`, `MalwareBench_Normalized=0`, `MB_Status="refusal"` — this is a valid completed state, not an error, and the row is never retried. If VT polling fails 3 times, `VT_Status` is left as `"pending"` for the next run. Infinite retry loops are forbidden. HTTP 429 responses are not counted toward the 3-attempt limit — they are quota events and trigger an immediate stop of VT polling for the run, not a per-row retry.

### Rule 9 — VT Polling via SHA256
`AV_Scan_ID` is fully retired and removed from the schema. VT polling on `PENDING_VT` rows re-derives the SHA256 hash from the response text at poll time and queries `GET /v3/files/{hash}` directly. The SHA256 hash is used only as an internal lookup key and is never stored in any column.
- **Existing report path** (`GET /v3/files/{hash}` returns 200): VT result columns are populated immediately and `VT_Status` is set to `"complete"`.
- **Upload path** (`GET /v3/files/{hash}` returns 404): code is uploaded via `POST /v3/files` and `VT_Status` is set to `"pending"`. The analysis ID returned by the upload is used only transiently and is never written to any column.

### Rule 10 — Retired Features
The following features are fully retired and must not appear in any schema, column reference, code, or analysis:
- `AV_Scan_ID`: removed. VT polling uses re-derived SHA256 hash via `GET /v3/files/{hash}`.
- `Reputation`: removed from schema and VT parser.
- `Saferpickle`: removed from schema and VT parser.
- `AV_Status`: renamed to `VT_Status`. Any legacy CSV with `AV_Status` is migrated on load.
- `AV_Score`: retired. The primary VT metric is `Malicious_Count`.
- `SR_Score` / `StrongReject`: retired. All evaluator and statistics references removed.
- `MB_Status="error"`: retired. MalwareBench failures now produce `MB_Status="refusal"` with `Score=0` instead of `pd.NA`.
- `av_poller.py` / `scan_virustotal.py`: retired scripts. All VT logic is inside `evaluator.py`.

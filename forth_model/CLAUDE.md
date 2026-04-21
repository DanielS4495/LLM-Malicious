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
- `evaluator.py`: Scores each response using a MalwareBench-style rubric and VirusTotal via a 3-pass idempotent state machine.
- `statistics.py`: Runs layered analytics and generates CSV summaries, plots, and HTML report.
- `EVALUATE_MISTRAL_codestral_MISTRAL_final.csv`: Evaluated dataset output (input for statistics).
- `stats_output/`: Generated analysis artifacts (CSV tables, figures, report).

---

## End-to-End Data Flow
1. Input prompts are read from `attack_prompts.xlsx` inside `model.py`.
2. Responses are generated and saved to `responses_results_<model>.csv`.
3. `evaluator.py` loads those responses and populates:
   - `MalwareBench_Score`
   - `MalwareBench_Normalized`
   - `MalwareBench_Reasoning`
   - VirusTotal columns (`AV_Scan_ID`, `AV_Status`, `Web_Link`, `VT_Verdict`, `Malicious_Count`, etc.)
4. Evaluated rows are saved to:
   - checkpoint CSV (`EVALUATE_<...>_checkpoint.csv`)
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
- Runs a 3-pass idempotent pipeline on every execution:
  - **Pass 1** — Gap Filling: re-processes all INCOMPLETE rows and polls PENDING_VT rows.
  - **Pass 2** — New Rows: processes input rows not yet in the checkpoint.
  - **Pass 3** — Final Verification: prints state counts and writes the final CSV.
- Uses row hashing (`row_hash`) and `classify_row()` for state-based resume.
- Writes checkpoint immediately after every row in all passes.

### 3) Statistics (`statistics.py`)
- CLI-style script with arguments:
  - `--files` (required)
  - `--benchmark` (optional)
  - `--threshold` (default 0.5)
  - `--output` (default `stats_output`)
- Performs layered analysis:
  - L1: Descriptive statistics
  - L2: Binary success/failure rates
  - L3: Agreement analysis (MalwareBench vs VT)
  - L4: Threshold sensitivity and ROC
  - L5: Segmentation by attack method
  - L6: Token counts vs score
  - L7: Stability / robustness
  - L8: Correlation matrices
  - L12: Drift over time
  - L13: Entropy / uncertainty
  - L14: Error taxonomy
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

## Mandatory Project Rules

### Rule 1 — Row Completeness
A row is complete when `MalwareBench_Score`, `MalwareBench_Normalized`, and `MalwareBench_Reasoning` are all populated (non-null, non-NA, non-empty-string). VT columns (`AV_Status`, `VT_Verdict`, etc.) may be pending or skipped — this does not make the row incomplete. A row with complete MalwareBench scores and `AV_Status="pending"` is valid and will have its VT columns resolved on a subsequent run.

### Rule 2 — Score Independence
`MalwareBench_*` scores and VT columns (`Malicious_Count`, `VT_Verdict`) come from independent evaluation systems with different methodologies. Never derive, map, substitute, or infer one from the other. Correlation analysis between them is permitted as informational only (see Layer 8). Layer 3 must use agreement analysis, not a confusion matrix framing.

### Rule 3 — VirusTotal Integration
VirusTotal is handled natively inside `evaluator.py` via a 3-pass idempotent state machine (COMPLETE / PENDING_VT / INCOMPLETE). `av_poller.py` and `scan_virustotal.py` are retired and must not be used or recreated. Each row is classified on every run; PENDING_VT rows are polled in Pass 1, and new rows are submitted in Pass 2. VT columns (`AV_Scan_ID`, `AV_Status`, `Web_Link`, `VT_Verdict`, `Malicious_Count`, `Saferpickle`, `File_Type`, `Tags`, `Sigma_Hits`, `MITRE_Techniques`, `Reputation`, `YARA_Rules`, `Threat_Category`, `Threat_Label`, `Engines_List`) are updated in-place. The primary VT metric is `Malicious_Count` (integer 0–70+). `AV_Score` is retired.

### Rule 4 — Token Count Approximation
Token counts in Layer 6 use a character-based approximation (`len(text) / 4`) suited for code-heavy content. The column is named `response_char_tokens_approx` to make this explicit. Do not substitute word-count proxies or NLP tokenizers for this column.

### Rule 5 — SR_Score Retired
`SR_Score` (StrongReject) is fully retired. It must not appear in any schema, column reference, or analysis. The `StrongRejectEvaluator` class and all imports from `strong_reject` have been removed from `evaluator.py`.

### Rule 6 — Idempotent Execution
`evaluator.py` can be run multiple times safely. Each run performs 3 passes: fix incomplete rows (Pass 1), poll pending VT rows (Pass 1), process new rows (Pass 2), verify final state (Pass 3). No row is re-processed unnecessarily. COMPLETE rows are never touched.

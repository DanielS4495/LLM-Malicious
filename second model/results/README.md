# Folder: second model / results

## Context & Purpose

Output store for the [`second model`](../README.md) iteration. It collects the **judged**
(StrongReject) and **AV-scanned** (VirusTotal) result CSVs produced by that iteration's
`evaluator.py` and `scan_virustotal.py`. These are generated artifacts — snapshots of
experiment runs — not source code, and serve as the evidence base for the jailbreak-rate
and consistency analysis of the second iteration.

## Key Components

All files are CSV outputs (regenerate by re-running the parent scripts):

- **`responses_results_evaluated.csv`** — StrongReject-judged responses. Columns include
  `row_id, attack_method, forbidden_prompt, response, jailbroken_prompt, refusal,
  convincingness, specificity, judge_model, score, evaluator`.
- **`responses_results_evaluated_new.csv`** — a later/alternative evaluated run.
- **`responses_results_evaluated_gemini_2.5.csv`** — evaluated using the Gemini 2.5 judge
  back-end.
- **`responses_results_evaluated_preplexity_sonar.csv`** — evaluated Perplexity `sonar`
  target responses.
- **`responses_results_huggingface_meta-llama-Meta-Llama-3-70B-Instruct.csv`** — raw
  responses from the HuggingFace Llama-3-70B target.
- **`eval_checkpoint.csv`** — per-row StrongReject checkpoint used for resume.
- **`virustotal_scan_results_groq-3.1_llama-3.1-8b-instant.csv`** — VirusTotal scan output
  for the Groq Llama-3.1-8B target (verdict, malicious count, Sigma/MITRE/YARA, etc.).

## Execution / Usage

Not executable. Consumed by inspection and by the parent iteration's statistics routines.
To reproduce, run the scripts documented in [`second model`](../README.md) with the
matching `INPUT_FILE`/output constants.

## Dependencies

- No code dependencies. Schema is defined by the producing scripts
  ([`second model/evaluator.py`](../evaluator.py),
  [`second model/scan_virustotal.py`](../scan_virustotal.py)).

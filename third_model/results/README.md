# Folder: third_model / results

## Context & Purpose

Primary output store for the [`third_model`](../README.md) iteration. It holds the
evaluated (StrongReject + MalwareBench) result CSVs, a VirusTotal scan output, and a large
**archive of earlier judge experiments** kept for comparison. These are generated
artifacts documenting which judge back-end produced which scores; they are the evidence
base for that iteration's consistency analysis.

## Key Components

Top-level files:

- **`responses_results_groq-3.1_llama-3.1-8b-instant_evaluated_malwarebench.csv`** — the
  Groq Llama-3.1-8B target responses scored with the MalwareBench rubric. Columns include
  `target_model, forbidden_prompt, response, SR_Score, SR_Jailbreak_Score,
  MalwareBench_Score, MalwareBench_Normalized, MalwareBench_Reasoning, attack_method,
  row_hash, timestamp, SR_Refusal, MalwareBench_Is_Refusal`.
- **`responses_results_huggingface_meta-llama-Meta-Llama-3-70B-Instruct.csv`** — raw
  HuggingFace Llama-3-70B target responses.
- **`virustotal_scan_results_groq-3.1_llama-3.1-8b-instant.csv`** — VirusTotal scan output
  for the Groq Llama-3.1-8B target.

### `old_evaluations_huggingface/`

An archive of the same HuggingFace-target responses re-judged by **different judge
back-ends**, one subfolder per judge, each holding `EVALUATE_..._checkpoint*.csv`
(row-by-row checkpoints from `evaluator_v3.py`) and sometimes a `*_direct.log`:

- **`gemini/`** — judged by Gemini.
- **`groq_31_evaluation/`** — judged by Groq Llama-3.1.
- **`groq_33_evaluation/`** — judged by Groq Llama-3.3 (+ run log).
- **`groq_llama_4_scout/`** — judged by Groq Llama-4-Scout.
- **`hugging_face_evaluation/`** — judged by the HuggingFace endpoint (+ run log).
- **`ollama/`** — judged by a local Ollama model.
- **`openai_gpt/`** — judged by an OpenAI GPT model.

The folder also contains earlier merged evaluated CSVs
(`responses_results_evaluated*.csv`, incl. `_gemini_2.5`, `_new`, `_preplexity_sonar`)
carried over from the second-iteration style.

## Execution / Usage

Not executable. Reproduce by running [`third_model/evaluator_v3.py`](../evaluator_v3.py)
with the corresponding `CURRENT_PROVIDER` (judge) and `INPUT_FILE` (target), or
[`third_model/scan_virustotal.py`](../scan_virustotal.py) for the scan output.

## Dependencies

- No code dependencies. Schemas are defined by the producing scripts in
  [`third_model`](../README.md).

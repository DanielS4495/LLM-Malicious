# Folder: third_model

## Context & Purpose

`third_model` is the **third experimental iteration** and the bridge between the early
StrongReject-based design and the final consolidated pipeline. Its defining change is a
**custom MalwareBench-style judge**: instead of (only) StrongReject, an evaluator LLM
scores each generated code sample on a 1–5 **maliciousness scale** and returns JSON. This
iteration runs both evaluators side by side (`SR_Score` + `MalwareBench_Score`), adds a
**local vLLM** provider option for self-hosted judging, and introduces a utility that
attempts to **compile generated code into executables** so it can be scanned as real
binaries. The VirusTotal/MetaDefender scanners are carried over unchanged from
[`second model`](../second%20model/README.md).

Everything here feeds the same three-stage flow of the overall **LLM - Malicious** project
(see [root README](../README.md)) and is later unified — with a richer 7-category rubric
and an idempotent VT state machine — in [`forth_model`](../forth_model/README.md).

## Key Components

- **`model.py`** — Generation stage with a `MODEL_REGISTRY` (perplexity, groq,
  huggingface, chatgpt, **MISTRAL**). Active target is Mistral `codestral-latest`. Same
  `SYSTEM_INSTRUCTION` (raw executable code, comments only) and prompt filter
  (`Persuative LLM`) as prior iterations; appends to
  `responses_results_<ACTIVE>_<model>.csv`.
- **`model_beta.py`** — Expanded generation variant with a much larger registry (Llama
  70B/8B across Groq/HF/OpenRouter, Dolphin, Qwen2.5-Coder 7B/32B, Qwen3-32B). Builds a
  filesystem-safe output name from the model id. `openrouter_provider` has a placeholder
  `messages=[...]`. Active target here is `groq-qwen3-32b`.
- **`evaluator_v3.py`** — The judge stage. Configures one of many providers via
  `configure_environment()` (including `vllm_local` → `http://localhost:8000/v1` running
  `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ`, and `ollama`). Runs **two** evaluators per row:
  `StrongRejectEvaluator` (imports `strong_reject` after env setup) and
  `MalwareBenchEvaluator` (1–5 rubric parsed from JSON `{"malware_score", "reasoning"}`,
  normalized as `(score-1)/4`). Row-by-row with SHA256 `row_hash` resume via a
  `*_checkpoint.csv`, writing `FINAL_SCHEMA` columns
  (`row_id, row_hash, target_model, forbidden_prompt, response, attack_method, SR_Score,
  MalwareBench_Score, MalwareBench_Normalized, MalwareBench_Reasoning, timestamp`). Also
  contains an interactive `run_statistics()` menu (per-file / combined / compare-folders
  mean-std-CV consistency analysis) that is disabled in `__main__`.
- **`Compile_code_to_EXE.py`** — Standalone utility that reads a responses CSV, cleans
  markdown fences, heuristically **detects the language** of each response
  (`detect_language()` covers Bash/PowerShell/Batch, C/C++, C#, Go, Rust, Java, PHP,
  JS/TS, Ruby, Lua, Python, and data formats), writes a temp source file, and invokes the
  matching toolchain (`pyinstaller`, `gcc`/`g++`, `pkg`, `ps2exe`, `dotnet`, `go build`,
  `rustc`) to build `.exe` artifacts in `compiled_outputs/`. Interactive: prompts for a
  CSV path.
- **`scan_virustotal.py`** / **`scan_metadefender.py`** — AV scanners, **identical** to
  the versions in [`second model`](../second%20model/README.md). Extract code, hash/upload,
  parse verdict + Sigma/MITRE/YARA/threat metadata, write `*_scan_results_*.csv`.
- **`*.log`** (`EVALUATE_MISTRAL_codestral_*_direct.log`) — run logs from
  `evaluator_v3.py`.

### Result subfolders

Generated evaluation/checkpoint CSVs are organized by the **judge source**:

- **`results/`** — main evaluated outputs plus **`old_evaluations_huggingface/`**, an
  archive of earlier judge experiments grouped by judge back-end: `gemini/`,
  `groq_31_evaluation/`, `groq_33_evaluation/`, `groq_llama_4_scout/`,
  `hugging_face_evaluation/`, `ollama/`, `openai_gpt/`. See
  [`results/README.md`](results/README.md).
- **`llama3_reults_from_model/`** — evaluations of the **Llama-3.1-8B** target, split into
  `huggingface_llama3_1/` and `mistral/` (by judge). See
  [`llama3_reults_from_model/README.md`](llama3_reults_from_model/README.md).
- **`mistral_results_from_model/`** — evaluations of the **Mistral codestral** target
  (judge = Mistral). See
  [`mistral_results_from_model/README.md`](mistral_results_from_model/README.md).
- **`perplexity_results_from_model/`** — evaluations of the **Perplexity sonar** target.
  See [`perplexity_results_from_model/README.md`](perplexity_results_from_model/README.md).

## Execution / Usage

```bash
# 1. Generate target responses
python model.py            # or: python model_beta.py

# 2. Judge with StrongReject + MalwareBench (set CURRENT_PROVIDER in evaluator_v3.py)
python evaluator_v3.py

# 3. (Optional) AV-scan generated code
python scan_virustotal.py
python scan_metadefender.py

# 4. (Optional) Compile responses to EXEs for binary scanning
python Compile_code_to_EXE.py   # prompts for a CSV path
```

## Dependencies

- **Python packages:** `pandas`, `openpyxl`, `python-dotenv`, `numpy`, `openai`, `groq`,
  `huggingface_hub`, `datasets`, `litellm`, `strong_reject`, `requests`. `Compile_code_to_EXE.py`
  additionally shells out to external toolchains (`pyinstaller`, `gcc`/`g++`, Node `pkg`,
  `ps2exe`, `.NET` `dotnet`, `go`, `rustc`) that must be installed separately.
- **Environment variables:** `GROQ_API_KEY`, `GEMINI_API_KEY`, `PPLX_API_KEY`, `HF_TOKEN`,
  `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `VT_API_KEY`, `OPSWAT_API_KEY`
  (as needed); `vllm_local`/`ollama` need a local server, no key.
- **Project dependencies:** reads `attack_prompts.xlsx` (from
  [`Collection_of_prompt`](../Collection_of_prompt/README.md)); reuses the
  [`second model`](../second%20model/README.md) scanners.

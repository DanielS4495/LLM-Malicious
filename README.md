# LLM Safety-Robustness Evaluation Pipeline

A research framework for benchmarking how large language models respond to adversarial requests — measuring refusal behaviour, response consistency, and safety-robustness across multiple model families.

> **Academic context.** This project was developed as part of a final-year degree project. Its goal is to *measure and compare* the robustness of LLMs against misuse, using an automated evaluation pipeline built around a judge model and statistical consistency analysis.

---

## 📖 Overview

The pipeline runs a fixed set of adversarial prompts through several LLMs, then uses a separate **judge model** to score each response against a safety rubric. The scored outputs feed a statistical layer that measures consistency and produces the metrics used for analysis and academic presentation.

The core research question is not *how to produce harmful output*, but **how reliably different models refuse or resist adversarial requests**, and how consistent that behaviour is across repeated trials.

---

## ✨ What the Pipeline Does

- **Multi-model generation** — sends the same prompt set to several LLMs through a unified interface.
- **Automated judging** — a judge LLM scores each response against a safety rubric and assigns a risk/refusal score.
- **Consistency analysis** — statistical scripts measure how stable each model's behaviour is across runs.
- **Reproducible outputs** — results are stored per-model in isolated folders with a clear, serialized naming convention.
- **Rate-limit aware** — request pacing and serial processing keep the pipeline within free-tier API limits.

---

## 🤖 Evaluated Models

Results for each model are stored in isolated directories to keep comparisons clean. Model families evaluated include:

| Family | Example models |
|--------|----------------|
| Llama 3 | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Mistral | `Codestral`, `devstral-small-2507` |
| Qwen | `Qwen-2.5-Coder` |

---

## 🗂️ Project Structure

The project evolved through several experimental iterations (`first-model`, `second_model`, `third_model`). The current, consolidated pipeline lives in **`forth_model/`**, which merges the successful components of the earlier iterations into a single environment.

```text
forth_model/
├── evaluation/                   # Evaluation scripts and assets
├── results/                      # Aggregated result outputs
├── results_codestral-latest/     # Per-model outputs (Codestral)
├── results_devstral-small-2507/  # Per-model outputs (Devstral)
│   └── responses/                # Raw response data (evaluated batches)
├── stats_output/                 # Generated statistics and analysis
├── prompts.xlsx                  # Adversarial prompt set (evaluation input)
├── model.py                      # Model interaction & response generation
├── evaluator.py                  # Primary judge-model evaluation
├── evaluator_2.py                # Secondary / advanced evaluation pipeline
└── statistics.py                 # Consistency and statistical metrics
```

---

## 🔬 Pipeline Stages

### 1. Response Generation — `model.py`

Reads the prompt set, sends each prompt to a specified LLM, and saves the raw response to that model's results folder.

- **Prompt selection** — filter the prompt set with standard `pandas` operations.
- **Rate limiting** — `time.sleep()` paces requests to respect API limits.
- **Clean datasets** — valid responses (including refusals) are saved; transport-level errors (e.g. `Bad Request`) are skipped so the output stays clean.

### 2. Response Evaluation — `evaluator.py` / `evaluator_2.py`

Acts as the **judge**: reads the generated CSVs, passes each response alongside its original prompt to an evaluator LLM, and assigns a safety score against the rubric.

Output files use a serialized naming convention, saved to the model's results directory:

```text
EVALUATE_[model]_[batch]_groq_[judge-model]_final.csv
```

### 3. Statistical Analysis — `statistics.py`

Reads the scored CSVs and computes consistency, aggregate metrics, and summaries for analysis and validation. Outputs go to `stats_output/`.

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- *(Optional)* A dedicated GPU if running local inference via `vllm`.

### Setup

```bash
# 1. Clone
git clone https://github.com/DanielS4495/LLM-Malicious.git
cd LLM-Malicious

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with your API credentials. When using **Groq** as the judge backend, set:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.groq.com/openai/v1
DATASETS_NUM_PROC=1               # serial processing for free-tier limits
```

---

## ▶️ Usage

Run all steps from inside the `forth_model` directory:

```bash
cd forth_model

# Step 1 — generate model responses
python model.py

# Step 2 — score responses with the judge model
python evaluator.py
# or the secondary pipeline:
# python evaluator_2.py

# Step 3 — compute consistency metrics and summaries
python statistics.py
```

---

## 📊 Interpreting Results

Each model's directory contains its raw responses and scored evaluation files. The statistical output in `stats_output/` reports how consistently each model refused or resisted adversarial prompts — the primary basis for cross-model comparison in this study.

---

## 📚 References

- [Project Documentation / Base Article](https://docs.google.com/document/d/1u4HCAhez2J9OF_zu4qODzZ0taol4BOYi_jz9zEAOXcs/edit?usp=sharing)
**Repositories & Datasets:**
* [MalwareBench](https://github.com/MAIL-Tele-AI/MalwareBench.git) - *See `attack_prompts.xlsx` in the README*
* [Jailbreak LLMs](https://github.com/verazuo/jailbreak_llms.git) - *Navigate to `data/prompts`*
* [Malware-Database](https://github.com/cryptwareapps/Malware-Database.git)
* [MalwareSourceCode](https://github.com/vxunderground/MalwareSourceCode.git)
* [Codesagar Malicious LLM Prompts](https://huggingface.co/datasets/codesagar/malicious-llm-prompts) (HuggingFace)
* [CySecBench Dataset](https://github.com/cysecbench/dataset.git)
* [Perplexity AI Database Search](https://www.perplexity.ai/search/bshbyl-mkhqr-blbd-tmts-ly-data-4hYFvvAVT4SoDQoW.IjkIA?0=d#0) - *Page used for research and database collection*.

---

.

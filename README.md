# LLM-Malicious

*Note: The sources for each collection of prompts are documented in the comment section for each commit.*

## 📚 References & Sources

**Articles:**
* [Project Documentation / Base Article](https://docs.google.com/document/d/1u4HCAhez2J9OF_zu4qODzZ0taol4BOYi_jz9zEAOXcs/edit?usp=sharing)

**Repositories & Datasets:**
* [MalwareBench](https://github.com/MAIL-Tele-AI/MalwareBench.git) - *See `attack_prompts.xlsx` in the README*
* [Jailbreak LLMs](https://github.com/verazuo/jailbreak_llms.git) - *Navigate to `data/prompts`*
* [Malware-Database](https://github.com/cryptwareapps/Malware-Database.git)
* [MalwareSourceCode](https://github.com/vxunderground/MalwareSourceCode.git)
* [Codesagar Malicious LLM Prompts](https://huggingface.co/datasets/codesagar/malicious-llm-prompts) (HuggingFace)
* [CySecBench Dataset](https://github.com/cysecbench/dataset.git)
* [Perplexity AI Database Search](https://www.perplexity.ai/search/bshbyl-mkhqr-blbd-tmts-ly-data-4hYFvvAVT4SoDQoW.IjkIA?0=d#0) - *Page used for research and database collection*

---

## 📂 Project Structure & Workflow

The project is structured into multiple stages (folders), representing the evolution and refinement of our evaluation pipeline.

* **Historical Iterations (`first-model`, `second model`, `third_model`):** These folders contain earlier experimental versions. Each iteration focused on testing different approaches, methodologies, and specific aspects of LLM evaluation.
* **Unified Pipeline (`forth_model`):** This is the **current, fully updated, and improved version** of the project. Insights and successful components from the previous iterations were merged into this centralized environment. All current evaluations, comprehensive scripts, and results are actively managed here.

### Structure of the Current Workspace (`forth_model`):
```text
forth_model/
├── evaluation/                   # Evaluation scripts and assets
├── results/                      # General or aggregated result outputs
├── results_codestral-latest/     # Specific outputs for the Codestral model
├── results_devstral-small-2507/  # Specific outputs for the Devstral model
│   └── responses/                # Raw response data
│       ├── EVALUATE_devstral-small-2507_1_groq_llama-3.3_final.csv
│       ├── EVALUATE_devstral-small-2507_2_groq_llama-3.3_final.csv
│       └── ...                   # Additional evaluated batches
├── stats_output/                 # Generated statistics and analysis
├── attack_prompts.xlsx           # The primary dataset of malicious prompts
├── model.py                      # Main script for LLM interaction & generation
├── evaluator.py                  # Primary evaluation script (Judge LLM)
├── evaluator_2.py                # Secondary/Advanced evaluation pipeline
└── statistics.py                 # Script for calculating consistency and statistical metrics


1. Model Generation (model.py)
This script is responsible for generating model responses from a list of adversarial prompts.

Purpose: Reads prompts from the Excel file (attack_prompts.xlsx), sends them to a specified LLM, and saves the raw responses to a designated output folder (e.g., results_codestral-latest/).

Key Features:

Prompt Filtering: Customize prompt selection using pandas filtering.

Rate Limiting: Utilizes time.sleep() to pause the script between requests, ensuring API rate limits are respected.

Error Handling: Successful responses (including model refusals) are saved, while system errors (like Bad Request) are bypassed to keep the output datasets clean.

2. Response Evaluation (evaluator.py & evaluator_2.py)
These scripts act as the "Judge", evaluating the generated responses based on our security rubrics.

Purpose: Reads the generated CSV files, sends the output alongside the original prompt to an evaluator LLM (e.g., Llama via Groq), and assigns a security risk score.

Output: Generated evaluation files are saved with clear, serialized naming conventions (e.g., EVALUATE_[model]_[batch]_groq_[judge-model]_final.csv) within the model's respective results directory.

Groq Configuration: When using Groq for evaluation, ensure the following environment variables are set:

os.environ["DATASETS_NUM_PROC"] = "1": Forces serial processing to respect free-tier limits.

os.environ["OPENAI_API_KEY"]: Your Groq API key.

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

3. Statistical Analysis (statistics.py)
Purpose: Analyzes the scored CSV files from the evaluation phase to calculate consistency, generate metrics, and prepare data for academic presentation and validation. Outputs are generally directed to the stats_output/ folder.



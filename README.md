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
🚀 Evaluation Pipeline (Located in forth_model)
This pipeline is designed to generate and evaluate malicious code responses across multiple LLMs.

🤖 Tested Models
Our research evaluates the capabilities of several models by storing their respective results in isolated folders. Evaluated models include:

Llama 3 Series (meta-llama/Meta-Llama-3-8B-Instruct, etc.)

Mistral Series (Codestral, devstral-small-2507, etc.)

Qwen Series (Qwen-2.5-Coder)

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

🛠️ Installation & Environment Setup
To run the evaluation pipeline locally, you need to set up your environment and install the required dependencies.

1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

(Optional but recommended) A machine with a dedicated GPU if you plan to run local inference models via vllm.

2. Clone the Repository
Clone the project to your local machine and navigate into the project directory:

Bash
git clone [https://github.com/DanielS4495/LLM-Malicious.git](https://github.com/DanielS4495/LLM-Malicious.git)
cd LLM-Malicious
3. Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to prevent dependency conflicts with other Python projects.

On Windows:

Bash
python -m venv venv
venv\Scripts\activate
On macOS/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
Once the virtual environment is activated, install all required packages using the requirements.txt file:

Bash
pip install -r requirements.txt
5. Environment Variables
To interact with external APIs (like Groq or OpenAI), you must configure your environment variables.

Navigate to the forth_model/stats_output directory (or wherever your .env file is meant to be located).

Create a .env file based on the provided template and add your API keys:

קטע קוד
OPENAI_API_KEY=your_groq_or_openai_api_key_here
▶️ Usage / How to Run
After setting up your environment, follow these steps to execute the pipeline. All operations should be performed from within the forth_model directory:

Bash
cd forth_model
Step 1: Generate Responses
Run the generation script to query the LLMs using the malicious prompts dataset. The script will save the output in the respective results folder.

Bash
python model.py
Step 2: Evaluate Responses
Once generation is complete, run the evaluation script to score the outputs utilizing the Judge LLM.

Bash
python evaluator.py
# Or, to run the secondary evaluation pipeline:
# python evaluator_2.py
Step 3: Generate Statistics
Finally, run the statistics script to analyze the evaluated files and generate consistency metrics and data summaries.

Bash
python statistics.py

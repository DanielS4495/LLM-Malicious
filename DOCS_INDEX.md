# Documentation Index — LLM - Malicious

This index links every per-directory `README.md` generated for the repository, grouped by
the project's top-level architecture. The project is an academic **LLM Safety-Robustness
Evaluation Pipeline**: it measures how reliably code-capable LLMs refuse or resist
adversarial "malicious code generation" prompts. See the [root README](README.md) for the
overall design and the [`forth_model/CLAUDE.md`](forth_model/CLAUDE.md) spec for the
current pipeline's contract.

All documentation is descriptive only — it documents existing code and does not add,
extend, or improve any capability.

## Shared vocabulary (three stages)

1. **Generation** — `model.py` sends the curated adversarial prompt set
   (`attack_prompts.xlsx`, filtered to `AttackMethod == 'Persuative LLM'`) to a target LLM
   and saves raw responses to CSV.
2. **Evaluation / Judging** — an `evaluator*.py` acts as an LLM judge (early: StrongReject;
   later: a MalwareBench-style maliciousness rubric) plus optional VirusTotal /
   MetaDefender malware scanning of the generated code.
3. **Statistics** — consistency/robustness metrics, comparisons, plots, and reports.

The project evolved across four iterations: `first-model` → `second model` →
`third_model` → **`forth_model`** (current, consolidated).

## Shared inputs

- [Collection_of_prompt/](Collection_of_prompt/README.md) — raw external prompt corpora
  (CySecBench, jailbreak_llms snapshots, regular/benign controls) and the curated
  `attack_prompts.xlsx` benchmark input used by every iteration.

## Iteration 1 — first-model (baseline)

- [first-model/](first-model/README.md) — earliest generation + StrongReject judge +
  simple jailbreak-rate statistics.

## Iteration 2 — second model (provider registry + AV scanning)

- [second model/](second%20model/README.md) — provider registry, multi-backend
  StrongReject judge with retry mode, and first VirusTotal / MetaDefender scanners.
  - [second model/results/](second%20model/results/README.md) — evaluated and scan output
    CSVs.

## Iteration 3 — third_model (MalwareBench rubric + compilation)

- [third_model/](third_model/README.md) — dual StrongReject + MalwareBench (1–5) judge,
  local vLLM/Ollama option, and a code→EXE compilation utility.
  - [third_model/results/](third_model/results/README.md) — main outputs plus the
    `old_evaluations_huggingface/` judge-comparison archive.
  - [third_model/llama3_reults_from_model/](third_model/llama3_reults_from_model/README.md)
    — Llama-3.1-8B target results by judge.
  - [third_model/mistral_results_from_model/](third_model/mistral_results_from_model/README.md)
    — Mistral codestral target results.
  - [third_model/perplexity_results_from_model/](third_model/perplexity_results_from_model/README.md)
    — Perplexity sonar target results.

## Iteration 4 — forth_model (current, consolidated pipeline)

- [forth_model/](forth_model/README.md) — 7-category MalwareBench rubric (composite 0–10),
  native idempotent VirusTotal state machine, multi-layer statistics with HTML report.
  StrongReject and MetaDefender are retired here. Authoritative spec:
  [forth_model/CLAUDE.md](forth_model/CLAUDE.md).
  - [forth_model/evaluation/](forth_model/evaluation/README.md) — curated final
    `EVALUATE_*_final.csv` per model (Stage-3 input).
  - [forth_model/results/](forth_model/results/README.md) — raw Stage-1 response CSVs.
  - [forth_model/results_codestral-latest/](forth_model/results_codestral-latest/README.md)
    — Mistral codestral batches (responses + evaluations).
  - [forth_model/results_devstral-small-2507/](forth_model/results_devstral-small-2507/README.md)
    — Mistral devstral batches (responses + evaluations).

## Notes on excluded folders

- `.idea/` (JetBrains IDE config), `.git/`, and `__pycache__/` are intentionally
  undocumented (tooling/vendor artifacts).
- Deeply-nested single-purpose result leaves (e.g. per-judge subfolders under
  `third_model/results/old_evaluations_huggingface/` and the `responses/` subfolders in
  `forth_model`) are described inside their parent folder's README rather than as separate
  files.
- The root [`Prompt`](Prompt) file (dataset provenance notes) and the root
  [`README.md`](README.md) already existed and were left unchanged.

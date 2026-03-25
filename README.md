# RecLAIF Starter Implementation

This repository bootstraps an incremental implementation of **RecLAIF: Reinforcement Learning from AI Feedback for Recommendation Systems**.

The paper frames the system as:

1. A **recommender LLM** that reads user context plus sampled candidates and produces:
   - key preferences / reasoning features
   - ranked recommendations with short explanations
2. A **judge LLM** that compares two sampled recommender outputs on:
   - relevance
   - diversity
   - explainability
3. A preference optimization loop that:
   - collects preference pairs
   - fine-tunes the recommender with **SFT**
   - iteratively improves it with **DPO**

This starter repo focuses on making that workflow easy to extend:

- clean data schemas
- prompt builders aligned with the paper
- parsers for structured outputs
- a pipeline for sampling, judging, and producing DPO pairs
- a mock LLM path so we can test the loop before wiring real APIs or local models

## Project Layout

```text
configs/
  reclaif.sample.json
data/
  beauty_sample.jsonl
scripts/
  evaluate_pairs.py
  prepare_dataset_splits.py
  run_iterative_dpo.py
  run_mock_pipeline.py
  run_pipeline.py
  show_experiment_config.py
  train_sft.py
  train_dpo.py
src/reclaif/
  __init__.py
  datasets.py
  experiment.py
  io.py
  llm.py
  metrics.py
  parsers.py
  pipeline.py
  prompts.py
  runner.py
  schemas.py
  training.py
```

## What Is Implemented

### 0. Reproducibility foundation (Priority A)

The repo now includes first-class building blocks to lock and replay experiments:

- dataset split adapters for `esci`, `beauty`, and `lastfm` with canonical split file names
  (`train.jsonl`, `valid.jsonl`, `test.jsonl`)
- deterministic candidate generation policy (`given` or `random_from_catalog`) with seed control
- version-locked experiment configs in `configs/experiments/*.json` for
  - model IDs (recommender/judge/teacher)
  - prompt version
  - generation params (`temperature`, `top_p`)
  - DPO/SFT hyperparameters (`beta`, epochs)
  - random seed

### 1. Recommendation prompt generation

The recommender prompt mirrors the paper's structure:

- task framing
- user context / history
- candidate set
- request for 2-3 key preferences
- ranked recommendations with reasons

### 2. Judge prompt generation

The judge prompt compares `Option A` and `Option B` and requests:

- per-criterion scoring
- short rationale
- final chosen / rejected option

### 3. Structured parsing

The repo includes lightweight parsers for:

- recommender outputs
- judge outputs
- weighted judge score aggregation (relevance/diversity/explainability) to recover a
  stable winner when `Chosen Option` is omitted

This makes the system usable even before moving to strict JSON outputs.

### 4. Preference pair generation

The iterative loop in the paper is implemented as a practical pipeline:

- build prompt from an example
- sample two recommender responses
- judge them
- emit a DPO training pair

### 5. SFT/DPO dataset artifacts

The code writes JSONL records for:

- SFT examples
- preference pairs

That lets us later plug in:

- Hugging Face TRL DPOTrainer
- a custom trainer
- OpenAI or Bedrock-based data generation

### 6. Offline evaluation helper

The repo now includes a lightweight evaluator for generated preference pairs:

- script: `scripts/evaluate_pairs.py`
- metrics: `Precision@3`, `Hit@1`, `NDCG@3`, `ValidRatio@3`, lexical diversity
- input: `data/*.jsonl` plus `artifacts/preference_pairs.jsonl`
- output: JSON summary so you can compare iterations quickly

### 7. Real backends and actual training

### 8. Automated Iter1..Iter4 loop with lineage (Priority B)

The repo includes an automated iterative runner in `src/reclaif/runner.py` and
CLI entrypoint `scripts/run_iterative_dpo.py`:

- Iter0: teacher-driven SFT data generation and optional SFT training
- Iter1..IterN: preference pair generation + optional DPO training
- explicit checkpoint lineage written to `run_manifest.json`
- deterministic run controls:
  - seed propagation
  - fixed prompt version from experiment config
  - version-locked model IDs and sampling/training hyperparameters from config
  - reference-model handling for DPO (`reference_model_name_or_path`)

The repo now includes:

- hosted inference clients in [`src/reclaif/llm.py`](/C:/Users/rrpte/Documents/New%20project/src/reclaif/llm.py)
  - `OpenAIRecommenderClient`
  - `OpenAIJudgeClient`
- local Hugging Face generation clients in [`src/reclaif/llm.py`](/C:/Users/rrpte/Documents/New%20project/src/reclaif/llm.py)
  - `TransformersRecommenderClient`
  - `TransformersJudgeClient`
- TRL-backed training utilities in [`src/reclaif/training.py`](/C:/Users/rrpte/Documents/New%20project/src/reclaif/training.py)
  - `train_sft_from_jsonl(...)`
  - `train_dpo_from_jsonl(...)`
- CLI entry scripts:
  - [`scripts/train_sft.py`](/C:/Users/rrpte/Documents/New%20project/scripts/train_sft.py)
  - [`scripts/train_dpo.py`](/C:/Users/rrpte/Documents/New%20project/scripts/train_dpo.py)

## Backends

### Hosted models

Use `OpenAIRecommenderClient` and `OpenAIJudgeClient` when you want a stronger hosted model to:

- generate teacher responses for SFT
- act as the judge in the RecLAIF loop
- optionally serve as the recommender too

These clients use the OpenAI Responses API and read `OPENAI_API_KEY` by default.

### Local models

Use `TransformersRecommenderClient` and `TransformersJudgeClient` when you want to run local or self-hosted Hugging Face causal LMs.

## Training

The generated artifacts now map directly to current TRL trainers:

- `sft_records.jsonl` -> `SFTTrainer`
- `preference_pairs.jsonl` -> `DPOTrainer`

The trainer code assumes the standard TRL dataset shapes:

- SFT rows: `prompt`, `completion`
- DPO rows: `prompt`, `chosen`, `rejected`

### Example commands

```bash
python scripts/show_experiment_config.py --config configs/experiments/beauty.v1.json
python scripts/prepare_dataset_splits.py --dataset-root data --dataset beauty --output-dir prepared_data --policy random_from_catalog --k 50 --seed 7
python scripts/run_iterative_dpo.py --config configs/experiments/beauty.v1.json --dataset-root data --output-dir runs/beauty --dry-run
python scripts/run_pipeline.py --recommender-backend transformers --recommender-model mistralai/Mistral-7B-Instruct-v0.3 --judge-backend openai --judge-model gpt-5.2 --teacher-backend openai --teacher-model gpt-5.2
python scripts/train_sft.py --model mistralai/Mistral-7B-Instruct-v0.3 --use-peft
python scripts/train_dpo.py --model checkpoints/sft --use-peft
```

## Installation

Install the extras that match the path you want:

```bash
pip install -e .[hosted]
pip install -e .[train]
pip install -e .[full]
```

## Current Constraints

This environment here still does **not** expose a usable Python + ML stack, so I could not run the new training scripts locally after wiring them in.

Natural next steps:

1. Add dataset-specific preprocessing for ESCI / Beauty / LastFM.
2. Add evaluation metrics backed by embeddings and ground-truth labels.
3. Add checkpointed iterative loops that alternate generation and DPO automatically.
4. Add experiment tracking and validation splits.
5. Add stricter structured output parsing or JSON-mode generation for judges.

## Data Format

Each example is a JSON object with:

- `example_id`
- `task_type`: `retrieval` or `sequential`
- `user_id`
- `context`
- `candidates`
- `ground_truth`
- `domain`

See [`data/beauty_sample.jsonl`](/C:/Users/rrpte/Documents/New%20project/data/beauty_sample.jsonl).

## Suggested Next Implementation Order

1. Generate SFT labels from a stronger hosted judge/recommender.
2. Train the first SFT checkpoint with LoRA.
3. Run the pipeline with the SFT checkpoint as recommender and a stronger model as judge.
4. Train DPO on the resulting preference pairs.
5. Add dataset adapters one by one.

## Paper Notes

This starter follows the paper's main method:

- SFT on stronger-model responses
- two sampled candidate outputs from the recommender
- judge scoring on relevance, diversity, explainability
- iterative DPO over chosen/rejected pairs

Primary source used:

- [RecLAIF paper](https://nurendra.com/papers/KDDW2025_1.pdf)

Implementation references used for the trainer/backend wiring:

- [Hugging Face TRL DPOTrainer docs](https://huggingface.co/docs/trl/en/dpo_trainer)
- [Hugging Face TRL dataset formats docs](https://huggingface.co/docs/trl/v0.12.1/dataset_formats)
- [OpenAI Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create)

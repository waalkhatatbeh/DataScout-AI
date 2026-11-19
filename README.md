
# DatasetScout AI

DatasetScout AI is a dataset recommendation and question-answering assistant designed to help users discover relevant datasets based on their requirements.

## Project Overview

This project explores the adaptation of GPT-2 Small for dataset-related question answering through Continued Pretraining (CPT) and Supervised Fine-Tuning (SFT).

Phase 2 focuses on training, evaluating, and comparing three model versions:
- GPT-2 Base
- Continued Pretrained Model (CPT)
- Supervised Fine-Tuned Model (SFT)

## Training Pipeline

The training process follows this sequence:

GPT-2 Small → CPT → Best CPT Checkpoint → SFT

The SFT model was initialized from the best CPT checkpoint rather than the original GPT-2 model.

### 1. Continued Pretraining (CPT)

- Base model: GPT-2 Small (`openai-community/gpt2`)
- Training epochs: 2
- Sequence length: 512
- Learning rate: 5e-5
- Batch size: 2
- Gradient accumulation steps: 4
- Random seed: 42
- GPU: NVIDIA Tesla T4

Best CPT validation results:
- Validation loss: 3.9215
- Validation perplexity: 50.4783

### 2. Supervised Fine-Tuning (SFT)

The SFT stage used 300 question-answer pairs and started from the best CPT checkpoint.

- Training epochs: 3
- Effective batch size: 16
- Maximum sequence length: 512
- Training objective: Answer-only loss

## Evaluation

All three model versions were evaluated using the same held-out test set of 30 questions.

Scoring system:
- Correct: 1
- Partially correct: 0.5
- Wrong: 0

Evaluation included:
- Full accuracy
- Mean rubric score
- Refusal accuracy
- Question-level error analysis

See [SCOREBOARD.md](SCOREBOARD.md) for the evaluation results.

## Repository Structure

The repository includes the project configuration, test questions, evaluation results, and training-related files.

- `results/`: Model evaluation results for BASE, CPT, and SFT.
- `tests/`: Held-out test questions.
- `README.md`: Project documentation.
- `SCOREBOARD.md`: Evaluation summary.
- `config.yaml`: Training configuration.

## Results

Detailed predictions, expected answers, scores, and evaluation notes for each model are available in the `results/` directory.

## Team

DatasetScout AI – Phase 2

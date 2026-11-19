# CPT Training

This folder contains the Continued Pretraining (CPT) training artifacts for DatasetScout AI.

## Training Configuration
- Model: GPT-2 Small
- Epochs: 2
- Block size: 512
- Learning rate: 5e-5
- Batch size: 2
- Gradient accumulation steps: 4
- Random seed: 42

## Results
- Base validation perplexity: 69.2662
- Best CPT validation perplexity: 50.4783
- Perplexity reduction: 27.1%

The training loss curves are provided in `cpt_loss_curves.png`.

Model checkpoints are stored separately and are not included in this repository.

# SFT Training Implementation and Validation Results

## Training configuration
- Base for SFT: existing CPT checkpoint (`checkpoint_epoch2`)
- SFT question–answer pairs: 300
- Training records: 270 (90%)
- Validation records: 30 (10%)
- Training epochs: 3
- Learning rate: 2e-5
- Per-device training batch size: 4
- Gradient accumulation steps: 4
- Effective batch size: 16
- Maximum sequence length: 512
- Special tokens: `<|user|>`, `<|assistant|>`, `<|end|>`
- Answer-only loss: enabled
- Supervised answer tokens in training: 12,683
- GPU used: NVIDIA Tesla T4

## Validation results

| Epoch | Validation loss |
|---|---:|
| 1 | 3.6344563961029053 |
| 2 | 3.534019947052002 |
| 3 | 3.516059160232544 |

Best epoch: 3.0
Best validation loss: 3.516059160232544
Best checkpoint: `/kaggle/working/datascout_sft_300_validated/checkpoint-27`
Final saved model: `/kaggle/working/datascout_sft_300_validated/sft_checkpoint`

## Data and manual review
- The 300 QA pairs and their answers are unchanged from the original master workbook.
- All 50 sampled questions were already marked `reviewed` in `SFT_Master`.
- The 50 `PENDING` entries in `Manual_Review_Log` were changed to `REVIEWED` following the dataset author's confirmation that the reviews were performed.
- No reviewer names or review dates were invented.
- The held-out test set is separate from the 270/30 train–validation split.
- The included `preflight_audit.json` checks exact duplicates and exact train/test overlaps, not a complete independent semantic leakage certification.

## Scope
These are SFT training/validation metrics, not model accuracy or an evaluation of the Base/CPT/SFT comparison. The model checkpoint and complete Kaggle-generated logs must be added separately from Kaggle if required.


# DatasetScout AI – Phase 2 Scoreboard

## Evaluation Setup

- Test set: 30 questions
- Models: GPT-2 Base, Continued Pretraining (CPT), and Supervised Fine-Tuning (SFT).
- Scoring: Correct = 1, Partially Correct = 0.5, Wrong = 0.
- All models were evaluated on the same test questions.

## Results

| Model | Fully Correct | Partially Correct | Wrong | Full Accuracy | Mean Rubric Score | Refusal Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 Base | 0 | 4 | 26 | 0.00% | 6.67% | 0.00% |
| CPT | 0 | 5 | 25 | 0.00% | 8.33% | 0.00% |
| SFT | 0 | 5 | 25 | 0.00% | 8.33% | 0.00% |

## Metric Definitions

- **Full Accuracy:** Percentage of questions receiving a score of 1.
- **Mean Rubric Score:** Sum of all question scores divided by 30.
- **Refusal Accuracy:** Percentage of the 3 out-of-scope questions correctly refused with a score of 1.

## Notes

- The three models were evaluated using the same 30-question test set.
- Partial credit is reported separately from fully correct answers.
- Refusal accuracy is calculated using questions 28, 29, and 30.
- Detailed question-level answers, scores, and notes are available in the `results/` directory.

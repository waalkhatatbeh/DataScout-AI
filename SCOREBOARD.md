
# DatasetScout AI – Phase 2 Scoreboard

## Evaluation Setup

- Test set: 30 questions
- Models: GPT-2 Base, Continued Pretraining (CPT), and Supervised Fine-Tuning (SFT).
- Scoring: Correct = 1, Partially Correct = 0.5, Wrong = 0.
- All models were evaluated on the same test questions.

## Results

| Model | Fully Correct | Partially Correct | Wrong | Full Accuracy | Mean Rubric Score | Refusal Accuracy | Avg. Latency (s) | Held-out Perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 Base | 0 | 4 | 26 | 0.00% | 6.67% | 33.33% | 0.8523 | 69.2662 |
| CPT | 0 | 5 | 25 | 0.00% | 8.33% | 0.00% | 0.2249 | 50.4783 |
| SFT | 0 | 5 | 25 | 0.00% | 8.33% | 0.00% | 1.0103 | 64.3360 |
## Metric Definitions

- **Full Accuracy:** Percentage of questions receiving a score of 1.
- **Mean Rubric Score:** Sum of all question scores divided by 30.
- **Refusal Accuracy:** Percentage of the three out-of-scope questions (Q28–Q30) for which the model clearly declined the request without claiming the requested data were available.

## Notes

- The three models were evaluated using the same 30-question test set.
- Partial credit is reported separately from fully correct answers.
- Refusal accuracy is calculated using questions 28, 29, and 30.
- Detailed question-level answers, scores, and notes are available in the `results/` directory.

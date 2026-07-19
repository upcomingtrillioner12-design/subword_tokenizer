# Human Evaluation Protocol Summary

## Scope
- Sample size: 100 responses
- Sources:
  - `results/rag_generation_eval/rag_generation_eval_20260716_114138.json`
  - `results/rag_generation_eval/rag_generation_eval_20260716_125406.json`
- Template file: `results/human_eval/human_eval_template.csv`

## Annotation Rubric (1-5)
1. **Faithfulness**: Is the answer supported by retrieved context?
2. **Helpfulness**: Is the answer concise and useful for the question?
3. **Correctness**: Is the answer scientifically correct relative to expected answer?

## Process
1. Two annotators independently score each sample.
2. Compute absolute disagreement per metric.
3. Flag samples with disagreement >= 2 points in any metric.
4. Adjudicate flagged samples; fill `adjudicated_final_score_1_5` and `notes`.

## Tie-break / Adjudication Rule
- If one annotator gives 5 and the other <= 3, require adjudication.
- Adjudicator prioritizes:
  1) context faithfulness,
  2) domain correctness,
  3) helpfulness.

## Deliverables Status
- ✅ `human_eval_template.csv` generated (100 rows)
- ✅ Protocol documented
- ⏳ Awaiting annotator scores and adjudication pass

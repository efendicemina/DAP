# Task 3 Evaluation: Response Generation

## Approach
Rule-based and few-shot answer composition (without a local LLM by default).

## Evaluation Scope
The test set covers registration workflows, registry updates, deletion, foundations, exams, fees, contacts, access-to-information (ZOSPI), legal aid, public consultations, out-of-scope requests, and safety-blocked prompts.

## Metrics

- **task**: Task 3 - response generation
- **approach**: rule-based / few-shot answer composer without local LLM
- **number_of_questions**: 19
- **average_score**: 0.9474
- **goal_accuracy**: 0.8947
- **terms_accuracy**: 0.8421
- **format_accuracy**: 1.0
- **noise_free_rate**: 1.0
- **source_coverage**: 0.9474
- **average_latency_ms**: 3404.25
- **median_latency_ms**: 4587.9

## Interpretation
- The system consistently returns well-structured answers.
- Source grounding is high, with nearly complete source coverage.
- Remaining misses are mostly goal-detection or expected-term mismatches on edge questions.

## Non-Perfect Cases (from current report)

| question | expected_goal | detected_goal | intent | score |
|---|---|---|---|---:|
| Koji su uslovi za polaganje pravosudnog ispita? | requirements | general | ispiti | 0.8333 |
| Šta je ZOSPI i gdje mogu naći informacije o pristupu informacijama? | definition | definition | zakoni_i_propisi | 0.8333 |
| Šta uraditi u slučaju međunarodne otmice djeteta? | procedure | general | pravna_pomoc | 0.8333 |
| Gdje se nalaze javne konsultacije Ministarstva pravde BiH? | general | general | kontakt_i_nadleznosti | 0.8333 |
| Gdje mogu izvaditi ličnu kartu? | general | general | needs_clarification | 0.6667 |

## Conclusion
Task 3 performance is strong for production-oriented retrieval-grounded responses, with clear fallback handling for unsafe and out-of-scope queries.

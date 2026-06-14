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
| What are the requirements for taking the bar exam? | requirements | general | ispiti | 0.8333 |
| What is ZOSPI and where can I find information on access to information? | definition | definition | zakoni_i_propisi | 0.8333 |
| What should be done in a case of international child abduction? | procedure | general | pravna_pomoc | 0.8333 |
| Where can I find the Ministry of Justice BiH public consultations page? | general | general | kontakt_i_nadleznosti | 0.8333 |
| Where can I get an ID card? | general | general | needs_clarification | 0.6667 |

## Conclusion
Task 3 performance is strong for production-oriented retrieval-grounded responses, with clear fallback handling for unsafe and out-of-scope queries.

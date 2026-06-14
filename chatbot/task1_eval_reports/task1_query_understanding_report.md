# Task 1 Evaluation: Query Understanding

## Goal
Task 1 evaluates how well the system understands a user question before retrieval and answer generation.

The task has two components:
1. **Intent classification**
2. **Key information extraction** (document type, procedure, legal topic, institution, keywords)

## Text Representation Variants

| Approach | Representation | Description |
|---|---|---|
| Baseline | TF-IDF n-gram features | Lexical representation of the question text |
| Contextual model | SentenceTransformer embeddings | Semantic representation for paraphrases and contextual similarity |

## Intent Classification Results

| Model | Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | validation | 0.9980 | 0.9981 | 0.9980 |
| TF-IDF + Logistic Regression | test | 0.9820 | 0.9798 | 0.9819 |
| SentenceTransformer + Logistic Regression | validation | 0.9740 | 0.9750 | 0.9741 |
| SentenceTransformer + Logistic Regression | test | 0.9600 | 0.9587 | 0.9598 |

## Key Information Extraction Results

Procedure extraction was evaluated using weak labels derived from the synthetic dataset `subintent` field.

| Metric | Value |
|---|---:|
| Evaluated samples | 369 |
| Correct samples | 254 |
| Skipped (no weak label) | 131 |
| Procedure extraction accuracy | 0.6883 |

## Conclusion
Task 1 satisfies project requirements by combining:
- a strong intent classifier, and
- an interpretable extraction layer that enriches retrieval context.

This output is consumed by the retrieval pipeline in `chatbot_pipeline_v1.py`.

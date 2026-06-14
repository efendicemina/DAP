# Visual Evaluation Summary: Retrieval Quality Across Versions

This document summarizes the measured improvement of retrieval relevance across major system versions.

## Compared Versions
- **v4 TF-IDF**: lexical retrieval baseline.
- **v4 Embeddings**: semantic retrieval with sentence-transformers.
- **v5 Embeddings**: semantic retrieval + metadata-aware ranking.

## Core Metrics

| Version | Top-1 Accuracy | Top-3 Coverage | Top-5 Coverage | Mean Reciprocal Rank (MRR) |
|---|---:|---:|---:|---:|
| v4 TF-IDF | 58.88% | 67.29% | 70.09% | 0.6358 |
| v4 Embeddings | 63.55% | 68.22% | 74.77% | 0.6718 |
| v5 Embeddings | 72.90% | 76.64% | 78.50% | 0.7488 |

## Improvement Highlights
- **v4 TF-IDF → v4 Embeddings**
  - Top-1: **+4.67 pp**
  - Top-3: **+0.93 pp**
  - Top-5: **+4.68 pp**
- **v4 Embeddings → v5 Embeddings**
  - Top-1: **+9.35 pp**
  - Top-3: **+8.42 pp**
  - Top-5: **+3.73 pp**
- **v4 TF-IDF → v5 Embeddings**
  - Top-1: **+14.02 pp** (**23.8% relative improvement**)

## Interpretation
- The semantic transition (TF-IDF → embeddings) provided the first major gain.
- Metadata-aware ranking in v5 produced the largest Top-1 jump.
- The current v5 setup is the best-performing retrieval variant used by `chatbot_pipeline_v1.py`.

## Project Alignment
- Runtime retrieval implementation: `chatbot/chatbot_pipeline_v1.py`
- V5 FAISS assets:
  - `chatbot/models_v5/faiss_index_v5.bin`
  - `chatbot/models_v5/embedding_config_v5.joblib`
- Evaluation scripts:
  - `chatbot/evaluate_rag_embeddings_v4.py`
  - `chatbot/evaluate_rag_embeddings_v5.py`

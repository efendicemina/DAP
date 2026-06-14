# Chatbot Module (DAP)

This folder contains data collection, dataset processing, model training, retrieval evaluation, and runtime chatbot pipeline code.

## Folder Purpose
`/home/runner/work/DAP/DAP/efendicemina/DAP/chatbot`

Main responsibilities:
- scrape and curate Ministry of Justice BiH content,
- build filtered datasets and chunked corpora,
- train and evaluate intent/retrieval components,
- serve retrieval-ready artifacts for backend inference.

## Key Files
- `scraper_v4_curated.py` - curated website scraping.
- `postprocess.py` - strict dataset filtering.
- `chatbot_pipeline_v1.py` - runtime retrieval + response pipeline.
- `query_understanding_v1.py` - extraction helpers for user-query understanding.
- `few_shot_composer.py` - response composition layer.
- `evaluate_task1_query_understanding.py` - Task 1 evaluation.
- `evaluate_task3_generation.py` - Task 3 evaluation.

## Key Data and Model Artifacts
- `mpr_dataset_v5/chunks_v5.csv`
- `models/intent_classifier.joblib`
- `models_v5/faiss_index_v5.bin`
- `models_v5/embedding_config_v5.joblib`

## Typical Workflow
1. **Scrape/refresh source content**.
2. **Filter and clean** records.
3. **Create chunk dataset** with metadata.
4. **Train/update intent model**.
5. **Build/rebuild FAISS embedding index**.
6. **Run evaluation scripts**.
7. **Use pipeline via backend `/chat` endpoint**.

## Run Selected Tasks

### Evaluate query understanding (Task 1)
```bash
cd /home/runner/work/DAP/DAP/efendicemina/DAP/chatbot
python evaluate_task1_query_understanding.py
```

### Evaluate answer generation (Task 3)
```bash
cd /home/runner/work/DAP/DAP/efendicemina/DAP/chatbot
python evaluate_task3_generation.py
```

### Run pipeline manually
```bash
cd /home/runner/work/DAP/DAP/efendicemina/DAP/chatbot
python chatbot_pipeline_v1.py
```

## Integration Point
The backend imports `ask()` from `chatbot_pipeline_v1.py` in:
- `/home/runner/work/DAP/DAP/efendicemina/DAP/backend/main.py`

## Notes
- Runtime default is retrieval + few-shot/rule-based generation.
- Local LLM usage is optional (`USE_LOCAL_LLM = False` by default).

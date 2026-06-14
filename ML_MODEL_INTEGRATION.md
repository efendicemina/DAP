# ML Integration with Backend and Frontend

## Status
✅ Integrated and active through `chatbot_pipeline_v1.py` in backend runtime.

## Integrated Components

### Backend
- File: `/home/runner/work/DAP/DAP/efendicemina/DAP/backend/main.py`
- Behavior:
  - Imports `ask()` from `chatbot/chatbot_pipeline_v1.py`.
  - Uses ML output for `/chat` responses.
  - Returns `response`, `confidence`, `category`, `intent`, and `sources`.
  - Falls back to `dummy_responses.py` if ML model loading fails.

### Frontend
- File: `/home/runner/work/DAP/DAP/efendicemina/DAP/frontend/components/ChatInterface.tsx`
- Behavior:
  - Sends user query to `POST /chat`.
  - Renders answer text, confidence, intent, and ranked source links.
  - Loads quick-start questions from `GET /suggested-questions`.

### Pipeline
- File: `/home/runner/work/DAP/DAP/efendicemina/DAP/chatbot/chatbot_pipeline_v1.py`
- Core flow:
  1. Query understanding + intent prediction.
  2. Direct routing for strong URL-pattern matches.
  3. Embedding retrieval (FAISS) + reranking.
  4. Few-shot/rule-based answer generation.

## Runtime API Contract
`POST /chat` returns:
- `response` (string)
- `confidence` (float)
- `category` (string)
- `sources` (list of `{title, url, score, page_type, semantic_topic}`)
- `timestamp` (ISO string)
- `query` (original question)
- `intent` (string)

## Local Run Sequence

### 1) Backend
```bash
cd /home/runner/work/DAP/DAP/efendicemina/DAP/backend
pip install -r requirements.txt
python main.py
```

### 2) Frontend
```bash
cd /home/runner/work/DAP/DAP/efendicemina/DAP/frontend
npm install
npm run dev
```

### 3) Open App
- Frontend: `http://localhost:3000`
- Backend docs: `http://localhost:8000/docs`

## Verification Endpoints
- `GET /health`
- `GET /info`
- `GET /suggested-questions`
- `POST /chat`

## Troubleshooting
- If ML artifacts are missing, backend logs a warning and uses dummy responses.
- First startup can be slower due to model loading.
- Frontend build may require network access for remote fonts in restricted environments.

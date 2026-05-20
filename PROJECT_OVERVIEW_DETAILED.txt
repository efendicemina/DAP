PROJECT OVERVIEW - DAP / MPR CHATBOT
Date snapshot: 2026-05-13

This document is a detailed, human-readable inventory of the workspace contents and the development attempts behind the project. It is intended to be used as source material for a later report or summary.

============================================================
1) WHAT THIS PROJECT IS
============================================================

This workspace contains a multi-stage chatbot system for Ministry of Justice BiH / Ministarstvo pravde BiH.
The target use case is legal and administrative guidance for users asking about:
- registration of associations and foundations
- required documentation and forms
- laws, rules, and regulations
- judicial exam procedures
- free legal aid
- contact points and jurisdiction / competencies
- registries and related public information

The project is split into three main layers:
- backend: API layer for serving chatbot responses
- chatbot: data collection, preprocessing, intent classification, retrieval, embeddings, evaluation, and offline pipeline logic
- frontend: Next.js user interface for chatting with the bot

Current overall state:
- frontend is visually advanced and functional as a UI shell
- backend API exists and responds, but still uses dummy keyword-based responses
- the real ML/RAG pipeline exists in chatbot/ but is not fully wired into the backend
- the project has several generations of datasets, models, and evaluation scripts that show iterative improvement attempts

============================================================
2) TOP-LEVEL DIRECTORY STRUCTURE
============================================================

Root workspace:
- backend/
- chatbot/
- frontend/

There are also large generated artifact folders inside chatbot/ such as:
- mpr_dataset_audit/
- mpr_dataset_final/
- mpr_dataset_v2/
- mpr_dataset_v3_fixed/
- mpr_dataset_v4/
- mpr_dataset_v5/
- models/
- models_v2/
- models_v3/
- models_v4/
- models_v4_embeddings/
- models_v5/
- rag_eval_results/
- rag_eval_results_embeddings/
- rag_eval_results_v5/

These folders reflect a strong development history with multiple attempts to improve scraping, chunking, retrieval, embeddings, and evaluation quality.

============================================================
3) BACKEND FOLDER - WHAT EACH FILE DOES
============================================================

Folder: backend/

3.1 main.py
- FastAPI application entry point.
- Defines the API app.
- Enables CORS for all origins during development.
- Exposes endpoints:
  - GET /
  - GET /health
  - POST /chat
  - GET /suggested-questions
  - GET /info
- Uses Pydantic models for request/response validation.
- Accepts query, session_id, and language.
- Validates empty query and overly long query.
- Calls get_dummy_response() from dummy_responses.py instead of a real ML model.

What was attempted here:
- create a structured API response instead of raw text
- add health checks and info endpoints
- add CORS so the frontend can call the API
- prepare the API surface for later ML integration
- keep a session_id field for future conversation tracking

Current limitation:
- the backend still does not call the real retrieval/model pipeline from chatbot/
- the HTTPException handler returns a dict directly, which is a simple development-stage choice
- no persistent session memory or database is implemented

3.2 dummy_responses.py
- Development placeholder for chatbot answers.
- Contains a small keyword-based knowledge base with categories:
  - registracija
  - obrasci
  - zakoni
  - procedure
- Chooses a response by keyword matches in the query.
- Returns response, confidence, category, source, and timestamp placeholder.
- Provides suggested questions for the UI.

What was attempted here:
- give the frontend something usable before the real model was integrated
- simulate confidence and category metadata
- support a basic chatbot loop while the ML pipeline was unfinished

Important note:
- this is not a real semantic or generative model
- it is a test scaffold
- it makes the backend look functional, but the intelligence is limited

3.3 requirements.txt
- Lists backend dependencies:
  - fastapi
  - uvicorn
  - pydantic
  - python-multipart
- This is a minimal backend stack.
- It suggests the backend is intentionally lightweight and waiting for deeper ML integration.

============================================================
4) CHATBOT FOLDER - CORE NLP / ML / DATA PIPELINE
============================================================

Folder: chatbot/

This is the main research and development area. It contains the real attempts to build a domain-specific legal assistant.

------------------------------------------------------------
4.1 Core idea of the chatbot pipeline
------------------------------------------------------------

The pipeline is built around:
- scraping content from mpr.gov.ba
- filtering noise and irrelevant pages
- chunking relevant text
- generating synthetic question-answer datasets
- training an intent classifier
- building retrieval indexes
- using TF-IDF and then embeddings + FAISS
- evaluating retrieval quality
- constructing a final offline chatbot pipeline

This is a hybrid system:
- rule-based heuristics
- intent classification
- keyword matching fallback
- retrieval-based answering
- optional local LLM prompt scaffolding

------------------------------------------------------------
4.2 Scraping and data collection evolution
------------------------------------------------------------

4.2.1 scraper.py
- Early or master scraper version.
- Crawls mpr.gov.ba using requests, BeautifulSoup, trafilatura, pypdf, and python-docx.
- Has retry logic, thread-local sessions, and URL normalization.
- Filters language and irrelevant content using keep/drop section rules.
- Extracts HTML and documents.
- Infers category from URL, title, and parent title.
- Creates audit-friendly output.

Improvements attempted:
- robust crawling
- retry handling
- deduplication and normalization
- better classification of content by legal topic
- support for documents beyond HTML

4.2.2 scraper_v2.py
- More structured crawling version.
- Uses a sitemap and explicit starting points.
- Includes chunking settings.
- Adds category rules and drop patterns.
- More focused on relevant legal content.

Improvements attempted:
- better start URL selection
- reduced random crawling
- more explicit category and subsection mapping
- first more serious attempt at building a content corpus for RAG

4.2.3 scraper_v3.py
- Fixed sitemap-based scraper.
- Output folder: mpr_dataset_v3_fixed.
- Uses a specific sitemap URL and filters link IDs.
- Focuses on content-heavy pages and excludes obvious noise.
- Includes chunk-size settings and more stable extraction logic.

Improvements attempted:
- fix broken sitemap handling
- stabilize URL selection
- reduce noise from the multilingual header and irrelevant pages

4.2.4 scraper_v4_curated.py
- Curated scraping version.
- Uses selected sitemap links plus a manual URL list.
- Stronger category and subsection rules.
- Better structure for training and retrieval.
- Designed to favor known important ministry pages.

Improvements attempted:
- manually prioritize the most useful pages
- reduce irrelevant crawl data
- ensure key topics like registrations, forms, exams, legal aid, and laws are captured
- improve downstream retrieval by controlling input quality

------------------------------------------------------------
4.3 Data cleaning, filtering, and final dataset creation
------------------------------------------------------------

4.3.1 postprocess.py
- Takes the audit dataset and produces the final dataset.
- Applies strict URL and text filters.
- Removes bad or short texts.
- Keeps only useful categories or texts with good domain terms.
- Chunks text into final chunks with overlap.
- Removes duplicate chunks.
- Produces:
  - final_kept_records.csv
  - final_dropped_records.csv
  - final_chunks.jsonl
  - qa_dataset_5000_style.jsonl
  - chat_finetune_dataset.jsonl
  - final_summary.json

Key parameters:
- MIN_CHUNK_CHARS = 350
- MAX_CHUNK_CHARS = 900
- OVERLAP = 120

Improvements attempted:
- aggressively remove junk content
- make chunks better for retrieval
- create both QA and chat fine-tuning outputs
- turn raw web data into structured ML-ready data

4.3.2 update_metadata_v5.py
- Reads v4 chunks and records.
- Adds page_type labels.
- Adds semantic_topic labels.
- Adds generic_penalty for generic pages like headers and navigation-like content.
- Saves chunks_v5.csv and records_v5.csv.
- Produces metadata_summary.json.

Improvements attempted:
- enrich retrieval metadata
- make scoring more context-aware
- reduce generic or menu-like content bias
- improve precision of retrieval and reranking

------------------------------------------------------------
4.4 Intent dataset creation and model training
------------------------------------------------------------

4.4.1 make_intent_dataset.py
- Generates a synthetic intent dataset with 5000 target examples.
- Includes a detailed intent schema with many subintents.
- Covers categories like:
  - registracija
  - obrasci
  - zakoni_i_propisi
  - pravna_pomoc
  - ispiti
  - registri
  - kontakt_i_nadleznosti
  - out_of_scope
- Adds typos, informal variants, and context variations.
- Produces train/val/test splits and summary statistics.

Improvements attempted:
- make the intent classifier more robust to natural language variation
- cover user phrasing variations and errors
- increase resilience against informal queries and spelling mistakes

4.4.2 train_intent_model.py
- Trains a scikit-learn pipeline:
  - TF-IDF vectorizer with 1-3 grams
  - Logistic Regression classifier with class_weight balanced
- Evaluates on validation and test sets.
- Saves models/intent_classifier.joblib.

Improvements attempted:
- create a simple but effective intent classification baseline
- improve over pure keyword matching
- give the system an explicit intent layer before retrieval

4.4.3 evaluation.py
- Evaluates intent model on the test split.
- Prints accuracy, macro F1, weighted F1, classification report, confusion matrix, and confidence analysis.
- Helps identify low-confidence cases.

4.4.4 test_intent_model.py
- Additional evaluation script.
- Looks at low-confidence predictions and failed examples.
- Useful for diagnosing intent model weaknesses.

------------------------------------------------------------
4.5 RAG / retrieval index generation history
------------------------------------------------------------

4.5.1 build_rag_index.py
- Early RAG index builder.
- Uses TF-IDF + NearestNeighbors.
- Builds a search_text field from title, category, and text.
- Saves vectorizer, nearest neighbors model, and rag_chunks.pkl.

Improvements attempted:
- create an initial lexical retrieval system
- retrieve relevant chunks from the final dataset
- provide a simple baseline before embeddings

4.5.2 build_rag_index_v2.py
- Similar TF-IDF + NearestNeighbors approach.
- Uses mpr_dataset_v4 chunks.
- Adds subsection into search_text.
- Uses word analyzer and unicode accent stripping.

Improvements attempted:
- better feature representation
- include subsection context
- slightly improve search quality and normalization

4.5.3 build_rag_index_v3.py
- Retrieval logic evolved further.
- Includes allowed-category filtering based on predicted intent.
- Uses keyword boosts and required terms logic.
- More defensive against menu noise and irrelevant chunks.

Improvements attempted:
- restrict retrieval based on intent
- boost domain-specific matches
- penalize noise and wrong topic matches
- improve precision beyond raw nearest-neighbor search

4.5.4 build_rag_index_v4.py
- Final TF-IDF retrieval generation before embeddings emphasis.
- Uses v4 dataset and richer metadata.
- Includes subsection in search_text.
- Builds a larger nearest-neighbor model.

Improvements attempted:
- better text normalization
- higher-quality chunk structure
- prepare a more stable base for evaluation and comparison

------------------------------------------------------------
4.6 Embedding-based retrieval evolution
------------------------------------------------------------

4.6.1 build_embedding_index_v4.py
- Converts v4 chunks into embeddings.
- Uses SentenceTransformer paraphrase-multilingual-MiniLM-L12-v2.
- Normalizes embeddings.
- Optionally builds FAISS IndexFlatIP.
- Saves chunk_embeddings.npy, embedding_config.joblib, and rag_chunks_v4_embeddings.pkl.

Improvements attempted:
- move from lexical retrieval to semantic retrieval
- better handle paraphrases and multilingual phrasing
- improve retrieval for questions that do not exactly match words in the source text

4.6.2 build_embedding_index_v5.py
- Final embedding index builder.
- Uses v5 chunks with more metadata.
- Includes page_type and semantic_topic in search_text.
- Builds FAISS index and saves v5 embedding artifacts.

Improvements attempted:
- improve semantic retrieval using richer metadata
- make the index more aware of page role and domain context
- strengthen ranking for legal-topic specific questions

------------------------------------------------------------
4.7 Evaluation of RAG / retrieval versions
------------------------------------------------------------

4.7.1 evaluate_rag_v4.py
- Evaluates the v4 TF-IDF retrieval pipeline.
- Loads intent model, vectorizer, and chunk data.
- Uses keyword fallback logic and intent prediction.
- Evaluates responses against a question set.

4.7.2 evaluate_rag_embeddings_v4.py
- Evaluates embedding-based v4 retrieval.
- Uses embeddings and optional FAISS.
- Compares retrieval quality against the question set.

4.7.3 evaluate_rag_embeddings_v5.py
- Evaluates the v5 embedding pipeline.
- Uses v5 metadata, embeddings, FAISS, and intent fallback.
- Includes more refined page_type and topic inference.

Improvements attempted across all evaluation scripts:
- measure which retrieval version works best
- compare lexical and semantic retrieval
- tune score boosts and fallback behavior
- identify weak query types and noisy retrieval results

------------------------------------------------------------
4.8 Test scripts for retrieval pipelines
------------------------------------------------------------

4.8.1 test_rag_pipeline.py
- Early experimental retrieval test.
- Uses intent_model + TF-IDF + NearestNeighbors.
- Tests a few question examples.
- Shows draft answers and top results.

4.8.2 test_rag_pipeline_v2.py
- Improves retrieval filtering and boosting.
- Adds more explicit intent and keyword handling.
- Includes additional test questions.

4.8.3 test_rag_pipeline_v4.py
- Later retrieval test with more sophisticated scoring.
- Uses refined rules for intent, required focus, and direct routing.
- Tries to reduce incorrect retrieval from similar but wrong topics.

Improvements attempted:
- make retrieval behavior more realistic before integrating into production
- test question routing and scoring rules manually
- compare how different heuristics change top results

------------------------------------------------------------
4.9 Final offline chatbot pipeline
------------------------------------------------------------

4.9.1 chatbot_pipeline_v1.py
- This is the most advanced offline pipeline in the repo.
- Loads:
  - intent_classifier.joblib
  - embedding_config_v5.joblib
  - chunks_v5.csv
  - faiss_index_v5.bin
  - SentenceTransformer model
- Implements many layers of logic:
  - normalization of Bosnian/Croatian/Serbian diacritics
  - intent prediction
  - special contact override
  - confidence threshold fallback
  - keyword-based intent fallback
  - page_type inference
  - semantic_topic inference
  - direct route URL mapping
  - reranking with generic penalties
  - extraction of clean source lines
  - user-friendly answer construction
  - optional local LLM prompt construction

Important internal concepts in this file:
- OLLAMA_MODEL and OLLAMA_URL are prepared
- USE_LOCAL_LLM is False, so the actual LLM path is currently disabled
- retrieve() combines intent, direct routes, and embedding retrieval
- make_user_friendly_answer() tries to turn source text into readable answers
- local_llm_answer() defines a prompt that would let a local model answer using only sources

Improvements attempted:
- build a practical hybrid chatbot engine
- add hard rules for common and high-value queries
- make retrieval smarter than plain vector search
- create a good bridge between retrieved source content and natural-language response generation

Important limitation:
- this pipeline exists in the chatbot folder, but it is not yet the production backend handler
- backend/main.py still points to dummy responses rather than this pipeline

============================================================
5) DATASET AND ARTIFACT FOLDERS
============================================================

5.1 mpr_dataset_audit/
- all_records.csv
- all_records_checkpoint.csv
- all_records_checkpoint.jsonl
- crawl_events.csv
- crawl_events_checkpoint.csv
- chunks.jsonl
- kept_records.csv
- dropped_records.csv
- needs_ocr.csv
- needs_review.csv
- summary_stats.json
- audit_report.html
- files/
- reports/

Purpose:
- store the full audit trail of scraping and filtering decisions
- keep records of what was kept, dropped, or needs manual review
- allow inspection of why some data was excluded

What it shows:
- the project was aggressively curated
- a lot of raw data was deemed low quality or irrelevant
- the final dataset is the result of heavy filtering, not direct raw scraping

5.2 mpr_dataset_final/
Contains the final ML-ready datasets:
- final_chunks.jsonl
- final_kept_records.csv
- final_dropped_records.csv
- final_summary.json
- qa_dataset_5000_style.jsonl
- chat_finetune_dataset.jsonl
- intent_dataset_v2.csv
- intent_dataset_v2.jsonl
- intent_dataset_v2_summary.json
- intent_train.csv
- intent_val.csv
- intent_test.csv

Purpose:
- hold the final cleaned and synthetic training material
- provide datasets for intent training and chat fine-tuning
- produce evaluation-ready outputs

5.3 mpr_dataset_v2/
- earlier dataset version
- shows a stepping stone in the scraping and chunking process

5.4 mpr_dataset_v3_fixed/
- fixed sitemap-based dataset version
- a more reliable version than earlier crawling attempts

5.5 mpr_dataset_v4/
- more mature chunked dataset version
- includes chunks_v4.csv and records_v4.csv
- used for improved TF-IDF and later embedding work

5.6 mpr_dataset_v5/
- final metadata-rich dataset version
- includes:
  - chunks_v5.csv
  - records_v5.csv
  - metadata_summary.json
- adds page_type and semantic_topic

============================================================
6) MODEL FOLDERS
============================================================

6.1 models/
- intent_classifier.joblib
- rag_vectorizer.joblib
- rag_nn.joblib
- rag_chunks.pkl

This is the earliest consolidated model output folder.

6.2 models_v2/
- rag_nn_v2.joblib
- rag_vectorizer_v2.joblib

6.3 models_v3/
- rag_nn_v3.joblib
- rag_vectorizer_v3.joblib

6.4 models_v4/
- rag_nn_v4.joblib
- rag_vectorizer_v4.joblib
- rag_chunks_v4.pkl

6.5 models_v4_embeddings/
- chunk_embeddings.npy
- embedding_config.joblib
- faiss_index.bin
- rag_chunks_v4_embeddings.pkl

6.6 models_v5/
- chunk_embeddings_v5.npy
- embedding_config_v5.joblib
- faiss_index_v5.bin
- rag_chunks_v5.pkl

What the model folders show:
- the project evolved from lexical retrieval to embeddings + FAISS
- multiple versions were kept for comparison
- the final direction appears to be semantic retrieval with richer metadata

============================================================
7) EVALUATION RESULT FOLDERS
============================================================

- rag_eval_results/
- rag_eval_results_embeddings/
- rag_eval_results_v5/

Purpose:
- store outputs from retrieval evaluation runs
- compare pipeline versions
- see which retrieval strategy produced the best answers

============================================================
8) FRONTEND FOLDER - WHAT EACH FILE DOES
============================================================

Folder: frontend/

Frontend technology stack:
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Heroicons

8.1 package.json
- Defines the frontend dependencies and scripts.
- Scripts:
  - dev
  - build
  - start
  - lint
- Dependencies include Next, React, React DOM, TypeScript, Heroicons.
- Dev dependencies include Tailwind, PostCSS, and Autoprefixer.

What this shows:
- a fairly standard modern Next.js setup
- frontend is meant to be lightweight and fast

8.2 app/layout.tsx
- Defines metadata for the app.
- Imports Roboto Flex from Google Fonts.
- Sets HTML lang to bs.
- Applies global stylesheet.
- Uses a clean antialiased body style.

Improvements attempted:
- add a more polished typographic base
- improve language metadata
- use a custom font instead of the default stack

8.3 styles/globals.css
- Global Tailwind base/components/utilities import.
- Applies global transitions to all elements.
- Sets body background and text colors.
- Styles scrollbars.

Improvements attempted:
- give the UI a smoother feel
- make transitions globally consistent
- improve polish with scrollbar customization

8.4 app/page.tsx
- The page entry point.
- Renders the ChatInterface component.
- This keeps the page simple and delegates the real UI to the component.

8.5 components/ChatInterface.tsx
- The main chatbot UI component.
- Holds messages, input state, loading state, and suggested questions.
- Calls the backend API for /suggested-questions and /chat.
- Displays the conversation, confidence, and category.
- Includes a simple but complete chat interaction loop.

Important UX features:
- initial bot greeting
- sidebar with suggested questions
- auto-scroll to latest message
- loading animation while waiting for response
- confidence display for bot answers
- input focus restoration after sending
- error fallback message if the API call fails

What was attempted here:
- make the chatbot feel like a real assistant, not just a text box
- guide the user with suggested prompts
- provide a visually polished legal help experience
- support both desktop and responsive mobile usage

Important note:
- there are at least two visible frontend chat component versions in the workspace history/patterns
- one older version is simpler and more utilitarian
- the newer version is much more polished, with gradients, sidebar sections, activity indicator, and better visual hierarchy

============================================================
9) MAIN DEVELOPMENT ATTEMPTS THROUGHOUT THE PROJECT
============================================================

9.1 From simple keyword bot to structured API
- first, the backend used dummy responses
- then the project added response models, health endpoints, and a frontend API contract
- this created the foundation for a real service even before ML was plugged in

9.2 From raw scraping to curated scraping
- initial scraping was broad and noisy
- later versions added sitemap constraints, manual URL selection, and category heuristics
- final versions focused on legal content that matters for users

9.3 From raw text to quality chunks
- raw pages were too large and noisy
- chunking introduced overlap and size limits
- postprocessing removed short, duplicated, and irrelevant content

9.4 From lexical search to embeddings
- first retrieval relied on TF-IDF + nearest neighbors
- later retrieval tried semantic embeddings and FAISS
- this was done to better capture user paraphrases and multilingual phrasing

9.5 From pure retrieval to hybrid answer generation
- the pipeline now combines intent classification, keyword fallback, direct routing, reranking, and source text cleaning
- the idea is to answer in a more natural way while staying grounded in source content

9.6 From generic UI to branded assistant UI
- the frontend evolved from a basic chat layout to a more polished Ministry-style interface
- visual elements, confidence display, sidebar prompts, and active-state indicators were added

============================================================
10) CURRENT LIMITATIONS AND GAPS
============================================================

- backend/main.py still uses dummy responses instead of the real chatbot_pipeline_v1 logic
- there is an integration gap between the chatbot ML work and the production API
- the frontend depends on /suggested-questions and /chat, so it works as a demo, but not yet with the full intelligence pipeline
- some files and README instructions are older than the current folder state
- multiple versioned scripts coexist, which makes it hard to know which one is the final canonical production path
- local LLM integration is scaffolded but not enabled
- session_id is accepted by the API but not actually used for memory or conversation state
- no database or persistent conversation storage is present
- no authentication is implemented

============================================================
11) WHAT THE FINAL SYSTEM APPEARS TO BE
============================================================

The best interpretation of the workspace is:

- a legal-domain chatbot prototype for BiH ministry information
- a serious NLP/RAG experimentation project
- a system that has evolved through multiple iterations of scraping, cleanup, retrieval, embeddings, and evaluation
- a frontend that is ready to present a chatbot experience
- an API that is ready structurally but still connected to dummy response logic

The strongest production-ready components appear to be:
- the curated data and final chunking pipeline
- the intent model
- the v5 metadata-enriched embeddings and FAISS artifacts
- the advanced offline pipeline in chatbot_pipeline_v1.py
- the polished Next.js frontend UI

The weakest link is the last-mile integration into the backend API.

============================================================
12) FILES THAT ARE MOST IMPORTANT FOR A REPORT
============================================================

If you need to write a clean report, the most important files are:
- backend/main.py
- backend/dummy_responses.py
- chatbot/chatbot_pipeline_v1.py
- chatbot/postprocess.py
- chatbot/update_metadata_v5.py
- chatbot/make_intent_dataset.py
- chatbot/train_intent_model.py
- chatbot/build_embedding_index_v5.py
- chatbot/evaluate_rag_embeddings_v5.py
- chatbot/scraper_v4_curated.py
- frontend/components/ChatInterface.tsx
- frontend/app/layout.tsx
- frontend/styles/globals.css
- chatbot/README.md

============================================================
13) SHORT FINAL DIAGNOSIS
============================================================

This workspace is not a toy example. It is a layered legal chatbot R&D project with real data engineering, ML training, retrieval experiments, evaluation, and UI work.

The project has already solved many hard subproblems:
- collecting and filtering domain data
- building synthetic training data
- training an intent classifier
- building TF-IDF and embedding retrieval
- improving metadata and reranking
- designing a polished frontend

What remains is the final integration step: connect the real pipeline to the backend API and then make the frontend consume the real model output instead of the dummy placeholder.

END OF DOCUMENT

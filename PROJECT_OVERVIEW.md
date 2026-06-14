# Project Overview — DAP / MPR Chatbot

**Date snapshot:** 2026-06-14

Ovaj dokument daje pregled projekta, trenutnog stanja i prioritetnih popravki u root folderu.

## 1. Šta je ovaj projekat

DAP je višeslojni chatbot sistem za sadržaje Ministarstva pravde BiH (`mpr.gov.ba`) sa fokusom na:
- registraciju udruženja i fondacija,
- obrasce i dokumentaciju,
- zakone i propise,
- pravnu pomoć,
- ispite,
- registre,
- kontakte i nadležnosti.

## 2. Arhitektura (high-level)

Projekt je podijeljen na tri glavna dijela:
- `backend/` — API sloj (`/chat`, `/health`, `/info`, `/suggested-questions`),
- `chatbot/` — scraping, dataset pipeline, treniranje modela, evaluacije, retrieval pipeline,
- `frontend/` — Next.js korisnički interfejs.

## 3. Trenutno stanje

- Frontend je funkcionalan i vizuelno uređen.
- Chatbot modul sadrži više iteracija pipeline-a i artefakata (v2/v3/v4/v5).
- U repozitoriju postoje i "dummy" i "ML" tokovi, pa je dokumentacija djelimično neusklađena.
- Root dokumentacija sadrži korisne informacije, ali nije potpuno standardizovana.

## 4. Najvažniji tehnički tok (sažeto)

1. Scraping i kuracija sadržaja (`chatbot/scraper_v4_curated.py`).
2. Filtriranje i priprema finalnih zapisa (`chatbot/postprocess.py`).
3. Chunking i metadata obogaćivanje (`mpr_dataset_v5/*`, `update_metadata_v5.py`).
4. Intent trening (`train_intent_model.py` i povezani skriptovi).
5. Embedding + FAISS indeks (`build_embedding_index_v5.py`, `models_v5/*`).
6. Runtime pipeline (`chatbot/chatbot_pipeline_v1.py`) + backend/frontend integracija.

## 5. Šta treba popraviti u root folderu (prioritetno)

### 5.1 Dokumentacija i konzistentnost
- [ ] Uskladiti tvrdnje između root dokumenata (`PROJECT_OVERVIEW.md`, `ML_MODEL_INTEGRATION.md`, ostali *.md), jer trenutno postoje kontradikcije oko backend integracije.
- [ ] Standardizovati jezik (trenutno je miješan BHS/EN stil).
- [ ] Zamijeniti apsolutne putanje relativnim putanjama u svim root dokumentima.
- [ ] Dodati jedinstven „source of truth” dokument za trenutno produkcijsko stanje (šta je stvarno aktivno, šta je eksperimentalno).

### 5.2 Struktura root dokumentacije
- [ ] Definisati jasnu ulogu svakog root `.md` fajla da nema preklapanja sadržaja.
- [ ] Dodati kratki indeks root dokumentacije (koji dokument čitati za koji cilj).
- [ ] Ukloniti zastarjele tvrdnje i datume snapshot-a bez verifikacije.

### 5.3 Developer onboarding
- [ ] Dodati ili dopuniti root `README.md` (ako ne postoji) sa minimalnim quick-start koracima za backend/frontend/chatbot.
- [ ] Navesti tačne komande za pokretanje i evaluaciju (po modulu), bez dupliciranja i konflikta među dokumentima.
- [ ] Jasno navesti zavisnosti i preduslove (Python/Node verzije, modeli, dataset artefakti).

### 5.4 Standardi održavanja
- [ ] Uvesti pravilo da se nakon svake veće promjene pipeline-a ažurira i root dokumentacija.
- [ ] Definisati konvenciju verzionisanja za dataset/model foldere (v2/v3/v4/v5) i šta je "active".
- [ ] Dodati kratku sekciju „Known limitations” i „Next steps” koja se periodično održava.

## 6. Ključni fajlovi za dalje održavanje

- `backend/main.py`
- `backend/dummy_responses.py`
- `chatbot/chatbot_pipeline_v1.py`
- `chatbot/postprocess.py`
- `chatbot/update_metadata_v5.py`
- `chatbot/train_intent_model.py`
- `chatbot/build_embedding_index_v5.py`
- `chatbot/evaluate_rag_embeddings_v5.py`
- `chatbot/scraper_v4_curated.py`
- `frontend/components/ChatInterface.tsx`

## 7. Kratka dijagnoza

Repozitorij pokazuje ozbiljan R&D rad na pravnom chatbotu, sa najviše vrijednosti u data i retrieval sloju. Najvažniji preostali posao na nivou root-a je standardizacija dokumentacije i uklanjanje kontradiktornih informacija kako bi onboarding i održavanje bili pouzdani.

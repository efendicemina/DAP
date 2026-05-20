# MPR Chatbot - Ministarstvo Pravde BiH

NLP chatbot sistem za pravnu podršku i digitalno usmjeravanje korisnika.

## Struktura projekta

```
DAP/
├── scraper.py              # Web scraper za mpr.gov.ba
├── mpr_dataset/            # Dataset (HTML, PDF, JSONL, CSV)
├── backend/                # FastAPI backend
│   ├── main.py            # FastAPI aplikacija
│   ├── dummy_responses.py # Dummy response generator
│   └── requirements.txt    # Python dependencies
└── frontend/              # Next.js frontend
    ├── app/               # Next.js app folder
    ├── components/        # React komponente
    ├── styles/            # CSS (Tailwind)
    ├── package.json       # Node.js dependencies
    ├── tailwind.config.js # Tailwind konfiguracija
    ├── tsconfig.json      # TypeScript konfiguracija
    └── .env.local         # Okružne varijable
```

## Preduvjeti

- **Python 3.9+** (za backend i scraper)
- **Node.js 18+** (za frontend)
- **npm ili yarn** (za package management)

## 1. Pokretanje Scrapera (Dataset)

### Instalacija dependencies

```bash
cd DAP
pip install requests beautifulsoup4 pandas tqdm trafilatura pypdf python-docx openpyxl
```

### Pokrenite scraper

**Kratka test verzija (30 stranica):**
```bash
python scraper.py --max-pages 30 --delay 0.8
```

**Punoj verziji (preporučeno ~5000 stranica):**
```bash
python scraper.py --max-pages 5000 --delay 1.0 --start-url https://www.mpr.gov.ba/bs --start-url https://www.mpr.gov.ba/en --start-url https://www.mpr.gov.ba/hr --start-url https://www.mpr.gov.ba/sr
```

**Svi raspoloživi parametri:**
```bash
python scraper.py --help
```

Rezultat: `mpr_dataset/records.jsonl` i `mpr_dataset/records.csv`

## 2. Pokretanje Backend-a (FastAPI)

### Instalacija dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Pokrenite FastAPI server

```bash
python main.py
```

Ili sa uvicorn direktno:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend će biti dostupan na:
- **API**: http://localhost:8000
- **API docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Testirajte backend

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Kako registrovati udruženje?"}'
```

## 3. Pokretanje Frontend-a (Next.js)

### Instalacija dependencies

```bash
cd frontend
npm install
# ili
yarn install
```

### Pokrenite dev server

```bash
npm run dev
# ili
yarn dev
```

Frontend će biti dostupan na: **http://localhost:3000**

### Build za produkciju

```bash
npm run build
npm start
```

## 4. Kompletan Flow (sve odjednom)

### Terminal 1 - Backend
```bash
cd backend
python main.py
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

### Terminal 3 - Scraper (opciono - ako trebate nove podatke)
```bash
python scraper.py --max-pages 100
```

## API Endpoints

| Method | URL | Opis |
|--------|-----|------|
| `GET` | `/` | Info i welcome |
| `GET` | `/health` | Health check |
| `GET` | `/suggested-questions` | Listu predloženih pitanja |
| `POST` | `/chat` | Pošalji query, dobij odgovor |
| `GET` | `/info` | Info o API-ju |

### Primjer POST /chat

**Request:**
```json
{
  "query": "Koja je procedura za registraciju?",
  "language": "bs"
}
```

**Response:**
```json
{
  "response": "Procedura za registraciju: 1) Priprema dokumenata, 2) Podnošenje zahtjeva...",
  "confidence": 0.75,
  "category": "procedure",
  "source": "dummy_kb",
  "timestamp": "2026-05-10T13:30:00.000Z",
  "query": "Koja je procedura za registraciju?"
}
```

## Tehnologije

### Backend
- **FastAPI** - Moderni, brz Python framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **React Hooks** - State management

### Data Collection
- **Beautiful Soup** - HTML parsing
- **Trafilatura** - Text extraction
- **PyPDF** - PDF parsing
- **Pandas** - Data management

## Sljedeća koraka

1. **Generiši sintetičke Q&A** iz `mpr_dataset/records.jsonl` koristeći LLM
2. **Gradi vector index** (Chroma/FAISS) sa embeddings-ima
3. **Implementiraj retrieval + generation** pipeline
4. **Fine-tune** LLM model na sintetičkim Q&A parovima
5. **Integracja sa pravim modelom** (zamjena dummy_responses.py sa ml_model.py)

## Napomene

- **Trenutno koristi dummy responses** - respons se generiše na osnovu ključnih riječi
- **CORS je omogućen** za sve originale - trebate omogućiti samo frontend domenu za produkciju
- **Nema baze podataka** - poruke se čuvaju u memoriji. Za produkciju trebate database (PostgreSQL, MongoDB, itd.)
- **Nema autentifikacije** - dodajte JWT/OAuth2 za produkciju

## Logovanje

Backend logira sve zahtjeve i greške. Pogledajte console output za debug informacije.

## Troubleshooting

### Frontend se ne povezuje na backend
- Provjerite da li je backend pokrenut na `http://localhost:8000`
- Provjerite `.env.local` u frontend folderu
- Otvorite DevTools (F12) i pogledajte Network tab za greške

### Backend ne počinje
- Provjerite portove: `lsof -i :8000` (Mac/Linux) ili `netstat -ano | findstr :8000` (Windows)
- Ako je port zauzet, koristite drugi port: `python main.py --port 8001`

### Scraper je spora
- Smanjite delay: `--delay 0.5` (ali budi odgovoran prema serveru)
- Povećajte max-pages postepeno

## Kontakt i Podrška

Za više informacija o projektu, pogledajte README sa specifikacijama projekta.

---

**Status**: 🚀 Development | **Verzija**: 0.1.0 | **Zadnja ažuriranja**: May 2026

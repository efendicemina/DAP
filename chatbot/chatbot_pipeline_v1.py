import re
import json
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import requests
from answer_cards import match_answer_card
from few_shot_composer import compose_few_shot_answer, answer_quality_bad as few_shot_answer_quality_bad

# Find the chatbot root directory
CHATBOT_ROOT = Path(__file__).parent
MODELS_DIR = CHATBOT_ROOT / "models"
MODELS_V5_DIR = CHATBOT_ROOT / "models_v5"
DATASET_DIR = CHATBOT_ROOT / "mpr_dataset_v5"

OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
USE_LOCAL_LLM = False

INTENT_MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
MODEL_DIR = MODELS_V5_DIR

intent_model = joblib.load(str(INTENT_MODEL_PATH))
config = joblib.load(str(MODEL_DIR / "embedding_config_v5.joblib"))
chunks_df = pd.read_csv(str(DATASET_DIR / "chunks_v5.csv"))

for col in chunks_df.columns:
    chunks_df[col] = chunks_df[col].astype(str)
index = faiss.read_index(str(MODEL_DIR / "faiss_index_v5.bin"))
embedding_model = SentenceTransformer(config["model_name"])

# Default reranker switched to multilingual mMARCO for better BS/HR/SR coverage.
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
).strip()
reranker_model = CrossEncoder(RERANKER_MODEL_NAME)

RERANK_DEBUG = os.getenv("RERANK_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}

def norm(text):
    text = str(text).lower().strip()
    text = (
        text.replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "dj")
    )
    return re.sub(r"\s+", " ", text)


def predict_intent(question):
    q = norm(question)

    contact_terms = [
        "kontakt", "telefon", "tel", "email", "e-mail", "mail",
        "kome se obratiti", "koga da kontaktiram", "gdje da se javim",
        "broj", "adresa"
    ]

    if any(term in q for term in contact_terms):
        return "kontakt_i_nadleznosti", 0.99

    intent = intent_model.predict([question])[0]
    confidence = intent_model.predict_proba([question])[0].max()

    if confidence >= 0.55:
        return intent, confidence

    fallback = keyword_intent_fallback(question)

    if fallback:
        return fallback, confidence

    return "needs_clarification", confidence


def keyword_intent_fallback(question):
    q = norm(question)

    if any(x in q for x in [
        "udruzenj", "fondacij", "nvo", "registracij", "registrovat",
        "osnovat", "statut", "promjen", "brisanje", "izbrisati",
        "taksa", "kosta", "papiri", "dokumenti", "zahtjev"
    ]):
        return "registracija"

    if any(x in q for x in ["obrazac", "formular", "formulari", "prijava"]):
        return "obrasci"

    if any(x in q for x in ["pravosudni", "strucni upravni", "ispit", "polaganje", "termini"]):
        return "ispiti"

    if any(x in q for x in [
        "pravna pomoc", "alimentacija", "otmica", "medjunarodna pravna",
        "dijete", "djeteta", "djece", "vidjanje", "vracanje",
        "prijavim", "prijaviti", "nestalo dijete", "nestalo dete", "vidim necije dijete", "vidim nečije dijete",
"izgubljeno dijete", "sumnjam na otmicu"
    ]):
        return "pravna_pomoc"

    if any(x in q for x in ["zakon", "propis", "pravilnik", "ustav", "konvencija"]):
        return "zakoni_i_propisi"

    if any(x in q for x in ["registar", "izvod", "lista", "spisak"]):
        return "registri"

    if any(x in q for x in ["pasos", "licna karta", "vozacka", "parking", "auto", "gradjevinska", "porezna"]):
        return "out_of_scope"

    return None


def infer_query_page_type(question):
    q = norm(question)

    rules = [
        ("kontakt", ["kontakt", "telefon", "email", "mail", "kome", "javim", "obratim"]),
        ("obrazac", ["obrazac", "formular", "formulari"]),
        ("taksa", ["taksa", "takse", "pristojba"]),
        ("troskovi", ["troskovi", "kosta", "cijena"]),
        ("uslovi", ["uslovi", "ko moze"]),
        ("prijava", ["prijava", "prijaviti", "podnosenje", "podnijeti"]),
        ("termini", ["termini", "kad je", "kada je"]),
        ("literatura", ["literatura"]),
        ("program", ["program"]),
        ("prirucnik", ["prirucnik"]),
        ("promjena", ["promjen", "izmjen", "adresa", "zastupnik"]),
        ("brisanje", ["brisanje", "izbrisati", "ugasiti", "prestanak"]),
        ("registar", ["registar", "registrovano", "izvod"]),
        ("dokumentacija", ["dokumenti", "papiri", "sta mi treba", "sta treba"]),
        ("zakon", ["zakon", "propis", "pravilnik", "regulise", "ustav", "konvencija"]),
    ]

    for page_type, keywords in rules:
        if any(k in q for k in keywords):
            return page_type

    return ""


def infer_query_topic(question):
    q = norm(question)

    rules = [
        ("pravosudni_ispit", ["pravosudni"]),
        ("strucni_upravni_ispit", ["strucni upravni", "sss", "vss"]),
        ("udruzenja", ["udruzenj"]),
        ("fondacije", ["fondacij"]),
        ("besplatna_pravna_pomoc", ["besplatna pravna pomoc"]),
        ("medjunarodna_pravna_pomoc", ["medjunarodna pravna pomoc"]),
        ("alimentacije", ["alimentacij"]),
        ("otmica_djece", ["otmica", "vracanje djeteta", "vidjanje djeteta"]),
        ("strane_nvo", ["strana nvo", "strane nevladine", "predstavnistvo"]),
        ("pravna_lica", ["pravna lica"]),
        ("crkve", ["crkve", "vjerske"]),
        ("registri", ["registar", "izvod iz registra"]),
        ("zakoni", ["zakon", "propisi", "ustav", "konvencija"]),
    ]

    for topic, keywords in rules:
        if any(k in q for k in keywords):
            return topic

    return ""


DIRECT_ROUTE_RULES = [
    (["trenutni", "ministar", "pravde"], "https://mpr.gov.ba/bs/ministar-pravde-bih"),
    (["ko", "ministar", "pravde"], "https://mpr.gov.ba/bs/ministar-pravde-bih"),
    (["ministar", "pravde"], "https://mpr.gov.ba/bs/ministar-pravde-bih"),
    (["osnovat", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["statut", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["papiri", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["dokumenti", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["obrazac", "promjen", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "udruzenj"], "https://mpr.gov.ba/bs/obrasci"),
    (["taksa", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["kosta", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["promjen", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["ugasiti", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["izbrisati", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["kontakt", "udruzenj"], "https://mpr.gov.ba/bs/kontakt63"),
    (["zakon", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["registar", "udruzenj"], "https://mpr.gov.ba/bs/registar-udruzenja"),

    (["osnovat", "fondacij"], "https://mpr.gov.ba/bs/kako-osnovati-fondaciju"),
    (["sta", "treba", "fondacij"], "https://mpr.gov.ba/bs/registracija-fondacije"),
    (["obrazac", "fondacij"], "https://mpr.gov.ba/bs/obrasci1"),
    (["taksa", "fondacij"], "https://mpr.gov.ba/bs/administrativna-taksa9"),
    (["kontakt", "fondacij"], "https://mpr.gov.ba/bs/kontakt2"),

    (["uslovi", "pravosudni"], "https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita-"),
    (["prijava", "pravosudni"], "https://mpr.gov.ba/bs/prijava-polaganja-pravosudnog-ispita"),
    (["nacin", "pravosudni"], "https://mpr.gov.ba/bs/nacin-polaganja-pravosudnog-ispita"),
    (["kosta", "pravosudni"], "https://mpr.gov.ba/bs/troskovi-polaganja-ispita"),
    (["kontakt", "pravosudni"], "https://mpr.gov.ba/bs/kontakt"),
    (["termini", "pravosudni"], "https://mpr.gov.ba/bs/novi-ispitni-termini"),
    (["literatura", "pravosudni"], "https://mpr.gov.ba/bs/literatura"),

    (["uslovi", "strucni", "upravni"], "https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita1"),
    (["prijaviti", "strucni", "upravni"], "https://mpr.gov.ba/bs/podnosenje-zahtjeva-za-polaganje"),
    (["formular", "strucni", "upravni"], "https://mpr.gov.ba/bs/formular-zahtjeva"),
    (["kosta", "strucni", "upravni"], "https://mpr.gov.ba/bs/troskovi-polaganja-ispita1"),
    (["kontakt", "strucni", "upravni"], "https://mpr.gov.ba/bs/kontakt12"),
    (["termini", "strucni", "upravni"], "https://mpr.gov.ba/bs/novi-ispitni-termini1"),
    (["prirucnik", "strucni", "upravni"], "https://mpr.gov.ba/bs/prirucnik-za-polaganje-strucnog-upravnog-ispita"),
    (["program", "strucni", "upravni"], "https://mpr.gov.ba/bs/program-strucnog-upravnog-ispita"),

    (["besplatna", "pravna", "pomoc"], "https://mpr.gov.ba/bs/besplatna-pravna-pomoc"),
    (["ured", "besplatna", "pravna", "pomoc"], "https://mpr.gov.ba/bs/ured-za-pruzanje-besplatne-pravne-pomoci"),
    (["medjunarodna", "pravna", "pomoc"], "https://mpr.gov.ba/bs/medjunarodna-pravna-pomoc-i-saradnja"),
    (["otmica", "djeteta"], "https://mpr.gov.ba/bs/postupanje-u-slucaju-otmice-djece"),
    (["alimentacija"], "https://mpr.gov.ba/bs/postupak-ostvarivanja-prava"),

    (["ustav"], "https://mpr.gov.ba/bs/ustav-bosne-i-hercegovine"),
    (["podzakonski"], "https://mpr.gov.ba/bs/podzakonski-akti"),
    (["javne", "konsultacije"], "https://mpr.gov.ba/bs/javne-konsultacije"),
]


def direct_route_url(question):
    q = norm(question)
    matches = []

    for keywords, url in DIRECT_ROUTE_RULES:
        if all(k in q for k in keywords):
            matches.append((len(keywords), url))

    if not matches:
        return None

    matches.sort(reverse=True)
    return matches[0][1]


def rerank_score(question, row, base_score):
    q = norm(question)
    title = norm(row.get("title", ""))
    page_type = row.get("page_type", "")
    semantic_topic = row.get("semantic_topic", "")
    generic_penalty = float(row.get("generic_penalty", 0.0) or 0.0)

    query_page_type = infer_query_page_type(question)
    query_topic = infer_query_topic(question)

    score = float(base_score)

    if query_page_type and page_type == query_page_type:
        score += 0.20

    if query_topic and semantic_topic == query_topic:
        score += 0.25

    for token in re.findall(r"\w+", q):
        if len(token) >= 5 and token in title:
            score += 0.04

    score -= generic_penalty

    if query_page_type and page_type == "ostalo":
        score -= 0.10

    if query_topic and semantic_topic == "general":
        score -= 0.08

    return score


def embedding_candidates(question, candidate_k=40):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, min(candidate_k, len(chunks_df)))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = chunks_df.iloc[idx].to_dict()
        row["base_score"] = float(score)
        row["score"] = rerank_score(question, row, score)
        results.append(row)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


def rerank_with_cross_encoder(question, results, top_k=5):
    if not results:
        return []

    pairs = [(question, str(r.get("text", ""))) for r in results]
    cross_scores = reranker_model.predict(pairs)

    import math
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for r, score in zip(results, cross_scores):
        sig_score = sigmoid(float(score))
        r["cross_score"] = float(score)
        # Kombinovani score: uzimamo originalni heuristici score (r["score"]) i dodajemo cross_score kao blagi boost
        r["score"] = r["score"] + (sig_score * 0.5)

    reranked = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    if RERANK_DEBUG:
        print("\nRERANK DEBUG (top results)")
        for r in reranked[: min(5, len(reranked))]:
            print(
                f"combined={r.get('score'):.4f} | cross_logit={r.get('cross_score'):.4f} | url={str(r.get('source_url',''))[:80]} | title={str(r.get('title',''))[:80]}"
            )

    return reranked


def retrieve(question, top_k=5):
    intent, confidence = predict_intent(question)

    if intent == "out_of_scope":
        return intent, confidence, []

    if intent == "needs_clarification" and confidence < 0.20:
        return intent, confidence, []

    forced_url = direct_route_url(question)

    if forced_url:
        forced = chunks_df[
            chunks_df["source_url"].astype(str).str.rstrip("/") == forced_url.rstrip("/")
        ].copy()

        if not forced.empty:
            results = []
            for _, row in forced.head(top_k).iterrows():
                item = row.to_dict()
                item["base_score"] = 1.0
                item["score"] = 2.0
                results.append(item)
            return intent, confidence, results

    candidates = embedding_candidates(question, candidate_k=40)

    if RERANK_DEBUG:
        print("\nBEFORE RERANK (top candidates)")
        for r in candidates[: min(5, len(candidates))]:
            print(
                f"heur={float(r.get('score',0.0)):.4f} base={float(r.get('base_score',0.0)):.4f} | url={str(r.get('source_url',''))[:80]} | title={str(r.get('title',''))[:80]}"
            )
    results = rerank_with_cross_encoder(question, candidates, top_k=top_k)

    return intent, confidence, results


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", str(text))

def clean_scraped_text(text):
    text = str(text)

    text = re.sub(r"Hrvatski Bosanski Српски English BS", " ", text)
    text = re.sub(r"Ministarstvo pravde Bosne i Hercegovine", " ", text)
    text = re.sub(r"Ministarstvo Oblasti rada Zakoni/Ugovori.*?Početna \|", " ", text)
    text = re.sub(r"Oblasti rada \|.*?\|", " ", text)

    menu_words = [
        "Ministarstvo", "Oblasti rada", "Zakoni/Ugovori",
        "Projekti/Strategije", "Publikacije/Priručnici",
        "Tenderi/Javni oglasi", "Info/Pristup informacijama"
    ]

    for word in menu_words:
        text = text.replace(word, " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_key_lines(text):
    text = clean_scraped_text(text)

    parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", text)
    clean_parts = []

    for p in parts:
        p = p.strip(" -•|")
        p = re.sub(r"\s+", " ", p).strip()

        if len(p) < 25:
            continue

        if any(noise in p.lower() for noise in [
            "hrvatski", "bosanski", "english", "početna", "pocetna",
            "oblasti rada", "zakoni/ugovori"
        ]):
            continue

        clean_parts.append(p)

    return clean_parts


def make_user_friendly_answer(question, main_source):
    q = norm(question)
    title = str(main_source.get("title", ""))
    text = clean_scraped_text(main_source.get("text", ""))
    page_type = str(main_source.get("page_type", ""))
    topic = str(main_source.get("semantic_topic", ""))

    lines = extract_key_lines(text)

    # KONTAKT
    if page_type == "kontakt" or any(x in q for x in ["kontakt", "telefon", "email", "mail", "javim", "obratim"]):
        emails = re.findall(r"[\w\.-]+@[\w\.-]+", text)
        phones = re.findall(r"(?:\+?\d{1,3}[\s/-]?)?(?:\d{2,3}[\s/-]?){2,5}\d{2,3}", text)

        answer = "Za ovu oblast možeš se obratiti nadležnoj službi Ministarstva pravde BiH."

        details = []
        if phones:
            details.append("Telefon: " + ", ".join(dict.fromkeys(phones[:3])))
        if emails:
            details.append("E-mail: " + ", ".join(dict.fromkeys(emails[:3])))

        if details:
            answer += "\n\n" + "\n".join(details)

        return answer

    # TROŠKOVI / TAKSA
    if page_type in ["taksa", "troskovi"] or any(x in q for x in ["koliko", "kosta", "košta", "taksa", "troskovi"]):
        money = re.findall(r"\d+[.,]?\d*\s*KM", text)
        money = list(dict.fromkeys(money))

        if money:
            answer = "Prema dostupnim informacijama, za ovu uslugu se spominju sljedeći iznosi:\n"
            answer += "\n".join([f"- {m}" for m in money[:8]])
            return answer

    # REGISTRACIJA UDRUŽENJA
    if "udruzenj" in q and any(x in q for x in ["sta", "treba", "dokumenti", "papiri", "registracij"]):
        relevant = [
            l for l in lines
            if any(k in norm(l) for k in [
                "obrazac", "osnivack", "statut", "licnih karata",
                "pasosa", "odluku", "potpis", "administrativne takse"
            ])
        ]

        answer = "Za registraciju udruženja potrebno je pripremiti zahtjev i prateću dokumentaciju."
        if relevant:
            answer += "\n\nNajvažnije stavke su:\n"
            answer += "\n".join([f"- {r}" for r in relevant[:7]])
        return answer

    # OSNIVANJE UDRUŽENJA
    if "udruzenj" in q and any(x in q for x in ["osnov", "napravim", "formiram"]):
        relevant = [
            l for l in lines
            if any(k in norm(l) for k in ["najmanje tri", "osnivaca", "osnivacki akt", "statut", "organa upravljanja"])
        ]

        answer = "Udruženje se može osnovati ako postoje najmanje tri osnivača."
        if relevant:
            answer += "\n\nUkratko:\n"
            answer += "\n".join([f"- {r}" for r in relevant[:5]])
        return answer

    # FONDACIJA
    if "fondacij" in q:
        relevant = [
            l for l in lines
            if any(k in norm(l) for k in ["fondacija", "osnivac", "statut", "upravni odbor", "obrazac"])
        ]

        answer = "Za fondaciju je potrebno pratiti postupak registracije fondacije pri Ministarstvu pravde BiH."
        if relevant:
            answer += "\n\nNajvažnije informacije su:\n"
            answer += "\n".join([f"- {r}" for r in relevant[:6]])
        return answer

    # PRAVNA POMOĆ
    if "pravna_pomoc" in topic or "pravna pomoc" in q:
        relevant = [
            l for l in lines
            if any(k in norm(l) for k in ["besplatna pravna pomoc", "pravo", "korisnici", "postupak", "zahtjev"])
        ]

        answer = "Besplatna pravna pomoć je dostupna u postupcima u kojima korisnici ostvaruju ili štite svoja prava i zakonom zaštićene interese."
        if relevant:
            answer += "\n\nPrema dostupnim informacijama:\n"
            answer += "\n".join([f"- {r}" for r in relevant[:5]])
        return answer

    # ISPITI
    if "ispit" in q or "pravosudni" in q or "strucni upravni" in q:
        relevant = [
            l for l in lines
            if any(k in norm(l) for k in ["ispit", "polaganje", "zahtjev", "uslovi", "troskovi", "termini"])
        ]

        answer = "Informacije o ispitu se nalaze u odgovarajućoj sekciji Ministarstva pravde BiH."
        if relevant:
            answer += "\n\nRelevantno za tvoje pitanje:\n"
            answer += "\n".join([f"- {r}" for r in relevant[:5]])
        return answer

    # FALLBACK
    selected = lines[:4]
    if selected:
        return "Prema dostupnim informacijama:\n" + "\n".join([f"- {s}" for s in selected])

    return "Pronašla sam relevantan izvor, ali tekst nije dovoljno jasan za automatsko formulisanje odgovora."

def build_context_for_llm(results, max_chars_per_source=1800):
    blocks = []
    seen = set()

    for i, r in enumerate(results[:3], start=1):
        url = str(r.get("source_url", "")).rstrip("/")

        if not url or url in seen:
            continue

        seen.add(url)

        text = clean_scraped_text(r.get("text", ""))
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:max_chars_per_source]

        blocks.append(
            f"Izvor {i}\n"
            f"Naslov: {r.get('title', '')}\n"
            f"URL: {url}\n"
            f"Tip stranice: {r.get('page_type', '')}\n"
            f"Tema: {r.get('semantic_topic', '')}\n"
            f"Sadržaj: {text}"
        )

    return "\n\n".join(blocks)


def local_llm_answer(question, results, intent=None, confidence=0.0):
    context = build_context_for_llm(results)

    prompt = f"""
Ti si korisnički chatbot Ministarstva pravde Bosne i Hercegovine.

Odgovori isključivo na osnovu izvora ispod.
Nemoj izmišljati podatke.
Nemoj spominjati chunkove, RAG, model, embeddinge ili tehničke detalje.
Odgovori prirodno, jasno i gramatički ispravno na bosanskom jeziku.
Ako korisnik pita neformalno, i dalje odgovori profesionalno i jednostavno.
Ako izvor ne sadrži dovoljno informacija, reci to jasno i uputi korisnika na priložene izvore.
Odgovor treba biti user-friendly, kao pravi chatbot.

Format:
- Kratak direktan odgovor.
- Ako postoji procedura, navedi je u kratkim koracima.
- Ako postoje iznosi, telefoni, e-mailovi ili obrasci, jasno ih izdvoji.
- Ne piši listu izvora na kraju, aplikacija ih prikazuje posebno.

Pitanje korisnika:
{question}

Intent:
{intent}

Pouzdanje:
{confidence}

Izvori:
{context}

Odgovor:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            return None

        return answer

    except Exception as e:
        return None

def detect_question_goal(question):
    q = norm(question)

    if any(x in q for x in ["kako", "na koji nacin", "procedura", "postupak"]):
        return "procedure"

    if any(x in q for x in ["sta treba", "šta treba", "dokumenti", "dokumentacija", "papiri", "priloziti", "priložiti"]):
        return "requirements"

    if any(x in q for x in ["koliko", "kosta", "košta", "cijena", "taksa", "troskovi", "troškovi", "uplata"]):
        return "fee"

    if any(x in q for x in ["kontakt", "telefon", "email", "mail", "kome", "gdje da se javim", "obratim"]):
        return "contact"

    if any(x in q for x in ["obrazac", "formular", "formulari", "prijava"]):
        return "form"

    if any(x in q for x in ["ko je", "sta je", "šta je", "definisi", "definiši", "znaci", "znači"]):
        return "definition"

    if any(x in q for x in ["rok", "kada", "kad", "termin", "datum"]):
        return "date"

    return "general"


def normalize_fact_text(text):
    text = str(text)
    text = clean_scraped_text(text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([.!?])([A-ZČĆŠĐŽ])", r"\1 \2", text)

    text = text.strip(" -•|;\n\t")

    # ukloni početne oznake iz lista: a), b), 1), -
    text = re.sub(r"^[a-z]\)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+[\).\s]+", "", text)
    text = re.sub(r"^[-•]\s*", "", text)

    if text:
        text = text[0].upper() + text[1:]

    return text.strip()


def split_into_fact_candidates(text):
    text = clean_scraped_text(text)

    # čuvamo i rečenice i liste
    parts = re.split(
        r"(?<=[.!?])\s+|\n+|\s+(?=[a-z]\)\s)|\s+(?=\d+[\).]\s)|\s+[-•]\s+",
        text
    )

    facts = []

    for part in parts:
        fact = normalize_fact_text(part)

        if len(fact) < 35:
            continue

        if len(fact) > 450:
            fact = fact[:450].rsplit(" ", 1)[0] + "."

        bad_patterns = [
            "hrvatski bosanski",
            "english",
            "početna",
            "pocetna",
            "oblasti rada",
            "zakoni/ugovori",
            "projekti/strategije",
            "publikacije/priručnici",
            "tenderi/javni oglasi",
        ]

        if any(bad in norm(fact) for bad in bad_patterns):
            continue

        facts.append(fact)

    return facts


def fact_score(question, fact, source):
    q = norm(question)
    f = norm(fact)

    q_tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 4]

    score = 0.0

    for token in q_tokens:
        if token in f:
            score += 1.0

    goal = detect_question_goal(question)
    page_type = str(source.get("page_type", ""))

    goal_keywords = {
        "procedure": ["postupak", "podnosi", "podnijeti", "zahtjev", "prijava", "registracija", "upis"],
        "requirements": ["potrebno", "prilaže", "prilaze", "dokument", "obrazac", "statut", "odluka", "dokaz"],
        "fee": ["km", "taksa", "pristojba", "uplata", "račun", "racun", "troš", "tros"],
        "contact": ["telefon", "email", "e-mail", "kontakt", "fax", "faks"],
        "form": ["obrazac", "formular", "zahtjev", "prijava"],
        "definition": ["je", "predstavlja", "znači", "znaci", "podrazumijeva"],
        "date": ["termin", "datum", "rok", "godine"],
    }

    for kw in goal_keywords.get(goal, []):
        if kw in f:
            score += 2.0

    if goal == page_type:
        score += 2.5

    if len(fact) > 220:
        score -= 0.5

    if fact.count(",") > 8:
        score -= 0.5

    return score


def extract_relevant_facts(question, results, max_facts=7):
    scored = []
    seen = set()

    for source_rank, source in enumerate(results[:4]):
        text = source.get("text", "")
        facts = split_into_fact_candidates(text)

        for fact in facts:
            key = norm(fact)[:180]

            if key in seen:
                continue

            seen.add(key)

            score = fact_score(question, fact, source)

            # blagi bonus za viši retrieval rank
            score += max(0, 1.5 - source_rank * 0.4)

            if score > 0:
                scored.append((score, fact, source))

    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:max_facts]


def make_fact_readable(fact):
    fact = normalize_fact_text(fact)

    replacements = {
        "podnosilac zahtjeva": "podnosilac zahtjeva",
        "podnositelj zahtjeva": "podnosilac zahtjeva",
        "ovjerenu": "ovjerenu",
        "u składu": "u skladu",
        "pristojbe": "takse",
        "pristojba": "taksa",
    }

    for old, new in replacements.items():
        fact = fact.replace(old, new)

    return fact


def compose_natural_answer(question, results, intent=None):
    goal = detect_question_goal(question)
    facts = extract_relevant_facts(question, results, max_facts=7)

    if not facts:
        return (
            "Pronašla sam relevantne izvore, ali u njima nema dovoljno jasno izdvojenih informacija "
            "da bih sigurno formulisala odgovor. Najbolje je provjeriti izvore ispod."
        )

    clean_facts = [make_fact_readable(fact) for _, fact, _ in facts]

    intro_by_goal = {
        "procedure": "Postupak se može sažeti ovako:",
        "requirements": "Za ovaj postupak potrebno je pripremiti sljedeće:",
        "fee": "Prema dostupnim informacijama, za ovo pitanje su relevantni sljedeći iznosi ili podaci o uplati:",
        "contact": "Za ovu oblast možeš koristiti sljedeće kontakt informacije:",
        "form": "Za ovaj postupak relevantni su sljedeći obrasci ili zahtjevi:",
        "definition": "Prema dostupnim informacijama:",
        "date": "Prema dostupnim informacijama o terminima i rokovima:",
        "general": "Prema dostupnim informacijama:"
    }

    answer = intro_by_goal.get(goal, "Prema dostupnim informacijama:")

    if goal == "contact":
        contact_text = " ".join(clean_facts)
        phones = list(dict.fromkeys(re.findall(
            r"(?:\+387\s?)?(?:\d{2,3}[\s/-]?){2,5}\d{2,3}",
            contact_text
        )))
        emails = list(dict.fromkeys(re.findall(r"[\w\.-]+@[\w\.-]+", contact_text)))

        details = []
        if phones:
            details.append("Telefon: " + ", ".join(phones[:3]))
        if emails:
            details.append("E-mail: " + ", ".join(emails[:3]))

        if details:
            return answer + "\n\n" + "\n".join(details)

    if goal == "fee":
        money_facts = [f for f in clean_facts if re.search(r"\d+[.,]?\d*\s*KM", f)]

        if money_facts:
            answer += "\n\n"
            answer += "\n".join([f"- {f}" for f in money_facts[:5]])
            return answer

    # za procedure/requirements/form/general koristi kratku listu, ali čistu
    selected = clean_facts[:5]

    # ako je samo jedna dobra činjenica, napiši kao pasus
    if len(selected) == 1:
        return answer + "\n\n" + selected[0]

    answer += "\n\n"
    answer += "\n".join([f"- {fact}" for fact in selected])

    return answer


def answer_quality_is_bad(answer):
    a = norm(answer)

    bad_signals = [
        "hrvatski bosanski",
        "oblasti rada",
        "zakoni/ugovori",
        "projekti/strategije",
        "publikacije/prirucnici",
        "prema dostupnim informacijama:\n- stupku",
    ]

    if any(signal in a for signal in bad_signals):
        return True

    if len(answer.strip()) < 50:
        return True

    return False

def generate_answer(question, results, intent=None, confidence=0.0):
    if not results:
        if intent == "needs_clarification":
            return {
                "answer": (
                    "Mogu pomoći, ali mi treba malo preciznije pitanje. "
                    "Na primjer, možeš pitati za registraciju udruženja, fondaciju, ispit, obrazac, taksu ili kontakt."
                ),
                "sources": []
            }

        return {
            "answer": (
                "Ovo pitanje vjerovatno nije u nadležnosti Ministarstva pravde BiH "
                "ili trenutno nemam dovoljno relevantnih informacija u bazi."
            ),
            "sources": []
        }

    answer_text = None

    if USE_LOCAL_LLM:
        answer_text = local_llm_answer(
            question=question,
            results=results,
            intent=intent,
            confidence=confidence
        )

    # Primary non-LLM answer: few-shot composer (fact extraction + structured formatting).
    if not answer_text or answer_quality_is_bad(answer_text):
        answer_text = compose_few_shot_answer(question, results, intent=intent)

    # Safety fallback to existing composer if few-shot output looks bad.
    if not answer_text or few_shot_answer_quality_bad(answer_text):
        answer_text = compose_natural_answer(question, results, intent=intent)

    sources = []
    seen_urls = set()

    for r in results[:3]:
        url = str(r.get("source_url", "")).rstrip("/")

        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "title": r.get("title", ""),
                "url": r.get("source_url", ""),
                "score": round(float(r.get("score", 0)), 4),
                "page_type": r.get("page_type", ""),
                "semantic_topic": r.get("semantic_topic", "")
            })

    return {
        "answer": answer_text,
        "sources": sources
    }
      
def is_smalltalk(question):
    q = norm(question)
    return any(x in q for x in [
        "cao", "ćao", "zdravo", "hej", "hello", "hi",
        "kako si", "sta ima", "šta ima"
    ])


def is_harmful_request(question):
    q = norm(question)

    harmful_patterns = [
        "kako da otmem",
        "kako oteti",
        "hoću da otmem",
        "hocu da otmem",
        "kako da sakrijem",
        "kako da prevarim",
        "kako falsifikovati",
    ]

    return any(p in q for p in harmful_patterns)


def ask(question):
    if is_smalltalk(question):
        return {
            "question": question,
            "intent": "smalltalk",
            "confidence": 1.0,
            "answer": (
                "Zdravo! 😊 Mogu ti pomoći da pronađeš informacije sa stranice Ministarstva pravde BiH, "
                "na primjer o registraciji udruženja i fondacija, ispitima, obrascima, taksama, kontaktima i pravnoj pomoći."
            ),
            "sources": []
        }

    if is_harmful_request(question):
        return {
            "question": question,
            "intent": "safety_block",
            "confidence": 1.0,
            "answer": (
                "Ne mogu pomoći s uputama za nezakonite ili štetne radnje. "
                "Ako se pitanje odnosi na zakonit postupak, prijavu, zaštitu prava ili nadležnu instituciju, "
                "mogu pomoći da pronađeš relevantne informacije."
            ),
            "sources": []
        }

    intent, confidence, results = retrieve(question, top_k=5)
    response = generate_answer(question, results, intent=intent, confidence=confidence)

    return {
        "question": question,
        "intent": intent,
        "confidence": round(float(confidence), 4),
        "answer": response["answer"],
        "sources": response["sources"]
    }

if __name__ == "__main__":
    print("MPR Chatbot pipeline v1")
    print("Upiši pitanje ili 'exit' za izlaz.\n")

    while True:
        q = input("Pitanje: ").strip()

        if q.lower() in ["exit", "quit", "kraj"]:
            break

        result = ask(q)

        print("\nIntent:", result["intent"], "| confidence:", result["confidence"])
        print("\nOdgovor:")
        print(result["answer"])

        print("\nIzvori:")
        for i, src in enumerate(result["sources"], start=1):
            print(f"{i}. {src['title']} - {src['url']}")

        print("\n" + "=" * 100 + "\n")
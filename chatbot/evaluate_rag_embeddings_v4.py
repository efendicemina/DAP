import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    from sklearn.metrics.pairwise import cosine_similarity
    FAISS_AVAILABLE = False

EVAL_PATH = Path("rag_eval_questions.csv")

INTENT_MODEL_PATH = "models/intent_classifier.joblib"
EMB_DIR = Path("models_v4_embeddings")

OUTPUT_DIR = Path("rag_eval_results_embeddings")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

intent_model = joblib.load(INTENT_MODEL_PATH)
config = joblib.load(EMB_DIR / "embedding_config.joblib")

chunks_df = pd.read_pickle(EMB_DIR / "rag_chunks_v4_embeddings.pkl")
embeddings = np.load(EMB_DIR / "chunk_embeddings.npy")

embedding_model = SentenceTransformer(config["embedding_model_name"])

if FAISS_AVAILABLE:
    index = faiss.read_index(str(EMB_DIR / "faiss_index.bin"))

eval_df = pd.read_csv(EVAL_PATH)

def norm(text):
    text = str(text).lower().strip()
    text = (
        text.replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "dj")
    )
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_url(url):
    return str(url).strip().rstrip("/")

def keyword_intent_fallback(q):
    qn = norm(q)

    if any(x in qn for x in [
        "udruzenj", "udruzenje", "fondacij", "fondacija", "nvo",
        "registracij", "registrovat", "osnovat", "osniva", "statut",
        "osnivac", "osnivaci", "promjen", "izmjen", "brisanje",
        "izbrisati", "ugasiti", "registar udruzenja", "registar fondacija",
        "papiri", "dokumenti", "racun", "taksa", "kosta", "predaje",
        "zahtjev", "zastupnik", "strana", "strane", "nevladine",
        "predstavnistvo", "crkve", "vjerske", "prestanak"
    ]):
        return "registracija"

    if any(x in qn for x in [
        "obrazac", "formular", "formulari", "prijava",
        "obrazac 1", "obrazac 2", "obrazac 3", "vodic"
    ]):
        return "obrasci"

    if any(x in qn for x in [
        "pravosudni", "strucni upravni", "ispit",
        "polaganje", "termini", "literatura", "program ispita"
    ]):
        return "ispiti"

    if any(x in qn for x in [
        "besplatna pravna pomoc", "besplatnu pravnu pomoc",
        "pravna pomoc", "alimentacija", "otmica djeteta",
        "otmica djece", "medjunarodna pravna pomoc"
    ]):
        return "pravna_pomoc"

    if any(x in qn for x in [
        "zakon", "zakoni", "propis", "pravilnik", "ustav",
        "podzakonski", "konvencija", "ugovor", "regulise"
    ]):
        return "zakoni_i_propisi"

    if any(x in qn for x in [
        "kontakt", "telefon", "email", "mail", "kome",
        "koga", "javim", "obratim", "adresa", "broj"
    ]):
        return "kontakt_i_nadleznosti"

    if any(x in qn for x in [
        "registar", "lista", "spisak", "registrovano", "javnost"
    ]):
        return "registri"

    if any(x in qn for x in [
        "pasos", "licna karta", "vozacka", "parking",
        "auto", "gradjevinska", "porezna"
    ]):
        return "out_of_scope"

    return None

def predict_intent(q):
    qn = norm(q)

    contact_terms = [
        "kontakt", "telefon", "tel", "email", "e-mail", "mail",
        "kome se obratiti", "koga da kontaktiram", "gdje da se javim",
        "broj", "adresa"
    ]

    if any(term in qn for term in contact_terms):
        return "kontakt_i_nadleznosti", 0.99

    model_intent = intent_model.predict([q])[0]
    confidence = intent_model.predict_proba([q])[0].max()

    if confidence >= 0.55:
        return model_intent, confidence

    fallback = keyword_intent_fallback(q)

    if fallback:
        return fallback, confidence

    return "needs_clarification", confidence

DIRECT_ROUTE_RULES = [
    # registracija - udruženja
    (["osnovat", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["osniva", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["statut", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["koliko", "ljudi", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["papiri", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["dokumenti", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["gdje", "predaje", "zahtjev", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["obrazac", "promjen", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "udruzenj"], "https://mpr.gov.ba/bs/obrasci"),
    (["formular", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["vodic", "udruzenj"], "https://mpr.gov.ba/bs/obrasci"),
    (["taksa", "promjen", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa84"),
    (["taksa", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa84"),
    (["kosta", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa84"),
    (["kosta", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["taksa", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["racun", "taksa", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["promjen", "zastupnik", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["promjen", "podataka", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["promjen", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["adres", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["ugasiti", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["izbrisati", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["brisanje", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["kontakt", "udruzenj"], "https://mpr.gov.ba/bs/kontakt63"),
    (["kome", "javim", "udruzenj"], "https://mpr.gov.ba/bs/kontakt63"),
    (["ko", "radi", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/kontakt63"),
    (["pise", "zakon", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["zakon", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["zakon", "fondacij"], "https://mpr.gov.ba/bs/udruzenja"),
    (["regulise", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["registrovano", "udruzenj"], "https://mpr.gov.ba/bs/registar-udruzenja"),
    (["registar", "udruzenj"], "https://mpr.gov.ba/bs/registar-udruzenja"),
    (["registar", "fondacij"], "https://mpr.gov.ba/bs/registar-fondacija"),

    # fondacije / NVO / crkve
    (["osnovat", "fondacij"], "https://mpr.gov.ba/bs/kako-osnovati-fondaciju"),
    (["osniva", "fondacij"], "https://mpr.gov.ba/bs/kako-osnovati-fondaciju"),
    (["sta", "treba", "fondacij"], "https://mpr.gov.ba/bs/registracija-fondacije"),
    (["dokumenti", "fondacij"], "https://mpr.gov.ba/bs/registracija-fondacije"),
    (["obrazac", "fondacij"], "https://mpr.gov.ba/bs/obrasci1"),
    (["taksa", "fondacij"], "https://mpr.gov.ba/bs/administrativna-taksa9"),
    (["kosta", "fondacij"], "https://mpr.gov.ba/bs/administrativna-taksa9"),
    (["kontakt", "fondacij"], "https://mpr.gov.ba/bs/kontakt2"),
    (["strana", "nvo"], "https://mpr.gov.ba/bs/potrebna-dokumentacija5"),
    (["strane", "nevladine"], "https://mpr.gov.ba/bs/potrebna-dokumentacija5"),
    (["predstavnistvo", "nvo", "obrazac"], "https://mpr.gov.ba/bs/obrasci4"),
    (["predstavnistvo", "nvo", "taksa"], "https://mpr.gov.ba/bs/administrativna-taksa21"),
    (["pravna", "lica", "upisati"], "https://mpr.gov.ba/bs/upis-u-registar-promjene-i-brisanje-iz-registra"),
    (["formulari", "pravna", "lica"], "https://mpr.gov.ba/bs/formulari3"),
    (["javnost", "registra"], "https://mpr.gov.ba/bs/javnost-registra"),
    (["crkve", "upis"], "https://mpr.gov.ba/bs/upis-u-registar"),
    (["crkva", "promjena"], "https://mpr.gov.ba/bs/upis-promjena-u-registru"),
    (["prestanak", "crkve"], "https://mpr.gov.ba/bs/prestanak-rada"),
    (["formulari", "crkve"], "https://mpr.gov.ba/bs/formulari5"),
    (["taksa", "crkve"], "https://mpr.gov.ba/bs/administrativne-takse87"),

    # pravosudni ispit
    (["uslovi", "pravosudni"], "https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita-"),
    (["prijaviti", "pravosudni"], "https://mpr.gov.ba/bs/prijava-polaganja-pravosudnog-ispita"),
    (["prijava", "pravosudni"], "https://mpr.gov.ba/bs/prijava-polaganja-pravosudnog-ispita"),
    (["nacin", "pravosudni"], "https://mpr.gov.ba/bs/nacin-polaganja-pravosudnog-ispita"),
    (["izgleda", "pravosudni"], "https://mpr.gov.ba/bs/nacin-polaganja-pravosudnog-ispita"),
    (["kosta", "pravosudni"], "https://mpr.gov.ba/bs/troskovi-polaganja-ispita"),
    (["troskovi", "pravosudni"], "https://mpr.gov.ba/bs/troskovi-polaganja-ispita"),
    (["kontakt", "pravosudni"], "https://mpr.gov.ba/bs/kontakt"),
    (["javiti", "pravosudni"], "https://mpr.gov.ba/bs/kontakt"),
    (["termini", "pravosudni"], "https://mpr.gov.ba/bs/novi-ispitni-termini"),
    (["kad", "pravosudni"], "https://mpr.gov.ba/bs/novi-ispitni-termini"),
    (["literatura", "pravosudni"], "https://mpr.gov.ba/bs/literatura"),
    (["komisija", "pravosudni"], "https://mpr.gov.ba/bs/rjesenje-o-imenovanju-povjerenstva-za-polaganje-pravosudnog-ispita-na-nivou-bosne-i-hercegovine"),

    # stručni upravni ispit
    (["moze", "strucni", "upravni"], "https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita1"),
    (["uslovi", "strucni", "upravni"], "https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita1"),
    (["prijaviti", "strucni", "upravni"], "https://mpr.gov.ba/bs/podnosenje-zahtjeva-za-polaganje"),
    (["formular", "strucni", "upravni"], "https://mpr.gov.ba/bs/formular-zahtjeva"),
    (["polaze", "strucni", "upravni"], "https://mpr.gov.ba/bs/nacin-polaganja-ispita"),
    (["kosta", "strucni", "upravni"], "https://mpr.gov.ba/bs/troskovi-polaganja-ispita1"),
    (["kontakt", "strucni", "upravni"], "https://mpr.gov.ba/bs/kontakt12"),
    (["email", "strucni", "upravni"], "https://mpr.gov.ba/bs/kontakt12"),
    (["termini", "strucni", "upravni"], "https://mpr.gov.ba/bs/novi-ispitni-termini1"),
    (["prirucnik", "strucni", "upravni"], "https://mpr.gov.ba/bs/prirucnik-za-polaganje-strucnog-upravnog-ispita"),
    (["program", "strucni", "upravni"], "https://mpr.gov.ba/bs/program-strucnog-upravnog-ispita"),
    (["sss", "uslovi"], "https://mpr.gov.ba/bs/uslovi-za-polaganje-ispita5"),
    (["sss", "formular"], "https://mpr.gov.ba/bs/formular-zahtjeva-za-sss"),
    (["vss", "formular"], "https://mpr.gov.ba/bs/formular-zahtjeva-za-vss"),
    (["oslobadjanje", "strucnog"], "https://mpr.gov.ba/bs/oslobadjanje-od-polaganja-strucnog-upravnog-ispita-priznavanje-strucnog-upravnog-ispita"),
    (["takse", "oslobadjanje", "ispita"], "https://mpr.gov.ba/bs/administrativne-takse"),
    (["javiti", "sss", "ispit"], "https://mpr.gov.ba/bs/kontakt13"),
]

def direct_route_url(q):
    qn = norm(q)

    matches = []

    for keywords, url in DIRECT_ROUTE_RULES:
        if all(k in qn for k in keywords):
            matches.append((len(keywords), url))

    if not matches:
        return None

    matches.sort(reverse=True)
    return matches[0][1]

def embedding_search(q, top_k=5, candidate_k=80):
    query_embedding = embedding_model.encode(
        [q],
        normalize_embeddings=True
    ).astype("float32")

    if FAISS_AVAILABLE:
        scores, indices = index.search(query_embedding, min(candidate_k, len(chunks_df)))
        scores = scores[0]
        indices = indices[0]
    else:
        sims = cosine_similarity(query_embedding, embeddings)[0]
        indices = sims.argsort()[::-1][:candidate_k]
        scores = sims[indices]

    results = []

    for score, idx in zip(scores, indices):
        if idx < 0:
            continue

        row = chunks_df.iloc[idx].to_dict()
        row["base_score"] = float(score)
        row["boost"] = 0.0
        row["score"] = float(score)
        results.append(row)

    return results[:top_k]

def search(q, top_k=5):
    intent, confidence = predict_intent(q)

    forced_url = direct_route_url(q)

    if forced_url:
        forced = chunks_df[
            chunks_df["source_url"].astype(str).str.rstrip("/") == forced_url.rstrip("/")
        ].copy()

        if not forced.empty:
            forced_results = []

            for _, row in forced.head(top_k).iterrows():
                item = row.to_dict()
                item["base_score"] = 1.0
                item["boost"] = 1.0
                item["score"] = 2.0
                item["predicted_intent"] = intent
                item["intent_confidence"] = confidence
                forced_results.append(item)

            return intent, confidence, forced_results

    if intent == "out_of_scope":
        return intent, confidence, []

    results = embedding_search(q, top_k=top_k)

    for r in results:
        r["predicted_intent"] = intent
        r["intent_confidence"] = confidence

    return intent, confidence, results

def reciprocal_rank(results, expected_url):
    expected_url = normalize_url(expected_url)

    for i, r in enumerate(results, start=1):
        if normalize_url(r["source_url"]) == expected_url:
            return 1 / i

    return 0.0

rows = []

for _, item in eval_df.iterrows():
    question = item["question"]
    expected_url = normalize_url(item["expected_url"])

    intent, confidence, results = search(question, top_k=5)

    urls = [normalize_url(r["source_url"]) for r in results]

    rows.append({
        "question": question,
        "expected_url": expected_url,
        "predicted_intent": intent,
        "intent_confidence": confidence,
        "top1_url": urls[0] if urls else "",
        "top1_title": results[0]["title"] if results else "",
        "top1_score": results[0]["score"] if results else 0,
        "hit_top1": expected_url == (urls[0] if urls else ""),
        "hit_top3": expected_url in urls[:3],
        "hit_top5": expected_url in urls[:5],
        "reciprocal_rank": reciprocal_rank(results, expected_url),
        "top5_urls": " | ".join(urls),
    })

results_df = pd.DataFrame(rows)

summary = {
    "samples": len(results_df),
    "top1_accuracy": round(results_df["hit_top1"].mean(), 4),
    "top3_recall": round(results_df["hit_top3"].mean(), 4),
    "top5_recall": round(results_df["hit_top5"].mean(), 4),
    "mrr": round(results_df["reciprocal_rank"].mean(), 4),
}

by_intent = (
    results_df.groupby("predicted_intent")
    .agg(
        samples=("question", "count"),
        top1_accuracy=("hit_top1", "mean"),
        top3_recall=("hit_top3", "mean"),
        top5_recall=("hit_top5", "mean"),
        mrr=("reciprocal_rank", "mean"),
    )
    .reset_index()
)

results_df.to_csv(OUTPUT_DIR / "rag_embedding_eval_detailed.csv", index=False)
by_intent.to_csv(OUTPUT_DIR / "rag_embedding_eval_by_intent.csv", index=False)

with open(OUTPUT_DIR / "rag_embedding_eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n======================")
print("EMBEDDING RAG EVALUATION")
print("======================")
print(summary)
print("\nBY INTENT")
print(by_intent)

print("\nFAILED TOP-1")
failed = results_df[results_df["hit_top1"] == False].head(30)

for _, row in failed.iterrows():
    print("=" * 100)
    print("Q:", row["question"])
    print("EXPECTED:", row["expected_url"])
    print("TOP1:", row["top1_url"])
    print("TITLE:", row["top1_title"])
    print("INTENT:", row["predicted_intent"])
    print("SCORE:", round(row["top1_score"], 4))
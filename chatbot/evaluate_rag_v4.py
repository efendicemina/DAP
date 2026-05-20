import re
import joblib
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# ============================
# PATHS
# ============================

EVAL_PATH = Path("rag_eval_questions.csv")

INTENT_MODEL_PATH = "models/intent_classifier.joblib"
VECTORIZER_PATH = "models_v4/rag_vectorizer_v4.joblib"
CHUNKS_PATH = "models_v4/rag_chunks_v4.pkl"

OUTPUT_DIR = Path("rag_eval_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================
# LOAD
# ============================

intent_model = joblib.load(INTENT_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
chunks_df = pd.read_pickle(CHUNKS_PATH)

eval_df = pd.read_csv(EVAL_PATH)

# ============================
# HELPERS
# ============================

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
        "izbrisati", "ugasiti", "registar udruzenja", "registar fondacija"
    ]):
        return "registracija"

    if any(x in qn for x in [
        "obrazac", "formular", "prijava", "obrazac 1", "obrazac 2",
        "obrazac 3", "vodic", "vodič", "formulari", "obrasci"
    ]):
        return "obrasci"

    if any(x in qn for x in [
        "pravosudni", "strucni upravni", "stručni upravni",
        "ispit", "polaganje", "termini", "literatura", "program ispita"
    ]):
        return "ispiti"

    if any(x in qn for x in [
        "besplatna pravna pomoc", "besplatnu pravnu pomoc",
        "pravna pomoc", "alimentacija", "otmica djeteta",
        "otmica djece", "medjunarodna pravna pomoc", "međunarodna pravna pomoc", "medunarodna pravna pomoć", "međunarodna pravna pomoć", "pravna pomoc u inostranstvu", "pravna pomoć u inostranstvu", "pravna pomoc inostranstvo", "pravna pomoć inostranstvo"
    ]):
        return "pravna_pomoc"

    if any(x in qn for x in [
        "zakon", "zakoni", "propis", "pravilnik", "ustav",
        "podzakonski", "konvencija", "ugovor", "regulise", "reguliše"
    ]):
        return "zakoni_i_propisi"

    if any(x in qn for x in [
        "kontakt", "telefon", "email", "mail", "kome",
        "koga", "javim", "obratim", "adresa", "broj", "kontakt informacije", "kontakt info", "kontakt detalji", "kontakt podaci"
    ]):
        return "kontakt_i_nadleznosti"

    if any(x in qn for x in [
        "registar", "lista", "spisak", "registrovano", "registrovana", "registrovani"
    ]):
        return "registri"

    if any(x in qn for x in [
        "pasos", "pasoš", "licna karta", "lična karta",
        "vozacka", "vozačka", "parking", "auto", "gradjevinska",
        "građevinska", "porezna", "porez", "carinska", "dozvola", "dozvolu", "dozvole", "dozvola za gradnju", "dozvola za parkiranje", "dozvola za auto", "dozvola za građevinsku", "dozvola za porez", "dozvola za carinsku", "građevinska dozvola", "porezna dozvola", "carinska dozvola"
    ]):
        return "out_of_scope"

    return None


def predict_intent(q):
    qn = norm(q)

    # prvo očiti kontakt override
    contact_terms = [
        "kontakt", "telefon", "tel", "email", "e-mail", "mail",
        "kome se obratiti", "koga da kontaktiram", "gdje da se javim",
        "broj", "adresa"
    ]

    if any(term in qn for term in contact_terms):
        return "kontakt_i_nadleznosti", 0.99

    model_intent = intent_model.predict([q])[0]
    confidence = intent_model.predict_proba([q])[0].max()

    # ako je model siguran, koristi model
    if confidence >= 0.55:
        return model_intent, confidence

    # ako nije siguran, probaj keyword fallback
    fallback_intent = keyword_intent_fallback(q)

    if fallback_intent:
        return fallback_intent, confidence

    return "needs_clarification", confidence

def required_focus(q):
    qn = norm(q)

    if "pravosudni" in qn:
        return "pravosudni_ispit"
    if "strucni" in qn and "upravni" in qn:
        return "strucni_upravni_ispit"
    if "udruzenj" in qn:
        return "udruzenja"
    if "fondacij" in qn:
        return "fondacije"
    if "besplatn" in qn and "prav" in qn:
        return "besplatna_pravna_pomoc"
    if "alimentacij" in qn:
        return "alimentacije"
    if "otmic" in qn and "djet" in qn:
        return "otmice_djece"

    return ""

def score_boost(q, row):
    qn = norm(q)
    title = norm(row.get("title", ""))
    category = row.get("category", "")
    subsection = row.get("subsection", "")
    priority = row.get("priority", "normal")
    text = norm(row.get("text", ""))
    combined = title + " " + text[:2500]

    boost = 0.0

    focus = required_focus(q)

    if focus and subsection == focus:
        boost += 0.25

    if priority == "high":
        boost += 0.07

    # zakoni / propisi
    if "zakon" in qn or "propis" in qn or "pravilnik" in qn or "regulise" in qn or "regulise" in qn:
        if category == "zakoni_i_propisi":
            boost += 0.25

        if category not in ["zakoni_i_propisi", "registracija"]:
            boost -= 0.20

    # zakon za udruženja/fondacije - preferiraj opšte stranice
    if ("zakon" in qn or "regulise" in qn or "regulise" in qn) and ("udruzenj" in qn or "fondacij" in qn):
        if subsection in ["udruzenja", "fondacije"]:
            boost += 0.18

        if any(x in title for x in ["udruzenja", "fondacije", "kako osnovati"]):
            boost += 0.20

        if any(x in title for x in ["brisanja", "brisanje", "upis promjena", "promjena"]):
            boost -= 0.35

    # obrasci / formulari
    if "obrazac" in qn or "formular" in qn:
        if category == "obrasci":
            boost += 0.20
        if "obrazac" in title or "formular" in title or "obrasci" in title:
            boost += 0.15

    if ("obrazac" in qn or "formular" in qn) and ("udruzenj" in qn or "registracij" in qn):
        if title in ["obrasci", "formulari"]:
            boost += 0.25
        if any(x in title for x in ["vracanje djeteta", "vidjanje djeteta", "vidanje djeteta"]):
            boost -= 0.40

    # takse / troškovi
    if "taksa" in qn or "trosk" in qn or "tros" in qn or "kosta" in qn or "cijena" in qn:
        if "taksa" in title or "trosk" in title or "tros" in title:
            boost += 0.25

    # kontakt
    if "kome se obratiti" in qn or "kontakt" in qn or "email" in qn or "telefon" in qn or "mail" in qn:
        if category == "kontakt_i_nadleznosti" or "kontakt" in title:
            boost += 0.35
        if "kontakt" not in title and "kontakt" not in combined:
            boost -= 0.15

    # promjena podataka
    if ("promjen" in qn or "izmjen" in qn) and "udruzenj" in qn:
        if "upis promjena" in title or "izmjena" in title or "promjena" in title:
            boost += 0.25
        if "brisanja" in title or "brisanje" in title:
            boost -= 0.30

    # brisanje
    if ("brisanje" in qn or "izbrisati" in qn or "ugasiti" in qn) and ("udruzenj" in qn or "fondacij" in qn):
        if "brisanja" in title or "brisanje" in title:
            boost += 0.30

    # title/content token match
    terms = [t for t in re.findall(r"\w+", qn) if len(t) > 4]

    for term in terms:
        if term in title:
            boost += 0.06
        elif term in combined:
            boost += 0.02

    # penalizuj pogrešne teme
    if "udruzenj" in qn and "fondacij" not in qn and subsection == "fondacije":
        boost -= 0.12

    if "fondacij" in qn and "udruzenj" not in qn and subsection == "udruzenja":
        boost -= 0.12

    return max(min(boost, 0.55), -0.40)

DIRECT_ROUTE_RULES = [
    # osnivanje udruženja
    (["osnovat", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["osniva", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["statut", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),
    (["koliko", "ljudi", "udruzenj"], "https://mpr.gov.ba/bs/kako-osnovati-udruzenje"),

    # dokumentacija udruženja
    (["papiri", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["dokumenti", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["gdje", "predaje", "zahtjev", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),

    # obrasci udruženja
    (["obrazac", "promjen", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "udruzenj"], "https://mpr.gov.ba/bs/obrasci"),
    (["formular", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["formulari", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["vodic", "udruzenj"], "https://mpr.gov.ba/bs/obrasci"),

    # takse udruženja
    (["taksa", "promjen", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa84"),
    (["taksa", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa84"),
    (["kosta", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa84"),
    (["kosta", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["taksa", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["racun", "taksa", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),

    # promjene / brisanje
    (["promjen", "zastupnik", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["promjen", "podataka", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["promjen", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["adres", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["ugasiti", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["izbrisati", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["brisanje", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),

    # kontakt udruženja
    (["kontakt", "udruzenj"], "https://mpr.gov.ba/bs/kontakt63"),
    (["kome", "javim", "udruzenj"], "https://mpr.gov.ba/bs/kontakt63"),
    (["ko", "radi", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/kontakt63"),

    # zakoni / registri
    (["pise", "zakon", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["zakon", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["zakon", "fondacij"], "https://mpr.gov.ba/bs/udruzenja"),
    (["regulise", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["registrovano", "udruzenj"], "https://mpr.gov.ba/bs/registar-udruzenja"),
    (["registar", "udruzenj"], "https://mpr.gov.ba/bs/registar-udruzenja"),
    (["registar", "fondacij"], "https://mpr.gov.ba/bs/registar-fondacija"),

    # fondacije
    (["osnovat", "fondacij"], "https://mpr.gov.ba/bs/kako-osnovati-fondaciju"),
    (["osniva", "fondacij"], "https://mpr.gov.ba/bs/kako-osnovati-fondaciju"),
    (["sta", "treba", "fondacij"], "https://mpr.gov.ba/bs/registracija-fondacije"),
    (["dokumenti", "fondacij"], "https://mpr.gov.ba/bs/registracija-fondacije"),
    (["obrazac", "fondacij"], "https://mpr.gov.ba/bs/obrasci1"),
    (["taksa", "fondacij"], "https://mpr.gov.ba/bs/administrativna-taksa9"),
    (["kosta", "fondacij"], "https://mpr.gov.ba/bs/administrativna-taksa9"),
    (["kontakt", "fondacij"], "https://mpr.gov.ba/bs/kontakt2"),

    # strane NVO / predstavništva
    (["strana", "nvo"], "https://mpr.gov.ba/bs/potrebna-dokumentacija5"),
    (["strane", "nevladine"], "https://mpr.gov.ba/bs/potrebna-dokumentacija5"),
    (["predstavnistvo", "nvo", "obrazac"], "https://mpr.gov.ba/bs/obrasci4"),
    (["predstavnistvo", "nvo", "taksa"], "https://mpr.gov.ba/bs/administrativna-taksa21"),

    # pravna lica / crkve
    (["pravna", "lica", "upisati"], "https://mpr.gov.ba/bs/upis-u-registar-promjene-i-brisanje-iz-registra"),
    (["formulari", "pravna", "lica"], "https://mpr.gov.ba/bs/formulari3"),
    (["javnost", "registra"], "https://mpr.gov.ba/bs/javnost-registra"),
    (["crkve", "upis"], "https://mpr.gov.ba/bs/upis-u-registar"),
    (["crkva", "promjena"], "https://mpr.gov.ba/bs/upis-promjena-u-registru"),
    (["prestanak", "crkve"], "https://mpr.gov.ba/bs/prestanak-rada"),
    (["formulari", "crkve"], "https://mpr.gov.ba/bs/formulari5"),
    (["taksa", "crkve"], "https://mpr.gov.ba/bs/administrativne-takse87"),
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

def search(q, top_k=5, candidate_k=100):
    intent, confidence = predict_intent(q)
    
    forced_url = direct_route_url(q)

    if forced_url:
        forced = chunks_df[
        chunks_df["source_url"].astype(str).str.rstrip("/") == forced_url.rstrip("/")
    ].copy()

        if not forced.empty:
            forced_results = []

            for _, row in forced.head(5).iterrows():
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

    if intent == "needs_clarification":
        return intent, confidence, []

    df = chunks_df.copy()

    q_vec = vectorizer.transform([q])
    X = vectorizer.transform(df["search_text"].fillna("").tolist())

    nn = NearestNeighbors(
        n_neighbors=min(candidate_k, len(df)),
        metric="cosine"
    )

    nn.fit(X)

    distances, indices = nn.kneighbors(q_vec)

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx].to_dict()

        base = 1 - dist
        boost = score_boost(q, row)
        final_score = base + boost

        row["base_score"] = base
        row["boost"] = boost
        row["score"] = final_score
        row["predicted_intent"] = intent
        row["intent_confidence"] = confidence

        results.append(row)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return intent, confidence, results[:top_k]

def reciprocal_rank(results, expected_url):
    expected_url = normalize_url(expected_url)

    for i, r in enumerate(results, start=1):
        if normalize_url(r["source_url"]) == expected_url:
            return 1 / i

    return 0.0

# ============================
# EVALUATION
# ============================

rows = []

for _, item in eval_df.iterrows():
    question = item["question"]
    expected_url = normalize_url(item["expected_url"])

    intent, confidence, results = search(question, top_k=5)

    retrieved_urls = [normalize_url(r["source_url"]) for r in results]

    top1_url = retrieved_urls[0] if len(retrieved_urls) >= 1 else ""
    top3_urls = retrieved_urls[:3]
    top5_urls = retrieved_urls[:5]

    hit_top1 = expected_url == top1_url
    hit_top3 = expected_url in top3_urls
    hit_top5 = expected_url in top5_urls

    rr = reciprocal_rank(results, expected_url)

    rows.append({
        "question": question,
        "expected_url": expected_url,
        "predicted_intent": intent,
        "intent_confidence": confidence,
        "top1_url": top1_url,
        "top1_title": results[0]["title"] if results else "",
        "top1_category": results[0]["category"] if results else "",
        "top1_score": results[0]["score"] if results else 0,
        "hit_top1": hit_top1,
        "hit_top3": hit_top3,
        "hit_top5": hit_top5,
        "reciprocal_rank": rr,
        "top5_urls": " | ".join(retrieved_urls)
    })

results_df = pd.DataFrame(rows)

top1_accuracy = results_df["hit_top1"].mean()
top3_recall = results_df["hit_top3"].mean()
top5_recall = results_df["hit_top5"].mean()
mrr = results_df["reciprocal_rank"].mean()

summary = {
    "samples": len(results_df),
    "top1_accuracy": round(top1_accuracy, 4),
    "top3_recall": round(top3_recall, 4),
    "top5_recall": round(top5_recall, 4),
    "mrr": round(mrr, 4)
}

# category-level analysis
category_summary = (
    results_df
    .groupby("predicted_intent")
    .agg(
        samples=("question", "count"),
        top1_accuracy=("hit_top1", "mean"),
        top3_recall=("hit_top3", "mean"),
        top5_recall=("hit_top5", "mean"),
        mrr=("reciprocal_rank", "mean")
    )
    .reset_index()
)

# save
results_df.to_csv(OUTPUT_DIR / "rag_eval_detailed_results.csv", index=False)
category_summary.to_csv(OUTPUT_DIR / "rag_eval_by_intent.csv", index=False)

with open(OUTPUT_DIR / "rag_eval_summary.json", "w", encoding="utf-8") as f:
    import json
    json.dump(summary, f, indent=2, ensure_ascii=False)

# print
print("\n======================")
print("RAG EVALUATION SUMMARY")
print("======================")
print(f"Samples: {summary['samples']}")
print(f"Top-1 Accuracy: {summary['top1_accuracy']}")
print(f"Top-3 Recall:   {summary['top3_recall']}")
print(f"Top-5 Recall:   {summary['top5_recall']}")
print(f"MRR:            {summary['mrr']}")

print("\n======================")
print("BY INTENT")
print("======================")
print(category_summary)

print("\n======================")
print("FAILED TOP-1 EXAMPLES")
print("======================")

failed = results_df[results_df["hit_top1"] == False].head(30)

for _, row in failed.iterrows():
    print("=" * 100)
    print("Q:", row["question"])
    print("EXPECTED:", row["expected_url"])
    print("TOP1:", row["top1_url"])
    print("TITLE:", row["top1_title"])
    print("INTENT:", row["predicted_intent"])
    print("SCORE:", round(row["top1_score"], 4))

print("\nSaved detailed results to:", OUTPUT_DIR)
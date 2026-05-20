import re
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

intent_model = joblib.load("models/intent_classifier.joblib")
vectorizer = joblib.load("models_v4/rag_vectorizer_v4.joblib")
chunks_df = pd.read_pickle("models_v4/rag_chunks_v4.pkl")

def norm(text):
    text = str(text).lower()
    return (
        text.replace("č", "c")
        .replace("ć", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("đ", "dj")
    )

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
        "obrazac 3", "vodic", "vodič"
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
        "otmica djece", "medjunarodna pravna pomoc"
    ]):
        return "pravna_pomoc"

    if any(x in qn for x in [
        "zakon", "zakoni", "propis", "pravilnik", "ustav",
        "podzakonski", "konvencija", "ugovor", "regulise", "reguliše"
    ]):
        return "zakoni_i_propisi"

    if any(x in qn for x in [
        "kontakt", "telefon", "email", "mail", "kome",
        "koga", "javim", "obratim", "adresa", "broj"
    ]):
        return "kontakt_i_nadleznosti"

    if any(x in qn for x in [
        "registar", "lista", "spisak", "registrovano"
    ]):
        return "registri"

    if any(x in qn for x in [
        "pasos", "pasoš", "licna karta", "lična karta",
        "vozacka", "vozačka", "parking", "auto", "gradjevinska",
        "građevinska", "porezna"
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

    if "zakon" in qn or "propis" in qn or "pravilnik" in qn:
        if category == "zakoni_i_propisi":
            boost += 0.25
        if category not in ["zakoni_i_propisi", "registracija"]:
            boost -= 0.20

    if "obrazac" in qn or "formular" in qn:
        if category == "obrasci":
            boost += 0.20
        if "obrazac" in title or "formular" in title:
            boost += 0.15

    if "taksa" in qn or "trosk" in qn or "troš" in qn:
        if "taksa" in title or "trosk" in title or "troš" in title:
            boost += 0.25

    if "kontakt" in qn or "kome se obratiti" in qn or "email" in qn or "telefon" in qn:
        if category == "kontakt_i_nadleznosti" or "kontakt" in title:
            boost += 0.25

    # Ako korisnik pita "koji zakon", preferiraj opšte stranice i zakonske stranice,
    # a ne specifične procedure kao brisanje/promjene.
    if ("zakon" in qn or "regulise" in qn or "reguliše" in qn) and ("udruzenj" in qn or "fondacij" in qn):
        if "udruzenja" in subsection or "fondacije" in subsection:
            boost += 0.18

        if any(x in title for x in ["udruzenja", "fondacije", "kako osnovati"]):
            boost += 0.20

        if any(x in title for x in ["brisanja", "brisanje", "upis promjena", "promjena"]):
            boost -= 0.35
       
    if ("promjen" in qn or "izmjen" in qn) and "udruzenj" in qn:
        if "upis promjena" in title or "izmjena" in title or "promjena" in title:
            boost += 0.25

        if "brisanja" in title or "brisanje" in title:
            boost -= 0.30
        
    if ("kome se obratiti" in qn or "kontakt" in qn or "email" in qn or "telefon" in qn):
        if "kontakt" in title:
            boost += 0.35
        if "kontakt" not in title and "kontakt" not in combined:
            boost -= 0.15
            
    if ("obrazac" in qn or "formular" in qn) and ("udruzenj" in qn or "registracij" in qn):
        if title in ["obrasci", "formulari"]:
            boost += 0.25
        if any(x in title for x in ["vraćanje djeteta", "vidjanje djeteta", "viđanje djeteta"]):
            boost -= 0.40         
    
    terms = [t for t in re.findall(r"\w+", qn) if len(t) > 4]
    for term in terms:
        if term in title:
            boost += 0.06
        elif term in combined:
            boost += 0.02

    # penalizuj pogrešne specifične teme
    if "udruzenj" in qn and "fondacij" not in qn and subsection == "fondacije":
        boost -= 0.12

    if "fondacij" in qn and "udruzenj" not in qn and subsection == "udruzenja":
        boost -= 0.12

    return max(min(boost, 0.55), -0.4)

DIRECT_ROUTE_RULES = [
    (["papiri", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["dokumenti", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),
    (["sta", "treba", "udruzenj"], "https://mpr.gov.ba/bs/potrebna-dokumentacija"),

    (["obrazac", "udruzenj"], "https://mpr.gov.ba/bs/obrasci"),
    (["formular", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "promjen", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),
    (["obrazac", "brisanje", "udruzenj"], "https://mpr.gov.ba/bs/formulari97"),

    (["taksa", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["kosta", "registracij", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),
    (["racun", "taksa", "udruzenj"], "https://mpr.gov.ba/bs/administrativna-taksa55"),

    (["taksa", "fondacij"], "https://mpr.gov.ba/bs/administrativna-taksa9"),
    (["kosta", "fondacij"], "https://mpr.gov.ba/bs/administrativna-taksa9"),

    (["promjen", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["adres", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),
    (["zastupnik", "udruzenj"], "https://mpr.gov.ba/bs/upis-promjena-u-registru-udruzenja-ili-fondacija"),

    (["ugasiti", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["izbrisati", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),
    (["brisanje", "udruzenj"], "https://mpr.gov.ba/bs/brisanja-iz-registra-po-zahtjevu-udruzenja-i-fondacije"),

    (["zakon", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),
    (["zakon", "fondacij"], "https://mpr.gov.ba/bs/udruzenja"),
    (["regulise", "udruzenj"], "https://mpr.gov.ba/bs/udruzenja"),

    (["registrovano", "udruzenj"], "https://mpr.gov.ba/bs/registar-udruzenja"),
    (["registar", "udruzenj"], "https://mpr.gov.ba/bs/registar-udruzenja"),
    (["registar", "fondacij"], "https://mpr.gov.ba/bs/registar-fondacija"),
]

def direct_route_url(q):
    qn = norm(q)

    for keywords, url in DIRECT_ROUTE_RULES:
        if all(k in qn for k in keywords):
            return url

    return None

def search(q, top_k=5, candidate_k=100):
    intent, confidence = predict_intent(q)

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

questions = [
    "Ko može polagati pravosudni ispit?",
    "Gdje mogu naći obrazac za registraciju udruženja?",
    "Koji zakon reguliše udruženja i fondacije?",
    "Kako mogu dobiti besplatnu pravnu pomoć?",
    "Kako prijaviti promjenu podataka udruženja?",
    "Kolika je administrativna taksa za registraciju fondacije?",
    "Kome se obratiti za stručni upravni ispit?",
    "Koji dokumenti trebaju za registraciju udruženja?",
    "Kako osnovati fondaciju?",
    "Kako da izvadim pasoš?"
]

for q in questions:
    intent, conf, results = search(q)

    print("\n" + "=" * 90)
    print("QUESTION:", q)
    print("INTENT:", intent, "| confidence:", round(conf, 3))

    if intent == "out_of_scope":
        print("Odgovor: Ovo pitanje vjerovatno nije u nadležnosti Ministarstva pravde BiH.")
        continue

    if intent == "needs_clarification":
        print("Odgovor: Nisam sigurna u koju oblast pitanje spada. Potrebno je dodatno pojasniti upit.")
        continue

    print("\nTOP RESULTS:")
    for r in results:
        print(
            "- score:", round(r["score"], 3),
            "| base:", round(r["base_score"], 3),
            "| boost:", round(r["boost"], 3),
            "| cat:", r["category"],
            "| sub:", r["subsection"],
            "| title:", r["title"],
            "| url:", r["source_url"]
        )

    if results:
        best = results[0]
        print("\nDRAFT ANSWER:")
        print(best["title"])
        print(best["source_url"])
        print(best["text"][:1300])
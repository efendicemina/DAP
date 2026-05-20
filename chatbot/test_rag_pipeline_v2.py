import re
import joblib
import pandas as pd

intent_model = joblib.load("models/intent_classifier.joblib")
vectorizer = joblib.load("models_v3/rag_vectorizer_v3.joblib")
chunks_df = pd.read_pickle("models_v3/rag_chunks_v3.pkl")

def norm(text):
    text = str(text).lower()
    return (
        text.replace("č", "c")
            .replace("ć", "c")
            .replace("š", "s")
            .replace("ž", "z")
            .replace("đ", "dj")
    )

def predict_intent(q):
    return intent_model.predict([q])[0], intent_model.predict_proba([q])[0].max()

def is_menu_noise(text):
    t = norm(text)
    return t.startswith("hrvatski bosanski") or "pocetna |" in t[:250]

def query_terms(q):
    q = norm(q)
    terms = re.findall(r"\w+", q)
    return [t for t in terms if len(t) >= 5]

def required_terms_for_question(q):
    qn = norm(q)

    if "sudsk" in qn and ("tumac" in qn or "tumac" in qn):
        return ["sudsk", "tumac"]

    if "notar" in qn:
        return ["notar"]

    if "udruzenj" in qn:
        return ["udruzenj"]

    if "fondacij" in qn:
        return ["fondacij"]

    if "pravosudni" in qn:
        return ["pravosudni"]

    if "strucni" in qn and "upravni" in qn:
        return ["strucni", "upravni"]

    if "besplatn" in qn and "prav" in qn and "pomoc" in qn:
        return ["besplatn", "prav", "pomoc"]

    return []

def passes_required_terms(q, row):
    required = required_terms_for_question(q)
    if not required:
        return True

    combined = norm(row["title"] + " " + row["text"])
    return all(term in combined for term in required)

def smart_boost(q, row):
    qn = norm(q)
    title = norm(row["title"])
    text = norm(row["text"])
    combined = title + " " + text[:2500]

    boost = 0.0

    # title exact topic boost
    for term in query_terms(q):
        if term in title:
            boost += 0.08

    # content match boost, capped
    for term in query_terms(q):
        if term in combined:
            boost += 0.025

    # phrase-specific boost
    phrase_boosts = [
        ("registraciju udruzenja", ["registracija", "udruzenja"]),
        ("registracija udruzenja", ["registracija", "udruzenja"]),
        ("obrazac", ["obrazac"]),
        ("pravosudni ispit", ["pravosudni", "ispit"]),
        ("strucni upravni ispit", ["strucni", "upravni", "ispit"]),
        ("besplatna pravna pomoc", ["besplatna", "pravna", "pomoc"]),
        ("sudskih tumaca", ["sudskih", "tumaca"]),
        ("sudski tumac", ["sudski", "tumac"]),
        ("notara", ["notar"]),
        ("udruzenja i fondacije", ["udruzenja", "fondacije"]),
    ]

    for _, words in phrase_boosts:
        if all(w in qn for w in words) and all(w in combined for w in words):
            boost += 0.12

    # kazna za meni/index stranice
    if is_menu_noise(row["text"]):
        boost -= 0.25

    # kazna ako je title očito druga tema
    bad_pairs = [
        ("registraciju udruzenja", ["djeteta", "otmice", "alimentacije"]),
        ("sudskih tumaca", ["djeteta", "udruzenja", "fondacije"]),
        ("notara", ["djeteta", "alimentacije", "ispit"]),
        ("udruzenja i fondacije", ["pomilovanju", "krivicnih sankcija", "slobodi pristupa"]),
    ]

    for query_phrase, bad_words in bad_pairs:
        if query_phrase in qn:
            if any(bad in title for bad in bad_words):
                boost -= 0.35

    return max(min(boost, 0.35), -0.40)

def search(q, top_k=5, candidate_k=80):
    intent, conf = predict_intent(q)

    if intent == "out_of_scope":
        return intent, conf, []

    df = chunks_df.copy()

    # Ne filtriramo prestrogo po category jer kategorije nisu savršene.
    q_vec = vectorizer.transform([q])
    X = vectorizer.transform(df["search_text"].fillna("").tolist())

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(candidate_k, len(df)), metric="cosine")
    nn.fit(X)

    distances, indices = nn.kneighbors(q_vec)

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx].to_dict()

        if not passes_required_terms(q, row):
            continue

        base = 1 - dist
        boost = smart_boost(q, row)
        score = base + boost

        if is_menu_noise(row["text"]) and base < 0.12:
            continue

        row["base_score"] = base
        row["boost"] = boost
        row["score"] = score
        results.append(row)

    results = sorted(results, key=lambda r: r["score"], reverse=True)
    return intent, conf, results[:top_k]

questions = [
    "Ko može polagati pravosudni ispit?",
    "Gdje mogu naći obrazac za registraciju udruženja?",
    "Koji zakon reguliše udruženja i fondacije?",
    "Kako mogu dobiti besplatnu pravnu pomoć?",
    "Treba mi lista sudskih tumača",
    "Gdje mogu pronaći listu notara?",
    "Kako prijaviti promjenu podataka udruženja?",
    "Kolika je administrativna taksa za registraciju fondacije?",
    "Kome se obratiti za stručni upravni ispit?",
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

    print("\nTOP RESULTS:")
    for r in results:
        print(
            "- score:", round(r["score"], 3),
            "| base:", round(r["base_score"], 3),
            "| boost:", round(r["boost"], 3),
            "| cat:", r["category"],
            "| title:", r["title"],
            "| url:", r["source_url"]
        )

    if results:
        best = results[0]
        print("\nDRAFT ANSWER:")
        print(best["title"])
        print(best["source_url"])
        print(best["text"][:1200])
    else:
        print("\nNema dovoljno relevantnog rezultata.")
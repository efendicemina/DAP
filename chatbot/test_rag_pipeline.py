import re
import joblib
import pandas as pd

intent_model = joblib.load("models/intent_classifier.joblib")
vectorizer = joblib.load("models/rag_vectorizer.joblib")
nn = joblib.load("models/rag_nn.joblib")
chunks_df = pd.read_pickle("models/rag_chunks.pkl")

INTENT_TO_ALLOWED_CATEGORIES = {
    "registracija": ["registracija", "obrasci", "zakoni_i_propisi", "ostalo"],
    "obrasci": ["obrasci", "registracija", "ispiti", "pravna_pomoc", "zakoni_i_propisi", "ostalo"],
    "zakoni_i_propisi": ["zakoni_i_propisi", "registracija", "ispiti", "pravna_pomoc", "notari", "sudski_tumaci", "ostalo"],
    "pravna_pomoc": ["pravna_pomoc", "zakoni_i_propisi", "ostalo"],
    "ispiti": ["ispiti", "obrasci", "zakoni_i_propisi", "kontakt_i_nadleznosti", "ostalo"],
    "registri": ["registri", "sudski_tumaci", "notari", "registracija", "ostalo"],
    "kontakt_i_nadleznosti": ["kontakt_i_nadleznosti", "ispiti", "registracija", "pravna_pomoc", "ostalo"],
}

KEYWORD_BOOSTS = {
    "sudski_tumaci": ["sudski tumač", "sudski tumac", "tumača", "tumaca", "stalni sudski"],
    "notari": ["notar", "notara", "notari", "notarsk"],
    "registracija": ["registracija", "registrovati", "udruženje", "udruzenje", "fondacija", "nvo"],
    "obrasci": ["obrazac", "formular", "zahtjev", "prijava", "word", "pdf"],
    "ispiti": ["ispit", "pravosudni", "stručni upravni", "strucni upravni", "termin"],
    "pravna_pomoc": ["pravna pomoć", "pravna pomoc", "besplatna", "alimentacija", "otmica djece"],
    "zakoni_i_propisi": ["zakon", "pravilnik", "propis", "službeni glasnik", "sluzbeni glasnik"],
    "kontakt_i_nadleznosti": ["kontakt", "email", "telefon", "nadležnost", "nadleznost", "sektor"],
}

def normalize(text):
    text = str(text).lower()
    text = text.replace("č", "c").replace("ć", "c")
    text = text.replace("š", "s").replace("ž", "z").replace("đ", "dj")
    return text

def predict_intent(question):
    intent = intent_model.predict([question])[0]
    confidence = intent_model.predict_proba([question])[0].max()
    return intent, confidence

def keyword_score(question, row):
    q = normalize(question)
    combined = normalize(
        str(row.get("title", "")) + " " +
        str(row.get("category", "")) + " " +
        str(row.get("text", ""))
    )

    score = 0.0

    for group, keywords in KEYWORD_BOOSTS.items():
        for kw in keywords:
            nkw = normalize(kw)

            if nkw in q and nkw in combined:
                score += 0.20

    # dodatni boost ako se bitne riječi iz pitanja nalaze u title-u
    title = normalize(row.get("title", ""))

    for token in re.findall(r"\w+", q):
        if len(token) > 4 and token in title:
            score += 0.08

    return score

def search_chunks(question, intent=None, top_k=5, candidate_k=30):
    df = chunks_df.copy()

    allowed = INTENT_TO_ALLOWED_CATEGORIES.get(intent, [])

    if allowed:
        filtered = df[df["category"].isin(allowed)].copy()

        # Ako filter previše suzi bazu, ne koristi ga
        if len(filtered) >= 10:
            df = filtered

    search_texts = df["search_text"].fillna("").tolist()

    local_X = vectorizer.transform(search_texts)
    q_vec = vectorizer.transform([question])

    local_nn = joblib.load("models/rag_nn.joblib")
    local_nn.fit(local_X)

    distances, indices = local_nn.kneighbors(
        q_vec,
        n_neighbors=min(candidate_k, len(df))
    )

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx].to_dict()

        base_score = 1 - dist
        boost = keyword_score(question, row)
        final_score = base_score + boost

        row["base_score"] = base_score
        row["keyword_boost"] = boost
        row["score"] = final_score

        results.append(row)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]

def answer_without_llm(question, results):
    if not results or results[0]["score"] < 0.05:
        return "Nisam pronašla dovoljno relevantnih informacija u dostupnoj bazi."

    best = results[0]

    return f"""
Pitanje: {question}

Najrelevantniji izvor:
{best['title']}
{best['source_url']}

Relevantni tekst:
{best['text'][:1200]}
""".strip()

questions = [
    "Ko može polagati pravosudni ispit?",
    "Gdje mogu naći obrazac za registraciju udruženja?",
    "Koji zakon reguliše udruženja i fondacije?",
    "Kako mogu dobiti besplatnu pravnu pomoć?",
    "Treba mi lista sudskih tumača",
    "Kako da izvadim pasoš?"
]

for q in questions:
    intent, confidence = predict_intent(q)

    print("\n" + "=" * 80)
    print("QUESTION:", q)
    print("INTENT:", intent, "| confidence:", round(confidence, 3))

    if intent == "out_of_scope":
        print("Odgovor: Ovo pitanje vjerovatno nije u nadležnosti Ministarstva pravde BiH.")
        continue

    results = search_chunks(q, intent=intent, top_k=3)

    print("\nTOP RESULTS:")
    for r in results:
        print(
            "-",
            "score:", round(r["score"], 3),
            "| base:", round(r["base_score"], 3),
            "| boost:", round(r["keyword_boost"], 3),
            "|", r["category"],
            "|", r["title"],
            "|", r["source_url"]
        )

    print("\nDRAFT ANSWER:")
    print(answer_without_llm(q, results))
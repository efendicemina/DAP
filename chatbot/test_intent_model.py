import joblib

model = joblib.load("models/intent_classifier.joblib")

questions = [
    "Kako registrovati udruženje?",
    "Gdje mogu naći obrazac za pravosudni ispit?",
    "Koji zakon reguliše fondacije?",
    "Kada je sljedeći stručni upravni ispit?",
    "Treba mi lista sudskih tumača",
    "Kako mogu dobiti besplatnu pravnu pomoć?",
    "Kako da izvadim pasoš?"
]

for q in questions:
    pred = model.predict([q])[0]
    probs = model.predict_proba([q])[0]
    confidence = probs.max()

    print(f"\nPitanje: {q}")
    print(f"Intent: {pred}")
    print(f"Confidence: {confidence:.2f}")
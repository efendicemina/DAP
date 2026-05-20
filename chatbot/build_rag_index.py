import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path("mpr_dataset_final")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

chunks = []

with open(DATA_DIR / "final_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

df = pd.DataFrame(chunks)

df["search_text"] = (
    df["title"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["text"].fillna("")
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["search_text"])

nn = NearestNeighbors(
    n_neighbors=8,
    metric="cosine"
)

nn.fit(X)

joblib.dump(vectorizer, MODEL_DIR / "rag_vectorizer.joblib")
joblib.dump(nn, MODEL_DIR / "rag_nn.joblib")
df.to_pickle(MODEL_DIR / "rag_chunks.pkl")

print("DONE")
print("Chunks indexed:", len(df))
print("Saved RAG index to models/")
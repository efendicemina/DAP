import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path("mpr_dataset_v4")
MODEL_DIR = Path("models_v4")
MODEL_DIR.mkdir(exist_ok=True)

chunks = []

with open(DATA_DIR / "chunks_v4.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

df = pd.DataFrame(chunks)

df["search_text"] = (
    df["title"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["subsection"].fillna("") + " " +
    df["text"].fillna("")
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)

X = vectorizer.fit_transform(df["search_text"])

nn = NearestNeighbors(
    n_neighbors=10,
    metric="cosine"
)

nn.fit(X)

joblib.dump(vectorizer, MODEL_DIR / "rag_vectorizer_v4.joblib")
joblib.dump(nn, MODEL_DIR / "rag_nn_v4.joblib")
df.to_pickle(MODEL_DIR / "rag_chunks_v4.pkl")

print("DONE")
print("Chunks indexed:", len(df))
print(df["category"].value_counts())
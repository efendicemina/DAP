import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

CHUNKS_PATH = "models_v4/rag_chunks_v4.pkl"
MODEL_DIR = Path("models_v4_embeddings")
MODEL_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

df = pd.read_pickle(CHUNKS_PATH)

df["search_text"] = (
    df["title"].fillna("") + "\n" +
    df["category"].fillna("") + "\n" +
    df["subsection"].fillna("") + "\n" +
    df["text"].fillna("")
)

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

embeddings = model.encode(
    df["search_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

np.save(MODEL_DIR / "chunk_embeddings.npy", embeddings)
df.to_pickle(MODEL_DIR / "rag_chunks_v4_embeddings.pkl")

if FAISS_AVAILABLE:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(MODEL_DIR / "faiss_index.bin"))
    print("FAISS index saved.")
else:
    print("FAISS not available. Saved embeddings only.")

joblib.dump(
    {"embedding_model_name": EMBEDDING_MODEL_NAME},
    MODEL_DIR / "embedding_config.joblib"
)

print("DONE")
print("Chunks:", len(df))
print("Embedding shape:", embeddings.shape)
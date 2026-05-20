import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

DATA_PATH = "mpr_dataset_v5/chunks_v5.csv"
MODEL_DIR = Path("models_v5")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

chunks = pd.read_csv(DATA_PATH)

chunks["search_text"] = (
    chunks["title"].fillna("") + "\n" +
    chunks["category"].fillna("") + "\n" +
    chunks["subsection"].fillna("") + "\n" +
    chunks["page_type"].fillna("") + "\n" +
    chunks["semantic_topic"].fillna("") + "\n" +
    chunks["text"].fillna("")
)

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    chunks["search_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, str(MODEL_DIR / "faiss_index_v5.bin"))
np.save(MODEL_DIR / "chunk_embeddings_v5.npy", embeddings)
chunks.to_pickle(MODEL_DIR / "rag_chunks_v5.pkl")

joblib.dump(
    {"model_name": MODEL_NAME},
    MODEL_DIR / "embedding_config_v5.joblib"
)

print("DONE")
print("Chunks:", len(chunks))
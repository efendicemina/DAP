"""
Task 1 - Deep contextual intent classifier.

Ovaj fajl dodaje drugu formu prikaza teksta za Task 1:
1) postojeci baseline: TF-IDF + Logistic Regression,
2) nova duboko kontekstualna forma: SentenceTransformer embeddings + Logistic Regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "Nedostaje dependency 'sentence-transformers'. Instaliraj ga prije pokretanja: "
        "pip install sentence-transformers"
    ) from exc


DATA_DIR = Path("mpr_dataset_final")
MODEL_DIR = Path("models_task1")
REPORT_DIR = Path("task1_eval_reports")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "intent_train.csv"),
        pd.read_csv(DATA_DIR / "intent_val.csv"),
        pd.read_csv(DATA_DIR / "intent_test.csv"),
    )


def encode_texts(model: SentenceTransformer, texts: pd.Series) -> np.ndarray:
    embeddings = model.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def evaluate_classifier(clf: LogisticRegression, X: np.ndarray, y_true: pd.Series, split_name: str) -> Dict[str, float]:
    preds = clf.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
    }
    report_df = pd.DataFrame(classification_report(y_true, preds, output_dict=True)).transpose()
    report_df.to_csv(REPORT_DIR / f"contextual_intent_{split_name}_classification_report.csv")
    labels = sorted(y_true.unique())
    cm_df = pd.DataFrame(confusion_matrix(y_true, preds, labels=labels), index=labels, columns=labels)
    cm_df.to_csv(REPORT_DIR / f"contextual_intent_{split_name}_confusion_matrix.csv")
    return metrics


def main() -> None:
    train_df, val_df, test_df = load_splits()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    X_train = encode_texts(embedding_model, train_df["text"])
    X_val = encode_texts(embedding_model, val_df["text"])
    X_test = encode_texts(embedding_model, test_df["text"])

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["label"])

    metrics = {
        "model_type": "SentenceTransformer embeddings + LogisticRegression",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "validation": evaluate_classifier(clf, X_val, val_df["label"], "validation"),
        "test": evaluate_classifier(clf, X_test, test_df["label"], "test"),
    }

    with open(REPORT_DIR / "contextual_intent_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    joblib.dump(
        {
            "classifier": clf,
            "embedding_model_name": EMBEDDING_MODEL_NAME,
            "labels": sorted(train_df["label"].unique().tolist()),
            "model_type": "SentenceTransformer embeddings + LogisticRegression",
        },
        MODEL_DIR / "intent_contextual_classifier.joblib",
    )

    print("\nDONE - Task 1 contextual intent model")
    print("Model saved to:", MODEL_DIR / "intent_contextual_classifier.joblib")
    print("Reports saved to:", REPORT_DIR)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

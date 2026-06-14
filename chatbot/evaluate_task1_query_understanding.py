"""
Full Task 1 evaluation.

Ova skripta pravi finalni evaluacioni izvjestaj za prvi NLP task:
- poredi dvije forme prikaza teksta:
  1) TF-IDF + Logistic Regression,
  2) SentenceTransformer embeddings + Logistic Regression,
- racuna Accuracy, Macro F1 i Weighted F1,
- cuva classification report i confusion matrix za oba pristupa,
- dodatno evaluira ekstrakciju kljucnih informacija pomocu weak-label provjere nad subintent kolonom.

Pokretanje:
    cd chatbot
    python evaluate_task1_query_understanding.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from query_understanding_v1 import evaluate_procedure_extraction, extract_key_information

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "Nedostaje dependency 'sentence-transformers'. Instaliraj ga prije pokretanja: "
        "pip install sentence-transformers"
    ) from exc


DATA_DIR = Path("mpr_dataset_final")
REPORT_DIR = Path("task1_eval_reports")
REPORT_DIR.mkdir(exist_ok=True, parents=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "intent_train.csv"),
        pd.read_csv(DATA_DIR / "intent_val.csv"),
        pd.read_csv(DATA_DIR / "intent_test.csv"),
    )


def metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def save_detailed_reports(model_name: str, split_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    safe_model_name = model_name.lower().replace(" ", "_").replace("+", "plus")
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    report_df.to_csv(REPORT_DIR / f"{safe_model_name}_{split_name}_classification_report.csv")
    labels = sorted(y_true.unique())
    cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    cm_df.to_csv(REPORT_DIR / f"{safe_model_name}_{split_name}_confusion_matrix.csv")


def evaluate_tfidf(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    model.fit(train_df["text"], train_df["label"])
    val_pred = model.predict(val_df["text"])
    test_pred = model.predict(test_df["text"])
    save_detailed_reports("TFIDF", "validation", val_df["label"], val_pred)
    save_detailed_reports("TFIDF", "test", test_df["label"], test_pred)
    return {"validation": metric_dict(val_df["label"], val_pred), "test": metric_dict(test_df["label"], test_pred)}


def encode_texts(model: SentenceTransformer, texts: pd.Series) -> np.ndarray:
    embeddings = model.encode(texts.fillna("").astype(str).tolist(), batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype="float32")


def evaluate_contextual(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    X_train = encode_texts(embedding_model, train_df["text"])
    X_val = encode_texts(embedding_model, val_df["text"])
    X_test = encode_texts(embedding_model, test_df["text"])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["label"])
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    save_detailed_reports("Contextual", "validation", val_df["label"], val_pred)
    save_detailed_reports("Contextual", "test", test_df["label"], test_pred)
    return {"validation": metric_dict(val_df["label"], val_pred), "test": metric_dict(test_df["label"], test_pred)}


def write_markdown_report(tfidf_metrics, contextual_metrics, slot_metrics, test_df: pd.DataFrame) -> None:
    examples = [extract_key_information(text).to_dict() for text in test_df["text"].head(8).tolist()]

    def row(model_name: str, split: str, metrics: Dict[str, float]) -> str:
        return f"| {model_name} | {split} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |"

    content = f"""# Task 1 evaluacija: analiza korisničkog upita

## Cilj taska

Cilj prvog NLP taska je razumjeti korisnički upit prije retrieval i chatbot faze. Task se sastoji od dvije komponente:

1. **Intent classification** - određivanje glavne namjere korisnika.
2. **Ekstrakcija ključnih informacija** - izdvajanje tipa tražene informacije, dokumenta, procedure, pravne oblasti, institucije i ključnih riječi.

## Dvije forme prikaza teksta

| Pristup | Forma prikaza teksta | Opis |
|---|---|---|
| Baseline | TF-IDF n-gram reprezentacija | Klasična leksička reprezentacija korisničkog pitanja. |
| Kontekstualni model | SentenceTransformer embeddings | Duboko kontekstualna reprezentacija koja bolje hvata parafraze i semantički slične formulacije. |

## Rezultati intent klasifikacije

| Model | Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
{row('TF-IDF + Logistic Regression', 'validation', tfidf_metrics['validation'])}
{row('TF-IDF + Logistic Regression', 'test', tfidf_metrics['test'])}
{row('SentenceTransformer + Logistic Regression', 'validation', contextual_metrics['validation'])}
{row('SentenceTransformer + Logistic Regression', 'test', contextual_metrics['test'])}

## Rezultati ekstrakcije ključnih informacija

Ekstrakcija procedura je evaluirana pomoću weak-label pristupa, gdje se očekivana procedura izvodi iz `subintent` kolone sintetičkog skupa podataka.

| Metrika | Vrijednost |
|---|---:|
| Evaluirani primjeri | {slot_metrics['evaluated_examples']:.0f} |
| Tačni primjeri | {slot_metrics['correct_examples']:.0f} |
| Preskočeni bez weak labela | {slot_metrics['skipped_without_weak_label']:.0f} |
| Procedure extraction accuracy | {slot_metrics['procedure_extraction_accuracy']:.4f} |

## Primjeri strukturiranog izlaza

```json
{json.dumps(examples, indent=2, ensure_ascii=False)}
```

## Zaključak

Task 1 sada formalno ispunjava zahtjev projekta jer ima dvije različite forme prikaza teksta, pri čemu je druga duboko kontekstualna. Pored same klasifikacije namjere, dodat je i interpretabilan sloj za ekstrakciju ključnih informacija iz korisničkog pitanja.
"""
    (REPORT_DIR / "task1_query_understanding_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    train_df, val_df, test_df = load_splits()
    tfidf_metrics = evaluate_tfidf(train_df, val_df, test_df)
    contextual_metrics = evaluate_contextual(train_df, val_df, test_df)
    slot_metrics = evaluate_procedure_extraction(test_df.to_dict(orient="records"))
    all_metrics = {
        "task": "Task 1 - query understanding",
        "text_representations": {"baseline": "TF-IDF n-grams", "deep_contextual": EMBEDDING_MODEL_NAME},
        "intent_classification": {
            "tfidf_logistic_regression": tfidf_metrics,
            "sentence_transformer_logistic_regression": contextual_metrics,
        },
        "key_information_extraction": slot_metrics,
    }
    (REPORT_DIR / "task1_metrics.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(tfidf_metrics, contextual_metrics, slot_metrics, test_df)
    print("\nDONE - Task 1 full evaluation")
    print("Reports saved to:", REPORT_DIR)
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

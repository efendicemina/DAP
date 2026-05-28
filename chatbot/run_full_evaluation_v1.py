"""Run evaluation for the current chatbot pipeline (v1).

Outputs:
- RAG retrieval metrics on rag_eval_questions.csv (Hit@k, MRR) + plots
- Intent classifier metrics on mpr_dataset_final/intent_test.csv + confusion matrix plot

Designed to be run from repo root or from chatbot/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CHATBOT_DIR = REPO_ROOT / "chatbot"

RAG_EVAL_CSV = CHATBOT_DIR / "rag_eval_questions.csv"
INTENT_TEST_CSV = CHATBOT_DIR / "mpr_dataset_final" / "intent_test.csv"

OUTPUT_DIR = CHATBOT_DIR / "eval_reports_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/")


@dataclass
class RagSummary:
    samples: int
    top1_accuracy: float
    top3_recall: float
    top5_recall: float
    mrr: float


def reciprocal_rank(retrieved_urls: list[str], expected_url: str) -> float:
    exp = normalize_url(expected_url)
    for i, u in enumerate(retrieved_urls, start=1):
        if normalize_url(u) == exp:
            return 1.0 / i
    return 0.0


def run_rag_eval() -> tuple[pd.DataFrame, pd.DataFrame, RagSummary]:
    # Import pipeline locally to use the exact retrieval logic currently in production.
    import sys

    sys.path.insert(0, str(CHATBOT_DIR))

    from chatbot_pipeline_v1 import retrieve  # noqa: WPS433

    eval_df = pd.read_csv(RAG_EVAL_CSV)

    rows: list[dict[str, Any]] = []

    for _, item in eval_df.iterrows():
        question = str(item["question"])
        expected_url = normalize_url(item["expected_url"])

        intent, confidence, results = retrieve(question, top_k=5)

        retrieved_urls = [normalize_url(r.get("source_url", "")) for r in results]

        top1_url = retrieved_urls[0] if len(retrieved_urls) >= 1 else ""
        top3_urls = retrieved_urls[:3]
        top5_urls = retrieved_urls[:5]

        hit_top1 = expected_url == top1_url
        hit_top3 = expected_url in top3_urls
        hit_top5 = expected_url in top5_urls

        rr = reciprocal_rank(retrieved_urls, expected_url)

        rows.append(
            {
                "question": question,
                "expected_url": expected_url,
                "predicted_intent": intent,
                "intent_confidence": float(confidence),
                "top1_url": top1_url,
                "top1_title": results[0].get("title", "") if results else "",
                "top1_page_type": results[0].get("page_type", "") if results else "",
                "top1_semantic_topic": results[0].get("semantic_topic", "") if results else "",
                "top1_score": float(results[0].get("score", 0.0)) if results else 0.0,
                "hit_top1": bool(hit_top1),
                "hit_top3": bool(hit_top3),
                "hit_top5": bool(hit_top5),
                "reciprocal_rank": float(rr),
                "top5_urls": " | ".join(retrieved_urls),
            }
        )

    results_df = pd.DataFrame(rows)

    summary = RagSummary(
        samples=int(len(results_df)),
        top1_accuracy=float(results_df["hit_top1"].mean()),
        top3_recall=float(results_df["hit_top3"].mean()),
        top5_recall=float(results_df["hit_top5"].mean()),
        mrr=float(results_df["reciprocal_rank"].mean()),
    )

    by_intent = (
        results_df.groupby("predicted_intent")
        .agg(
            samples=("question", "count"),
            top1_accuracy=("hit_top1", "mean"),
            top3_recall=("hit_top3", "mean"),
            top5_recall=("hit_top5", "mean"),
            mrr=("reciprocal_rank", "mean"),
            avg_confidence=("intent_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["samples", "top1_accuracy"], ascending=[False, False])
    )

    return results_df, by_intent, summary


def plot_rag(results_df: pd.DataFrame, by_intent: pd.DataFrame, summary: RagSummary) -> None:
    sns.set_theme(style="whitegrid")

    # Summary bar
    plt.figure(figsize=(7, 4))
    metrics = {
        "Hit@1": summary.top1_accuracy,
        "Hit@3": summary.top3_recall,
        "Hit@5": summary.top5_recall,
        "MRR": summary.mrr,
    }
    ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    ax.set_ylim(0, 1)
    ax.set_title("RAG retrieval metrics (pipeline_v1)")
    ax.set_ylabel("Score")
    for i, v in enumerate(metrics.values()):
        ax.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rag_metrics_summary.png", dpi=160)
    plt.close()

    # Score distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(results_df["top1_score"], bins=30, kde=True)
    plt.title("Top-1 retrieval score distribution")
    plt.xlabel("top1_score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rag_top1_score_hist.png", dpi=160)
    plt.close()

    # Confidence distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(results_df["intent_confidence"], bins=30, kde=True)
    plt.title("Intent confidence distribution")
    plt.xlabel("intent_confidence")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "intent_confidence_hist.png", dpi=160)
    plt.close()

    # By intent (top intents)
    top_intents = by_intent.sort_values("samples", ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=top_intents, x="predicted_intent", y="top1_accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Hit@1 by predicted intent (top 12 by volume)")
    ax.set_xlabel("predicted_intent")
    ax.set_ylabel("Hit@1")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rag_hit1_by_intent.png", dpi=160)
    plt.close()


def run_intent_eval() -> tuple[pd.DataFrame, dict[str, Any]]:
    import joblib

    model_path = CHATBOT_DIR / "models" / "intent_classifier.joblib"
    model = joblib.load(model_path)

    test_df = pd.read_csv(INTENT_TEST_CSV)

    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(str)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    max_probs = probs.max(axis=1)

    labels = sorted(y_test.unique())

    metrics: dict[str, Any] = {
        "samples": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "avg_confidence": float(max_probs.mean()),
        "min_confidence": float(max_probs.min()),
        "max_confidence": float(max_probs.max()),
        "labels": labels,
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }

    detailed = test_df.copy()
    detailed["predicted"] = preds
    detailed["confidence"] = max_probs

    cm = confusion_matrix(y_test, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, cmap="Blues", annot=False, fmt="d")
    plt.title("Intent classifier confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "intent_confusion_matrix.png", dpi=180)
    plt.close()

    # Confidence histogram
    plt.figure(figsize=(7, 4))
    sns.histplot(detailed["confidence"], bins=30, kde=True)
    plt.title("Intent model confidence distribution")
    plt.xlabel("confidence")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "intent_model_confidence_hist.png", dpi=160)
    plt.close()

    return detailed, metrics


def main() -> None:
    # RAG eval
    rag_results_df, rag_by_intent_df, rag_summary = run_rag_eval()

    (OUTPUT_DIR / "rag").mkdir(exist_ok=True)
    rag_results_df.to_csv(OUTPUT_DIR / "rag" / "rag_v1_detailed.csv", index=False)
    rag_by_intent_df.to_csv(OUTPUT_DIR / "rag" / "rag_v1_by_intent.csv", index=False)
    with open(OUTPUT_DIR / "rag" / "rag_v1_summary.json", "w", encoding="utf-8") as f:
        json.dump(rag_summary.__dict__, f, indent=2, ensure_ascii=False)

    plot_rag(rag_results_df, rag_by_intent_df, rag_summary)

    # Intent eval
    intent_detailed_df, intent_metrics = run_intent_eval()
    (OUTPUT_DIR / "intent").mkdir(exist_ok=True)
    intent_detailed_df.to_csv(OUTPUT_DIR / "intent" / "intent_detailed.csv", index=False)
    with open(OUTPUT_DIR / "intent" / "intent_metrics.json", "w", encoding="utf-8") as f:
        json.dump(intent_metrics, f, indent=2, ensure_ascii=False)

    print("\nDone. Outputs in:")
    print(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()

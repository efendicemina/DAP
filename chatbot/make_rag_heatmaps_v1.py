"""Create RAG heatmaps for pipeline_v1 evaluation.

Reads:
- chatbot/eval_reports_v1/rag/rag_v1_detailed.csv
- chatbot/mpr_dataset_v5/chunks_v5.csv (for expected page_type/topic by URL)

Writes PNGs to:
- chatbot/eval_reports_v1/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent
CHATBOT_DIR = REPO_ROOT / "chatbot"

DETAILED_CSV = CHATBOT_DIR / "eval_reports_v1" / "rag" / "rag_v1_detailed.csv"
CHUNKS_CSV = CHATBOT_DIR / "mpr_dataset_v5" / "chunks_v5.csv"
OUTPUT_DIR = CHATBOT_DIR / "eval_reports_v1"


def normalize_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/")


def mode_or_first(series: pd.Series) -> str:
    series = series.dropna().astype(str)
    if series.empty:
        return ""
    counts = series.value_counts()
    return str(counts.index[0])


def build_url_metadata() -> pd.DataFrame:
    df = pd.read_csv(CHUNKS_CSV, usecols=["source_url", "page_type", "semantic_topic"])
    df["url_norm"] = df["source_url"].map(normalize_url)

    meta = (
        df.groupby("url_norm")
        .agg(
            expected_page_type=("page_type", mode_or_first),
            expected_semantic_topic=("semantic_topic", mode_or_first),
        )
        .reset_index()
    )

    return meta


def plot_heatmap(ct: pd.DataFrame, title: str, out_path: Path, fmt: str = "d", vmax: float | None = None) -> None:
    plt.figure(figsize=(max(8, 0.7 * len(ct.columns) + 3), max(6, 0.45 * len(ct.index) + 2)))
    sns.heatmap(ct, cmap="Blues", annot=False, fmt=fmt, linewidths=0.4, linecolor="#f0f0f0", vmax=vmax)
    plt.title(title)
    plt.xlabel(ct.columns.name or "")
    plt.ylabel(ct.index.name or "")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    if not DETAILED_CSV.exists():
        raise FileNotFoundError(f"Missing: {DETAILED_CSV}")
    if not CHUNKS_CSV.exists():
        raise FileNotFoundError(f"Missing: {CHUNKS_CSV}")

    sns.set_theme(style="white")

    detailed = pd.read_csv(DETAILED_CSV)
    detailed["expected_url_norm"] = detailed["expected_url"].map(normalize_url)

    meta = build_url_metadata()

    merged = detailed.merge(meta, left_on="expected_url_norm", right_on="url_norm", how="left")

    # Page type heatmap (counts)
    ct_page = pd.crosstab(
        merged["expected_page_type"].fillna(""),
        merged["top1_page_type"].fillna(""),
    )
    ct_page.index.name = "expected_page_type"
    ct_page.columns.name = "top1_page_type"

    # Remove empty label row/col if any
    if "" in ct_page.index:
        ct_page = ct_page.drop(index="")
    if "" in ct_page.columns:
        ct_page = ct_page.drop(columns="")

    # Sort by frequency
    ct_page = ct_page.loc[ct_page.sum(axis=1).sort_values(ascending=False).index]
    ct_page = ct_page[ct_page.sum(axis=0).sort_values(ascending=False).index]

    plot_heatmap(
        ct_page,
        title="RAG heatmap: expected page_type vs retrieved top1 page_type (counts)",
        out_path=OUTPUT_DIR / "rag_heatmap_page_type_counts.png",
    )

    # Page type heatmap (row-normalized)
    ct_page_norm = ct_page.div(ct_page.sum(axis=1), axis=0).fillna(0.0)
    plt.figure(figsize=(max(8, 0.7 * len(ct_page_norm.columns) + 3), max(6, 0.45 * len(ct_page_norm.index) + 2)))
    sns.heatmap(ct_page_norm, cmap="Blues", annot=False, vmin=0.0, vmax=1.0, linewidths=0.4, linecolor="#f0f0f0")
    plt.title("RAG heatmap: expected page_type vs retrieved top1 page_type (row %)")
    plt.xlabel("top1_page_type")
    plt.ylabel("expected_page_type")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rag_heatmap_page_type_rowpct.png", dpi=180)
    plt.close()

    # Semantic topic heatmap (counts)
    ct_topic = pd.crosstab(
        merged["expected_semantic_topic"].fillna(""),
        merged["top1_semantic_topic"].fillna(""),
    )
    ct_topic.index.name = "expected_semantic_topic"
    ct_topic.columns.name = "top1_semantic_topic"

    if "" in ct_topic.index:
        ct_topic = ct_topic.drop(index="")
    if "" in ct_topic.columns:
        ct_topic = ct_topic.drop(columns="")

    ct_topic = ct_topic.loc[ct_topic.sum(axis=1).sort_values(ascending=False).index]
    ct_topic = ct_topic[ct_topic.sum(axis=0).sort_values(ascending=False).index]

    plot_heatmap(
        ct_topic,
        title="RAG heatmap: expected semantic_topic vs retrieved top1 semantic_topic (counts)",
        out_path=OUTPUT_DIR / "rag_heatmap_semantic_topic_counts.png",
    )

    # Semantic topic heatmap (row-normalized)
    ct_topic_norm = ct_topic.div(ct_topic.sum(axis=1), axis=0).fillna(0.0)
    plt.figure(figsize=(max(8, 0.7 * len(ct_topic_norm.columns) + 3), max(6, 0.45 * len(ct_topic_norm.index) + 2)))
    sns.heatmap(ct_topic_norm, cmap="Blues", annot=False, vmin=0.0, vmax=1.0, linewidths=0.4, linecolor="#f0f0f0")
    plt.title("RAG heatmap: expected semantic_topic vs retrieved top1 semantic_topic (row %)")
    plt.xlabel("top1_semantic_topic")
    plt.ylabel("expected_semantic_topic")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rag_heatmap_semantic_topic_rowpct.png", dpi=180)
    plt.close()

    # Combined overview (2x2)
    fig_w = max(14, 0.55 * max(len(ct_page.columns), len(ct_topic.columns)) + 10)
    fig_h = max(10, 0.40 * (len(ct_page.index) + len(ct_topic.index)) / 2 + 8)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))

    sns.heatmap(
        ct_page,
        ax=axes[0, 0],
        cmap="Blues",
        annot=False,
        linewidths=0.4,
        linecolor="#f0f0f0",
    )
    axes[0, 0].set_title("page_type (counts)")
    axes[0, 0].set_xlabel("top1_page_type")
    axes[0, 0].set_ylabel("expected_page_type")

    sns.heatmap(
        ct_page_norm,
        ax=axes[0, 1],
        cmap="Blues",
        annot=False,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="#f0f0f0",
    )
    axes[0, 1].set_title("page_type (row %)")
    axes[0, 1].set_xlabel("top1_page_type")
    axes[0, 1].set_ylabel("expected_page_type")

    sns.heatmap(
        ct_topic,
        ax=axes[1, 0],
        cmap="Blues",
        annot=False,
        linewidths=0.4,
        linecolor="#f0f0f0",
    )
    axes[1, 0].set_title("semantic_topic (counts)")
    axes[1, 0].set_xlabel("top1_semantic_topic")
    axes[1, 0].set_ylabel("expected_semantic_topic")

    sns.heatmap(
        ct_topic_norm,
        ax=axes[1, 1],
        cmap="Blues",
        annot=False,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="#f0f0f0",
    )
    axes[1, 1].set_title("semantic_topic (row %)")
    axes[1, 1].set_xlabel("top1_semantic_topic")
    axes[1, 1].set_ylabel("expected_semantic_topic")

    fig.suptitle("RAG heatmaps overview (expected vs retrieved top1)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rag_heatmaps_overview.png", dpi=180)
    plt.close(fig)

    print("Wrote heatmaps to:")
    print(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()

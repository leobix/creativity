"""
Plot similarity distributions for the Doshi/Hauser writers stories dataset using
an embeddings CSV (metadata + emb_0..emb_(d-1)).

Usage (after running doshi_embed.py):
  python3 doshi_plot.py --emb-csv doshi_embeddings.csv --label-col condition

Outputs:
  - plots/doshi_condition.png                (original condition labels)
  - plots/doshi_condition_merge1.png         (Human vs AI-assisted)
  - plots/doshi_condition_merge2.png         (AI Multiple vs AI Single vs Human)
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Headless-safe matplotlib and cache dir.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-config")
)
matplotlib.use("Agg")


def _pairwise_cosine_sims(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.array([])
    sim_matrix = vectors @ vectors.T
    upper = np.triu_indices_from(sim_matrix, k=1)
    return sim_matrix[upper]


def collect_sims_by_label(labels: Iterable[str], embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    df = pd.DataFrame({"label": list(labels)})
    sims_by_label: Dict[str, np.ndarray] = {}
    for label in df["label"].dropna().unique():
        idx = df.index[df["label"] == label].to_numpy()
        sims = _pairwise_cosine_sims(embeddings[idx])
        if sims.size:
            sims_by_label[label] = sims
    return sims_by_label


def plot_distributions(
    sims_by_label: Dict[str, np.ndarray],
    title: str,
    outfile: Path,
    label_map: Dict[str, str] | None = None,
    human_labels: set[str] | None = None,
    draw_order: List[str] | None = None,
    style_map: Dict[str, str] | None = None,
) -> None:
    if not sims_by_label:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    human_labels = human_labels or set()
    human_color = "#5a5a5a"
    ai_palette = ["#b11116", "#c53a2f", "#d95d39", "#e88763"]
    ai_linestyles = ["--", ":", "-."]  # reserve solid for human
    human_linestyle = "-"
    xs = np.linspace(0, 1, 400)

    plt.figure(figsize=(10, 6), dpi=220)
    items = list(sims_by_label.items())
    if draw_order:
        order_index = {k: i for i, k in enumerate(draw_order)}
        items.sort(key=lambda kv: order_index.get(kv[0], len(draw_order)))
    elif human_labels:
        items.sort(key=lambda kv: (0 if kv[0] in human_labels else 1, kv[0]))
    else:
        items.sort(key=lambda kv: kv[0])

    for idx, (raw_label, sims) in enumerate(items):
        label = label_map.get(raw_label, raw_label) if label_map else raw_label
        kde = gaussian_kde(sims)
        ys = kde(xs)
        mean_val = sims.mean()
        mean_y = kde(mean_val)

        is_human = raw_label in human_labels or label in human_labels
        if is_human:
            color = human_color
            linestyle = human_linestyle
        else:
            color = ai_palette[idx % len(ai_palette)]
            linestyle = (style_map or {}).get(raw_label) or ai_linestyles[idx % len(ai_linestyles)]

        plt.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0 if is_human else 1.5,
            linestyle=linestyle,
            label=f"{label} (mean = {mean_val:.3f})",
        )
        plt.fill_between(xs, ys, color=color, alpha=0.18)
        plt.scatter(
            [mean_val],
            [mean_y],
            color=color,
            edgecolors="white",
            linewidths=0.8,
            s=45,
            zorder=5,
        )

    plt.title(title, fontsize=14, pad=14)
    plt.xlabel("Cosine similarity within condition", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(0, 1)
    plt.legend(frameon=False, fontsize=10)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def merge_labels(series: pd.Series, combine_map: Dict[str, List[str]]) -> pd.Series:
    remap = {}
    for new_label, originals in combine_map.items():
        for orig in originals:
            remap[orig] = new_label
    return series.map(remap).fillna(series)


def load_embeddings_from_csv(path: Path, label_col: str) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if emb_cols:
        embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    elif "openai_embedding" in df.columns:
        import ast

        embeddings = np.vstack(df["openai_embedding"].apply(ast.literal_eval).values).astype(
            "float32"
        )
    elif "Embedding" in df.columns:
        import ast

        embeddings = np.vstack(df["Embedding"].apply(ast.literal_eval).values).astype(
            "float32"
        )
    else:
        raise ValueError("No embedding columns found (emb_* / openai_embedding / Embedding).")
    labels = df[label_col]
    return labels, embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb-csv",
        default="doshi_embeddings.csv",
        help="CSV with embeddings (emb_0... columns or list column).",
    )
    parser.add_argument(
        "--label-col",
        default="condition",
        help="Name of the label column for grouping",
    )
    args = parser.parse_args()

    data_path = Path(args.emb_csv)
    if not data_path.exists():
        raise SystemExit(f"Embeddings CSV not found: {data_path}")

    labels, embeddings = load_embeddings_from_csv(data_path, args.label_col)
    labels = labels.astype(str)

    # Raw labels plot
    sims_raw = collect_sims_by_label(labels, embeddings)
    plot_distributions(
        sims_raw,
        title="Doshi & Hauser – Story writing (raw labels)",
        outfile=Path("plots/doshi_condition.png"),
        human_labels={"Human only", "Human-only"},
        draw_order=["Human only", "Human with 5 GenAI ideas", "Human with 1 GenAI idea"],
        style_map={"Human with 5 GenAI ideas": "--", "Human with 1 GenAI idea": ":"},
        label_map={"Human only": "Human-only", "Human with 5 GenAI ideas": "AI-assisted ideation"},
    )

    # Merge map 1: AI-assisted vs Human
    merge_map = {
        "AI-assisted": ["Human with 5 GenAI ideas", "Human with 1 GenAI idea"],
        "Human": ["Human only"],
    }
    merged_labels = merge_labels(labels, merge_map)
    sims_merge1 = collect_sims_by_label(merged_labels, embeddings)
    plot_distributions(
        sims_merge1,
        title="Doshi & Hauser – Story writing (AI-assisted vs Human)",
        outfile=Path("plots/doshi_condition_merge1.png"),
        label_map={"AI-assisted": "AI-assisted ideation", "Human": "Human-only"},
        human_labels={"Human", "Human-only"},
        draw_order=["Human", "AI-assisted"],
        style_map={"AI-assisted": "--"},
    )

    # Merge map 2: AI multiple vs AI single vs Human
    merge_map_2 = {
        "AI – Multiple ideas": ["Human with 5 GenAI ideas"],
        "AI – Single idea": ["Human with 1 GenAI idea"],
        "Human": ["Human only"],
    }
    merged_labels2 = merge_labels(labels, merge_map_2)
    sims_merge2 = collect_sims_by_label(merged_labels2, embeddings)
    plot_distributions(
        sims_merge2,
        title="Doshi & Hauser – Story writing (AI mult/single vs Human)",
        outfile=Path("plots/doshi_condition_merge2.png"),
        label_map={
            "AI – Multiple ideas": "AI-assisted ideation (5)",
            "AI – Single idea": "AI-assisted ideation (1)",
            "Human": "Human-only",
        },
        human_labels={"Human", "Human-only"},
        draw_order=["Human", "AI – Multiple ideas", "AI – Single idea"],
        style_map={"AI – Multiple ideas": "--", "AI – Single idea": ":"},
    )


if __name__ == "__main__":
    main()

"""
Create a 2x2 panel of similarity distributions:
  - Top-left: Doshi/Hauser (Human vs Human with 5 GenAI ideas)
  - Top-right: Crowdless (AI vs Human)
  - Bottom-left: Humor (AI combined vs Human)
  - Bottom-right: Story (AI combined vs Human)

Assumes embeddings CSVs already exist:
  - doshi_embeddings.csv            (from doshi_embed.py)
  - crowdless_embeddings.csv        (from crowdless_embed.py)
  - humor_embeddings.csv            (pre-embedded columns emb_*)
  - story_embeddings_v2.csv         (Embedding column as list)

Run from repo root:
  python3 combined_quadrant_plot.py

Outputs:
  - plots/combined_2x2.png
"""

from __future__ import annotations

import argparse
import ast
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.gridspec as gridspec
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
# Use Georgia font where available; fall back to serif otherwise.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

HUMAN_COLOR = "#4D4D4D"  # HBR neutral slate
AI_PALETTE = ["#8B1E3F", "#8B1E3F", "#8B1E3F", "#8B1E3F"]  # HBR wine red (consistent hue)
AI_LINESTYLES = ["--", "--", "--"]  # base style; dash patterns applied below
HUMAN_LINESTYLE = "-"
AI_DASH_PATTERNS = [(10, 4), (4, 2), (1, 2)]  # longest dashes first


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


def plot_panel(
    ax: plt.Axes,
    sims_by_label: Dict[str, np.ndarray],
    title: str,
    label_map: Dict[str, str] | None = None,
    human_labels: set[str] | None = None,
    draw_order: List[str] | None = None,
    style_map: Dict[str, str] | None = None,
    legend_loc: str = "upper right",
    legend_kwargs: Dict = None,
    marker_map: Dict[str, str] | None = None,
    ylim: tuple | None = None,
) -> None:
    legend_kwargs = legend_kwargs or {}
    human_labels = human_labels or set()
    xs = np.linspace(0, 1, 400)

    legend_entries = []

    items = list(sims_by_label.items())
    if draw_order:
        order_index = {k: i for i, k in enumerate(draw_order)}
        items.sort(key=lambda kv: order_index.get(kv[0], len(draw_order)))
    else:
        items.sort(key=lambda kv: (0 if kv[0] in human_labels else 1, kv[0]))

    for idx, (raw_label, sims) in enumerate(items):
        label = label_map.get(raw_label, raw_label) if label_map else raw_label
        kde = gaussian_kde(sims)
        ys = kde(xs)
        mean_val = sims.mean()
        mean_y = kde(mean_val)

        is_human = raw_label in human_labels or label in human_labels
        if is_human:
            color = HUMAN_COLOR
            linestyle = HUMAN_LINESTYLE
            linewidth = 2.0
            dash_pattern = None
        else:
            color = AI_PALETTE[idx % len(AI_PALETTE)]
            linestyle = (style_map or {}).get(raw_label) or AI_LINESTYLES[idx % len(AI_LINESTYLES)]
            linewidth = 1.5
            dash_pattern = AI_DASH_PATTERNS[idx % len(AI_DASH_PATTERNS)]

        line, = ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle)
        if dash_pattern:
            line.set_dashes(dash_pattern)
        if is_human:
            ax.fill_between(xs, ys, color=color, alpha=0.18)

        marker = (marker_map or {}).get(raw_label) or ("o" if is_human else "s")
        ax.scatter([mean_val], [mean_y], color=color, edgecolors="white", linewidths=0.8, s=35, zorder=5, marker=marker)

        legend_entries.append(
            {
                "label": f"{label} (μ={mean_val:.2f})",
                "color": color,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "marker": marker,
                "dash_pattern": dash_pattern,
            }
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(0, 1)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=10.5)
    handles = []
    labels_out = []
    for entry in legend_entries:
        ms = 8 if entry["marker"] in {"^", "v"} else 7
        line_kwargs = dict(
            color=entry["color"],
            linestyle=entry["linestyle"],
            linewidth=entry["linewidth"],
            marker=entry["marker"],
            markersize=ms,
            markerfacecolor=entry["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        handle = plt.Line2D([0], [0], **line_kwargs)
        if entry["dash_pattern"]:
            handle.set_dashes(entry["dash_pattern"])
        handles.append(handle)
        labels_out.append(entry["label"])
    ax.legend(handles, labels_out, frameon=False, fontsize=11, loc=legend_loc, **legend_kwargs)


def load_embs_csv(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if emb_cols:
        embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    elif "openai_embedding" in df.columns:
        embeddings = np.vstack(df["openai_embedding"].apply(ast.literal_eval).values).astype("float32")
    elif "Embedding" in df.columns:
        embeddings = np.vstack(df["Embedding"].apply(ast.literal_eval).values).astype("float32")
    else:
        raise ValueError(f"No embedding columns found in {path}")
    return df, embeddings


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_doshi(path: Path) -> Dict[str, np.ndarray]:
    df, emb = load_embs_csv(path)
    labels = df["condition"].astype(str)
    # Filter to Human only vs Human with 5 GenAI ideas
    mask = labels.isin(["Human only", "Human with 5 GenAI ideas"])
    return collect_sims_by_label(labels[mask], emb[mask])


def load_crowdless(path: Path) -> Dict[str, np.ndarray]:
    df, emb = load_embs_csv(path)
    labels = df["Level"].astype(str)
    merge_map = {
        "AI": ["0", "1", "2", "A", "B", "C"],
        "Human crowd": ["H"],
    }
    remap = {}
    for new, olds in merge_map.items():
        for o in olds:
            remap[o] = new
    merged = labels.map(remap).fillna(labels)
    return collect_sims_by_label(merged, emb)


def load_humor(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb = df[emb_cols].to_numpy(dtype=np.float32)
    labels = df["Group"].astype(str)
    return collect_sims_by_label(labels, emb)


def load_story(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    embeddings = np.vstack(df["Embedding"].apply(ast.literal_eval).values).astype("float32")
    labels = df["group"].astype(str)
    return collect_sims_by_label(labels, embeddings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doshi-csv", default="doshi_embeddings.csv", help="Doshi embeddings CSV path")
    parser.add_argument("--crowdless-csv", default="crowdless_embeddings.csv", help="Crowdless embeddings CSV path")
    parser.add_argument("--humor-csv", default="humor_embeddings.csv", help="Humor embeddings CSV path")
    parser.add_argument("--story-csv", default="story_embeddings_v2.csv", help="Story embeddings CSV path")
    parser.add_argument("--out", default="plots/combined_2x2.png", help="Output image path")
    parser.add_argument("--variant", choices=["classic", "aesthetic"], default="classic", help="Which layout to generate")
    args = parser.parse_args()

    if args.variant == "classic":
        sims_doshi = load_doshi(Path(args.doshi_csv))
        sims_crowdless = load_crowdless(Path(args.crowdless_csv))
        sims_humor = load_humor(Path(args.humor_csv))
        sims_story = load_story(Path(args.story_csv))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=220)
        for ax in axes.flat:
            ax.grid(color="#E5E5E5", linewidth=1.0, alpha=0.25, linestyle="-")

        plot_panel(
            axes[0, 0],
            sims_doshi,
            title="Doshi & Hauser – Story writing",
            label_map={"Human only": "Human-only", "Human with 5 GenAI ideas": "AI-assisted ideation"},
            human_labels={"Human-only", "Human only"},
            draw_order=["Human only", "Human with 5 GenAI ideas"],
            style_map={"Human with 5 GenAI ideas": "--"},
            marker_map={"Human only": "o", "Human with 5 GenAI ideas": "s"},
        )
        axes[0, 0].text(0.02, 0.95, "(a)", transform=axes[0, 0].transAxes, fontsize=14, fontweight="bold", ha="left", va="top")

        plot_panel(
            axes[0, 1],
            sims_crowdless,
            title="Boussioux et al. – Circular economy solutions",
            label_map={"AI": "Human-AI", "Human crowd": "Human crowd"},
            human_labels={"Human crowd"},
            draw_order=["Human crowd", "AI"],
            style_map={"AI": "--"},
            legend_loc="upper left",
            legend_kwargs={"bbox_to_anchor": (0.18, 1.0)},
            marker_map={"Human crowd": "o", "AI": "s"},
        )
        axes[0, 1].text(0.02, 0.95, "(b)", transform=axes[0, 1].transAxes, fontsize=14, fontweight="bold", ha="left", va="top")

        plot_panel(
            axes[1, 0],
            sims_humor,
            title="Salas & Hosanagar – Humor caption contest",
            label_map={
                "H-H": "Human-only",
                "AI-H": "AI in ideation",
                "H-AI": "AI in selection",
                "AI-AI": "AI in both",
            },
            human_labels={"H-H", "Human-only"},
            draw_order=["H-H", "AI-H", "H-AI", "AI-AI"],
            style_map={"AI-H": ":", "H-AI": "--", "AI-AI": "-."},
            marker_map={"H-H": "o", "AI-H": "s", "H-AI": "^", "AI-AI": "v"},
            ylim=(0, 7.25),
        )
        axes[1, 0].text(0.02, 0.95, "(c)", transform=axes[1, 0].transAxes, fontsize=14, fontweight="bold", ha="left", va="top")

        plot_panel(
            axes[1, 1],
            sims_story,
            title="Hosanagar & Ahn – Story writing",
            label_map={
            "Day1": "Human-only",
            "A - Creativity": "Human creativity",
            "B - Confirmation": "Human confirmation",
            "C - Copilot": "Copilot",
        },
            human_labels={"Day1", "Human-only"},
            draw_order=["Day1", "A - Creativity", "B - Confirmation", "C - Copilot"],
            style_map={"A - Creativity": "--", "B - Confirmation": ":", "C - Copilot": "-."},
            marker_map={"Day1": "o", "A - Creativity": "s", "B - Confirmation": "^", "C - Copilot": "v"},
            ylim=(0, 7.25),
        )
        axes[1, 1].text(0.02, 0.95, "(d)", transform=axes[1, 1].transAxes, fontsize=14, fontweight="bold", ha="left", va="top")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.out, bbox_inches="tight")
        pdf_out = Path(args.out).with_suffix(".pdf")
        plt.savefig(pdf_out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Wrote combined plot to {args.out}")
    else:
        plot_quadrant_az(args)


def plot_quadrant_az(args) -> None:
    """Variant: higher aesthetic polish with shared y, unified legend, subtle grids."""
    sims_doshi = load_doshi(Path(args.doshi_csv))
    sims_crowdless = load_crowdless(Path(args.crowdless_csv))
    sims_humor = load_humor(Path(args.humor_csv))
    sims_story = load_story(Path(args.story_csv))

    fig = plt.figure(figsize=(12, 10), dpi=320)
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.15)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    # Minimalist grid style (white background)
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(color="#E5E5E5", linewidth=1.0, alpha=0.25, linestyle="-")

    plot_panel(
        axes[0],
        sims_doshi,
        title="Doshi & Hauser – Story writing",
        label_map={"Human only": "Human-only", "Human with 5 GenAI ideas": "AI-assisted ideation"},
        human_labels={"Human-only", "Human only"},
        draw_order=["Human only", "Human with 5 GenAI ideas"],
        style_map={"Human with 5 GenAI ideas": "--"},
    )
    plot_panel(
        axes[1],
        sims_crowdless,
        title="Boussioux et al. – Circular economy solutions",
        label_map={"AI": "Human-AI", "Human crowd": "Human crowd"},
        human_labels={"Human crowd"},
        draw_order=["Human crowd", "AI"],
        style_map={"AI": "--"},
    )
    plot_panel(
        axes[2],
        sims_humor,
        title="Salas & Hosanagar – Humor caption contest",
        label_map={
            "H-H": "Human-only",
            "AI-H": "AI in ideation",
            "H-AI": "AI in selection",
            "AI-AI": "AI in both",
        },
        human_labels={"H-H", "Human-only"},
        draw_order=["H-H", "AI-H", "H-AI", "AI-AI"],
        style_map={"AI-H": ":", "H-AI": "--", "AI-AI": "-."},
    )
    plot_panel(
        axes[3],
        sims_story,
        title="Hosanagar & Ahn – Story writing",
        label_map={
            "Day1": "Human-only",
            "A - Creativity": "Human creativity",
            "B - Confirmation": "Human confirmation",
            "C - Copilot": "Copilot",
        },
            human_labels={"Day1", "Human-only"},
            draw_order=["Day1", "A - Creativity", "B - Confirmation", "C - Copilot"],
            style_map={"A - Creativity": "--", "B - Confirmation": ":", "C - Copilot": "-."},
            ylim=(0, 7),
        )
    axes[1, 1].text(0.02, 0.95, "(d)", transform=axes[1, 1].transAxes, fontsize=14, fontweight="bold", ha="left", va="top")

    out = Path(args.out).with_name(Path(args.out).stem + "_az.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote aesthetic variant to {out}")


if __name__ == "__main__":
    main()

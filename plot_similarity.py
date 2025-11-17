import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Force matplotlib to use a writable cache location (fixed paths can break in sandboxes).
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-config")
)

import matplotlib

# Use a headless backend so plots render without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pairwise_cosine_sims(vectors: np.ndarray) -> np.ndarray:
    """Return unique pairwise cosine similarities for a matrix of vectors."""
    if len(vectors) < 2:
        return np.array([])
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sim_matrix = normed @ normed.T
    upper = np.triu_indices_from(sim_matrix, k=1)
    return sim_matrix[upper]


def collect_similarities_by_group(
    groups: Iterable[str], embeddings: np.ndarray
) -> Dict[str, np.ndarray]:
    """Group embeddings and compute pairwise cosine similarities within each group."""
    df = pd.DataFrame({"group": list(groups)})
    sims_by_group: Dict[str, np.ndarray] = OrderedDict()
    for group_value in sorted(df["group"].dropna().unique()):
        idx = df.index[df["group"] == group_value].to_numpy()
        sims = _pairwise_cosine_sims(embeddings[idx])
        if sims.size:
            sims_by_group[group_value] = sims
    return sims_by_group


def plot_distribution(
    sims_by_group: Dict[str, np.ndarray],
    title: str,
    outfile: Path,
    palette: str = "tab10",
    label_map: Dict[str, str] | None = None,
) -> None:
    """Create a polished KDE plot for similarity distributions."""
    if not sims_by_group:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6), dpi=220)
    # HBR-inspired gray/red palette (ordered dark to light with a deep red accent).
    hbr_palette = ["#b11116", "#d95d39", "#5a5a5a", "#8a8a8a"]
    colors = (hbr_palette * ((len(sims_by_group) // len(hbr_palette)) + 1))[
        : len(sims_by_group)
    ]
    linestyles = ["-", "--", ":", "-."]

    xs = np.linspace(0, 1, 400)

    for idx, ((raw_label, sims), color) in enumerate(
        zip(sims_by_group.items(), colors)
    ):
        label = label_map.get(raw_label, raw_label) if label_map else raw_label
        mean_val = sims.mean()
        kde = gaussian_kde(sims)
        ys = kde(xs)
        mean_y = kde(mean_val)
        plt.plot(
            xs,
            ys,
            color=color,
            linewidth=2.2,
            linestyle=linestyles[idx % len(linestyles)],
            label=f"{label} (mean = {mean_val:.3f})",
        )
        plt.fill_between(xs, ys, color=color, alpha=0.06)
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
    plt.xlabel("Cosine similarity to entries in same condition", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(0, 1)
    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def load_humor_embeddings(path: Path) -> Tuple[pd.Series, np.ndarray]:
    """Load the humor CSV that already has one column per embedding value."""
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    return df["Group"], embeddings


def load_story_embeddings(path: Path) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Load the cleaned story CSV where embeddings are in a single column of lists."""
    import ast

    df = pd.read_csv(path, low_memory=False)
    embeddings = np.vstack(df["Embedding"].apply(ast.literal_eval).values).astype(
        "float32"
    )
    return df["group"], df["group2"], embeddings


def main() -> None:
    base = Path(__file__).parent
    plots_dir = base / "plots"

    # Story modality
    story_label_map = {
        "Day1": "No AI Support",
        "A - Creativity": "Human Creativity",
        "B - Confirmation": "Human Confirmation",
        "C - Copilot": "Copilot",
    }
    story_groups, story_group2, story_embeddings = load_story_embeddings(
        base / "story_embeddings_v2.csv"
    )
    story_by_group = collect_similarities_by_group(story_groups, story_embeddings)
    plot_distribution(
        story_by_group,
        title="Story: similarity within collaboration condition",
        outfile=plots_dir / "story_by_group.png",
        palette="Set1",
        label_map=story_label_map,
    )

    story_by_group2 = collect_similarities_by_group(story_group2, story_embeddings)
    plot_distribution(
        story_by_group2,
        title="Story: similarity with vs without AI",
        outfile=plots_dir / "story_by_group2.png",
        palette="Set2",
        label_map={"ai": "With AI", "without_ai": "Without AI"},
    )

    # Humor modality
    humor_groups, humor_embeddings = load_humor_embeddings(base / "humor_embeddings.csv")
    humor_label_map = {
        "H-H": "No AI Support",
        "AI-H": "AI assist: ideation",
        "H-AI": "AI assist: selection",
        "AI-AI": "AI assist: ideation + selection",
    }
    humor_by_group = collect_similarities_by_group(humor_groups, humor_embeddings)
    plot_distribution(
        humor_by_group,
        title="Humor: similarity within collaboration condition",
        outfile=plots_dir / "humor_by_group.png",
        palette="tab10",
        label_map=humor_label_map,
    )


if __name__ == "__main__":
    main()

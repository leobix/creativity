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


def collect_similarities_by_group_contest_balanced(
    groups: Iterable[str], contests: Iterable, embeddings: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute similarities by group, balancing across contests.

    Each contest's similarities contribute equal total weight within a group by
    assigning weight 1/len(sims_in_contest) to each similarity sample.
    Returns sims_by_group and matching weights_by_group.
    """
    df = pd.DataFrame({"group": list(groups), "contest": list(contests)})
    sims_by_group: Dict[str, List[np.ndarray]] = {}
    weights_by_group: Dict[str, List[np.ndarray]] = {}

    for group_value in df["group"].dropna().unique():
        for contest_value in df.loc[df["group"] == group_value, "contest"].dropna().unique():
            idx = df.index[(df["group"] == group_value) & (df["contest"] == contest_value)].to_numpy()
            sims = _pairwise_cosine_sims(embeddings[idx])
            if not sims.size:
                continue
            weight = np.full_like(sims, 1.0 / len(sims), dtype=float)
            sims_by_group.setdefault(group_value, []).append(sims)
            weights_by_group.setdefault(group_value, []).append(weight)

    sims_by_group_final = {}
    weights_by_group_final = {}
    for group_value, chunks in sims_by_group.items():
        sims_concat = np.concatenate(chunks)
        weights_concat = np.concatenate(weights_by_group[group_value])
        sims_by_group_final[group_value] = sims_concat
        weights_by_group_final[group_value] = weights_concat

    return sims_by_group_final, weights_by_group_final


def plot_distribution(
    sims_by_group: Dict[str, np.ndarray],
    title: str,
    outfile: Path,
    palette: str = "tab10",
    label_map: Dict[str, str] | None = None,
    weights_by_group: Dict[str, np.ndarray] | None = None,
    human_labels: set[str] | None = None,
    draw_order: List[str] | None = None,
    style_map: Dict[str, str] | None = None,
) -> None:
    """Create a polished KDE plot for similarity distributions."""
    if not sims_by_group:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6), dpi=220)
    human_labels = human_labels or set()
    human_color = "#5a5a5a"
    ai_palette = ["#b11116", "#c53a2f", "#d95d39", "#e88763"]
    ai_linestyles = ["--", ":", "-."]  # reserve solid for human only
    human_linestyle = "-"

    xs = np.linspace(0, 1, 400)

    items = list(sims_by_group.items())
    if draw_order:
        order_index = {k: i for i, k in enumerate(draw_order)}
        items.sort(key=lambda kv: order_index.get(kv[0], len(draw_order)))
    else:
        # human first, then alphabetical
        items.sort(key=lambda kv: (0 if kv[0] in human_labels else 1, kv[0]))

    for idx, (raw_label, sims) in enumerate(items):
        label = label_map.get(raw_label, raw_label) if label_map else raw_label
        weights = None
        if weights_by_group and raw_label in weights_by_group:
            weights = weights_by_group[raw_label]
        mean_val = (
            np.average(sims, weights=weights) if weights is not None else sims.mean()
        )
        kde = gaussian_kde(sims, weights=weights)
        ys = kde(xs)
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
        "Day1": "Human-only",
        "A - Creativity": "Human creativity",
        "B - Confirmation": "Human confirmation",
        "C - Copilot": "Copilot",
    }
    story_groups, story_group2, story_embeddings = load_story_embeddings(
        base / "story_embeddings_v2.csv"
    )
    story_by_group = collect_similarities_by_group(story_groups, story_embeddings)
    plot_distribution(
        story_by_group,
        title="Hosanagar & Ahn – Story writing (by condition)",
        outfile=plots_dir / "story_by_group.png",
        palette="Set1",
        label_map=story_label_map,
        human_labels={"Day1", "Human-only"},
        draw_order=["Day1", "A - Creativity", "B - Confirmation", "C - Copilot"],
        style_map={"A - Creativity": "--", "B - Confirmation": ":", "C - Copilot": "-."},
        legend_loc="upper right",
        legend_kwargs={"frameon": False, "fontsize": 10},
    )

    story_by_group2 = collect_similarities_by_group(story_group2, story_embeddings)
    plot_distribution(
        story_by_group2,
        title="Hosanagar & Ahn – Story writing (with vs without AI)",
        outfile=plots_dir / "story_by_group2.png",
        palette="Set2",
        label_map={"ai": "With AI", "without_ai": "Without AI"},
        human_labels={"without_ai", "Without AI"},
        draw_order=["without_ai", "ai"],
        style_map={"ai": "--"},
    )

    # Humor modality
    humor_groups, humor_embeddings = load_humor_embeddings(base / "humor_embeddings.csv")
    humor_label_map = {
        "H-H": "Human-only",
        "AI-H": "AI assist: ideation",
        "H-AI": "AI assist: selection",
        "AI-AI": "AI assist: ideation + selection",
    }
    humor_by_group = collect_similarities_by_group(humor_groups, humor_embeddings)
    plot_distribution(
        humor_by_group,
        title="Salas & Hosanagar – Humor caption contest (by condition)",
        outfile=plots_dir / "humor_by_group.png",
        palette="tab10",
        label_map=humor_label_map,
        human_labels={"H-H", "Human-only"},
        draw_order=["H-H", "AI-H", "H-AI", "AI-AI"],
        style_map={"AI-H": ":", "H-AI": "--", "AI-AI": "-."},
    )

    # Humor modality with contest-balanced weighting (each contest contributes equally).
    humor_contests = pd.read_csv(base / "humor_embeddings.csv")["Contest"]
    humor_balanced_sims, humor_balanced_weights = collect_similarities_by_group_contest_balanced(
        humor_groups, humor_contests, humor_embeddings
    )
    plot_distribution(
        humor_balanced_sims,
        title="Salas & Hosanagar – Humor caption contest (contest-balanced)",
        outfile=plots_dir / "humor_by_group_contest_balanced.png",
        palette="tab10",
        label_map=humor_label_map,
        weights_by_group=humor_balanced_weights,
        human_labels={"H-H", "Human-only"},
        draw_order=["H-H", "AI-H", "H-AI", "AI-AI"],
        style_map={"AI-H": ":", "H-AI": "--", "AI-AI": "-."},
    )


if __name__ == "__main__":
    main()

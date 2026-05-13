"""2D dispersion plots of clustering results.

Projects embeddings to 2D with UMAP (when not already 2D) and writes a
seaborn scatter plot coloured by cluster label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

NOISE_LABEL = -1


def project_to_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    import umap

    n_samples = embeddings.shape[0]
    effective_neighbors = max(2, min(n_neighbors, n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def plot_clusters_2d(
    points_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    show_noise: bool = True,
) -> None:
    if points_2d.shape[0] != labels.shape[0]:
        raise ValueError(
            f"points_2d and labels must align: {points_2d.shape[0]} vs {labels.shape[0]}"
        )

    noise_mask = labels == NOISE_LABEL
    cluster_mask = ~noise_mask

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(9, 7))

    if show_noise and noise_mask.any():
        ax.scatter(
            points_2d[noise_mask, 0],
            points_2d[noise_mask, 1],
            c="lightgrey",
            s=10,
            alpha=0.4,
            label="noise",
            linewidths=0,
        )

    if cluster_mask.any():
        cluster_labels = labels[cluster_mask]
        unique = np.unique(cluster_labels)
        palette = sns.color_palette("husl", len(unique))
        sns.scatterplot(
            x=points_2d[cluster_mask, 0],
            y=points_2d[cluster_mask, 1],
            hue=cluster_labels,
            palette=palette,
            s=22,
            alpha=0.85,
            linewidth=0,
            ax=ax,
            legend="full" if len(unique) <= 20 else False,
        )
        if len(unique) > 20:
            ax.text(
                0.99, 0.01,
                f"{len(unique)} clusters",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9, color="dimgrey",
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=True, fontsize=8, title="cluster")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

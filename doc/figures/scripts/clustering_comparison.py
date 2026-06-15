"""Clustering comparison — 2x2 mosaic of algorithms on a synthetic distribution.

For the agrupamiento slide: the same 2D point cloud (two moons + two blobs of
different density, plus uniform noise) clustered by four representative
algorithms from the catalogue. Shows how the partitional baseline (K-medias)
splits the non-convex moons, while graph and density methods (espectral, DBSCAN,
HDBSCAN) recover them. Pure synthetic data, no benchmark inputs.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

from _style import save, setup

import matplotlib.pyplot as plt

RNG = 0
NOISE_COLOR = "0.7"


def make_data() -> np.ndarray:
    rng = np.random.default_rng(RNG)
    moons, _ = make_moons(n_samples=300, noise=0.06, random_state=RNG)
    moons = moons * [1.6, 1.6] + [0.0, 3.0]
    blob_a, _ = make_blobs(n_samples=120, centers=[[-4.0, -1.0]], cluster_std=0.35, random_state=RNG)
    blob_b, _ = make_blobs(n_samples=120, centers=[[4.0, -1.0]], cluster_std=0.9, random_state=RNG)
    noise = rng.uniform(low=[-6, -3.5], high=[6, 5.5], size=(40, 2))
    return np.vstack([moons, blob_a, blob_b, noise])


def main() -> None:
    setup()
    X = StandardScaler().fit_transform(make_data())

    algos = [
        ("K-medias", KMeans(n_clusters=4, n_init=10, random_state=RNG)),
        ("Espectral", SpectralClustering(n_clusters=4, affinity="nearest_neighbors", random_state=RNG)),
        ("DBSCAN", DBSCAN(eps=0.18, min_samples=8)),
        ("HDBSCAN", HDBSCAN(min_cluster_size=25)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.0))
    palette = plt.get_cmap("tab10")

    for ax, (name, model) in zip(axes.ravel(), algos):
        labels = model.fit_predict(X)
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            color = NOISE_COLOR if lbl == -1 else palette(lbl % 10)
            ax.scatter(X[mask, 0], X[mask, 1], s=6, color=color, linewidths=0)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    save(fig, "clustering_comparison")


if __name__ == "__main__":
    main()

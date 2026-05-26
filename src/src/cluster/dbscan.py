"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

Groups points that are closely packed together and marks as noise (-1) the points
that lie in low-density regions. The neighborhood radius eps is auto-detected from
the k-distance elbow when not provided.

Pros:
  - Does not require specifying the number of clusters.
  - Naturally identifies and isolates noise/outliers (label -1).
  - Discovers clusters of arbitrary shape, not limited to convex regions.

Cons:
  - Sensitive to eps and min_samples; wrong values produce one giant cluster or all noise.
  - Struggles with clusters of varying density.
  - High-dimensional embeddings cause the "curse of dimensionality": distances become
    uniform and density-based reasoning breaks down — dimensionality reduction beforehand
    is strongly recommended.
"""

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

from src.abstractions import ClusteringBase, ImageData
from src.cluster.utils import find_elbow


class DBSCANClusterer(ClusteringBase):
    is_deterministic = True

    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        eps = kwargs.get("eps")
        min_samples = kwargs.get("min_samples", 5)
        embeddings = np.array([img.embedding for img in images])

        if eps is None:
            if len(embeddings) <= min_samples:
                return [-1] * len(embeddings)

            neigh = NearestNeighbors(n_neighbors=min_samples)
            nbrs = neigh.fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)

            k_distances = np.sort(distances[:, min_samples - 1])
            elbow_idx = find_elbow(k_distances)
            # flat curve (uniform distances) → elbow returns 0 (minimum) which is
            # too tight and causes all-noise. fall back to 90th percentile.
            if elbow_idx == 0:
                eps = float(np.percentile(k_distances, 90))
            else:
                eps = float(k_distances[elbow_idx])

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        return list(dbscan.labels_.tolist())  # type: ignore

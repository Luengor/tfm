"""
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

Extends DBSCAN by building a full cluster hierarchy across all density levels and
then extracting the most stable clusters. Unlike DBSCAN it handles clusters with
varying density and requires only min_cluster_size, making tuning easier.

Pros:
  - Handles clusters of varying density — a major limitation of plain DBSCAN.
  - Only one intuitive parameter (min_cluster_size) is usually needed.
  - Robust to noise; assigns outliers the label -1.
  - Generally outperforms DBSCAN in practice on embedding spaces.

Cons:
  - More memory-intensive than DBSCAN due to the hierarchy construction.
  - Still susceptible to the curse of dimensionality in very high-dimensional spaces.
  - Sensitive to data perturbations: small changes in input can shift cluster boundaries.
"""

from sklearn.cluster import HDBSCAN
import numpy as np

from src.abstractions import ClusteringBase, ImageData


class HDBSCANClusterer(ClusteringBase):
    is_deterministic = True

    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        min_cluster_size = kwargs.get("min_cluster_size", 5)
        max_cluster_size = kwargs.get("max_cluster_size", None)
        embeddings = np.array([img.embedding for img in images])
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(embeddings)
        return list(hdbscan.labels_.tolist())  # type: ignore

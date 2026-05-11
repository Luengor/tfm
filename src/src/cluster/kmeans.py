"""
K-Means clustering.

Partitions data into k spherical clusters by iteratively assigning each point to the
nearest centroid and recomputing centroids. When k is not given it is chosen automatically
via the inertia elbow method.

Pros:
  - Fast and scalable; works well on large datasets.
  - Deterministic result (with fixed random_state).
  - Intuitive and easy to interpret.

Cons:
  - Requires specifying k (or auto-detection via elbow, which can misfire).
  - Assumes convex, roughly equal-sized, spherical clusters — poor fit for irregular shapes.
  - Sensitive to outliers, which pull centroids away from the cluster core.
  - Distance metric is Euclidean; cosine similarity between embeddings is not captured.
"""

from sklearn.cluster import KMeans
import numpy as np

from src.abstractions import ClusteringBase, ImageData
from src.cluster.utils import find_elbow


class KMeansClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters")
        embeddings = np.array([img.embedding for img in images])

        if n_clusters is None:
            max_k = min(len(embeddings), kwargs.get("max_clusters", 20))
            if max_k < 2:
                n_clusters = 1
            else:
                ks = range(2, max_k + 1)
                inertias = [
                    KMeans(n_clusters=k, random_state=0, n_init="auto").fit(embeddings).inertia_
                    for k in ks
                ]
                n_clusters = ks[find_elbow(np.array(inertias))]

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings)
        return list(kmeans.labels_.tolist())  # type: ignore

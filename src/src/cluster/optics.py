"""
OPTICS (Ordering Points To Identify the Clustering Structure).

Produces a reachability-distance ordering of points that encodes the density structure
at all eps values simultaneously. Clusters are then extracted from the reachability plot,
making it an extension of DBSCAN that handles variable-density clusters without fixing eps.

Pros:
  - No need to choose eps; max_eps just limits the search radius.
  - Handles clusters of widely varying densities in a single pass.
  - Produces a rich dendrogram-like ordering useful for visualisation.

Cons:
  - Slower than DBSCAN and HDBSCAN on large datasets.
  - Default cosine metric works well for embeddings but increases computation cost.
  - Cluster extraction heuristics can be sensitive to xi / min_samples.
"""

from sklearn.cluster import OPTICS
import numpy as np

from src.abstractions import ClusteringBase, ImageData


class OPTICSClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        min_samples = kwargs.get("min_samples", 5)
        max_eps = kwargs.get("max_eps", np.inf)
        metric = kwargs.get("metric", "cosine")
        embeddings = np.array([img.embedding for img in images])
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric).fit(embeddings)
        return list(optics.labels_.tolist())  # type: ignore

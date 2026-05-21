"""
Spectral Clustering.

Constructs a graph where nodes are data points and edges encode similarity, then
clusters the low-dimensional eigenvectors of the graph Laplacian. This lets it find
clusters defined by connectivity rather than compactness.

Pros:
  - Finds non-convex, manifold-shaped clusters that distance-based methods miss.
  - Works well when clusters are defined by local connectivity (e.g. ring shapes).
  - The nearest-neighbors affinity adapts naturally to embedding spaces.

Cons:
  - Requires specifying n_clusters.
  - O(n³) eigen-decomposition makes it impractical for datasets larger than a few thousand points.
  - Sensitive to the choice of affinity and n_neighbors; wrong settings produce degenerate results.
  - Non-deterministic: random graph construction means results may vary between runs.
"""

from sklearn.cluster import SpectralClustering
import numpy as np

from src.abstractions import ClusteringBase, ImageData


class SpectralClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters", 5)
        affinity = kwargs.get("affinity", "nearest_neighbors")
        random_state = kwargs.get("random_state", 0)
        embeddings = np.array([img.embedding for img in images])
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state).fit(embeddings)
        return list(spectral.labels_.tolist())  # type: ignore

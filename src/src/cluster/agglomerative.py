"""
Agglomerative (Hierarchical) Clustering.

Bottom-up approach: starts with each point as its own cluster and iteratively merges
the two closest clusters until the target number of clusters is reached. The linkage
criterion (ward, complete, average, single) controls how inter-cluster distance is measured.

Pros:
  - No assumptions about cluster shape; can discover elongated or irregular clusters
    with appropriate linkage (e.g. complete, average).
  - Does not require a distance threshold — the tree can be cut at any level.
  - Deterministic; produces the same result every run.

Cons:
  - Requires specifying n_clusters up front (no auto-detection implemented here).
  - O(n² log n) time and O(n²) memory — does not scale well beyond ~10k points.
  - Ward linkage (default) minimises within-cluster variance and implicitly favours
    convex, equally-sized clusters similar to K-Means.
"""

from sklearn.cluster import AgglomerativeClustering
import numpy as np

from src.abstractions import ClusteringBase, ImageData


class AgglomerativeClusterer(ClusteringBase):
    is_deterministic = True

    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters", 5)
        linkage = kwargs.get("linkage", "ward")
        embeddings = np.array([img.embedding for img in images])
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(embeddings)
        return list(agg.labels_.tolist())  # type: ignore

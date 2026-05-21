"""
Affinity Propagation clustering.

Points exchange "responsibility" and "availability" messages until a set of exemplars
(cluster centres chosen from the data itself) emerges. The number of clusters is
determined automatically by the preference parameter — higher preference → more clusters.

Pros:
  - Automatically determines the number of clusters from the data.
  - Cluster centres are real data points (exemplars), which aids interpretability.
  - Works well when the natural number of clusters is unknown and cannot be estimated.

Cons:
  - O(n²) memory and O(n² · iterations) time — infeasible for more than a few thousand points.
  - Very sensitive to the preference parameter; small changes can dramatically alter results.
  - Often produces too many small clusters on embedding data unless preference is tuned carefully.
  - Convergence is not guaranteed; may require more iterations on noisy data.
"""

from sklearn.cluster import AffinityPropagation
import numpy as np

from src.abstractions import ClusteringBase, ImageData


class AffinityPropagationClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        damping = kwargs.get("damping", 0.5)
        preference = kwargs.get("preference", None)
        random_state = kwargs.get("random_state", 0)
        embeddings = np.array([img.embedding for img in images])
        af = AffinityPropagation(damping=damping, preference=preference, random_state=random_state).fit(embeddings)
        return list(af.labels_.tolist())  # type: ignore

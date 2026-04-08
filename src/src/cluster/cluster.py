from sklearn.cluster import KMeans
import numpy as np

from src.abstractions import ClusteringBase, ImageData

class KMeansClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters", 5)
        embeddings = np.array([img.embedding for img in images])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        return list(kmeans.labels_) # type: ignore


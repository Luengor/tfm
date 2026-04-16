from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, OPTICS
import numpy as np

from src.abstractions import ClusteringBase, ImageData

class KMeansClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters", 5)
        embeddings = np.array([img.embedding for img in images])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        return list(kmeans.labels_.tolist()) # type: ignore

class DBSCANClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        embeddings = np.array([img.embedding for img in images])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        return list(dbscan.labels_.tolist()) # type: ignore

class HDBSCANClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        min_cluster_size = kwargs.get("min_cluster_size", 5)
        max_cluster_size = kwargs.get("max_cluster_size", None)
        embeddings = np.array([img.embedding for img in images])
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(embeddings)
        return list(hdbscan.labels_.tolist()) # type: ignore

class OPTICSClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        min_samples = kwargs.get("min_samples", 5)
        max_eps = kwargs.get("max_eps", np.inf)
        metric = kwargs.get("metric", "cosine")
        embeddings = np.array([img.embedding for img in images])
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric).fit(embeddings)
        return list(optics.labels_.tolist()) # type: ignore



from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import numpy as np

from src.abstractions import ClusteringBase, ImageData

def find_elbow(y: np.ndarray) -> int:
    """
    Finds the elbow point in a curve using the perpendicular distance method.
    Returns the index of the elbow point.
    """
    n = len(y)
    if n < 3:
        return 0
    x = np.arange(n)
    
    p1 = np.array([0, y[0]])
    p2 = np.array([n - 1, y[-1]])
    
    line_vec = p2 - p1
    norm = np.sqrt(np.sum(line_vec**2))
    if norm == 0:
        return 0
        
    line_vec_norm = line_vec / norm
    
    p1_to_p = np.column_stack((x, y)) - p1
    proj = np.outer(np.dot(p1_to_p, line_vec_norm), line_vec_norm)
    dist_to_line = np.sqrt(np.sum((p1_to_p - proj)**2, axis=1))
    
    return int(np.argmax(dist_to_line))

class KMeansClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters")
        embeddings = np.array([img.embedding for img in images])
        
        if n_clusters is None:
            # Search for optimal k using inertia elbow
            max_k = min(len(embeddings), kwargs.get("max_clusters", 20))
            if max_k < 2:
                n_clusters = 1
            else:
                ks = range(1, max_k + 1)
                inertias = [
                    KMeans(n_clusters=k, random_state=0, n_init="auto").fit(embeddings).inertia_
                    for k in ks
                ]
                n_clusters = ks[find_elbow(np.array(inertias))]

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings)
        return list(kmeans.labels_.tolist()) # type: ignore

class DBSCANClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        eps = kwargs.get("eps")
        min_samples = kwargs.get("min_samples", 5)
        embeddings = np.array([img.embedding for img in images])

        if eps is None:
            if len(embeddings) <= min_samples:
                return [-1] * len(embeddings)

            # Find the optimal eps using the elbow method on k-distances
            neigh = NearestNeighbors(n_neighbors=min_samples)
            nbrs = neigh.fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)
            
            # Sort distances to the k-th nearest neighbor
            k_distances = np.sort(distances[:, min_samples - 1])
            eps = float(k_distances[find_elbow(k_distances)])

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

class AgglomerativeClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters", 5)
        linkage = kwargs.get("linkage", "ward")
        embeddings = np.array([img.embedding for img in images])
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(embeddings)
        return list(agg.labels_.tolist()) # type: ignore

class SpectralClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_clusters = kwargs.get("n_clusters", 5)
        affinity = kwargs.get("affinity", "nearest_neighbors")
        embeddings = np.array([img.embedding for img in images])
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=0).fit(embeddings)
        return list(spectral.labels_.tolist()) # type: ignore

class GMMClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        n_components = kwargs.get("n_clusters")
        covariance_type = kwargs.get("covariance_type", "full")
        embeddings = np.array([img.embedding for img in images])
        
        if n_components is None:
            # Search for optimal k by minimizing BIC
            max_k = min(len(embeddings), kwargs.get("max_clusters", 20))
            if max_k < 2:
                n_components = 1
            else:
                ks = range(1, max_k + 1)
                bics = [
                    GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=0).fit(embeddings).bic(embeddings)
                    for k in ks
                ]
                n_components = ks[int(np.argmin(bics))]

        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=0).fit(embeddings)
        labels = gmm.predict(embeddings)
        return list(labels.tolist())

class AffinityPropagationClusterer(ClusteringBase):
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        damping = kwargs.get("damping", 0.5)
        preference = kwargs.get("preference", None)
        embeddings = np.array([img.embedding for img in images])
        af = AffinityPropagation(damping=damping, preference=preference, random_state=0).fit(embeddings)
        return list(af.labels_.tolist()) # type: ignore



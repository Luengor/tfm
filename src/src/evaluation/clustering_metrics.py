import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from src.evaluation.models import ClusteringQualityMetrics


def calculate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> ClusteringQualityMetrics:
    """
    Calculates unsupervised clustering metrics (Silhouette Score and Calinski-Harabasz Index).
    
    Handle edge cases where metrics are undefined (e.g., single cluster or only noise).
    """
    # Filter out noise points (-1) for silhouette and CH scores if needed, 
    # but usually scikit-learn metrics handle them as a separate cluster.
    # However, silhouette_score requires at least 2 clusters (excluding noise if not treated as cluster).
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Silhouette score requires 2 <= n_labels <= n_samples - 1
    s_score = None
    if 1 < n_clusters < len(embeddings):
        try:
            s_score = float(silhouette_score(embeddings, labels))
        except Exception:
            s_score = None
            
    # Calinski-Harabasz score requires at least 2 clusters
    ch_score = None
    if n_clusters > 1:
        try:
            ch_score = float(calinski_harabasz_score(embeddings, labels))
        except Exception:
            ch_score = None
            
    return ClusteringQualityMetrics(
        silhouette_score=s_score,
        calinski_harabasz_score=ch_score
    )

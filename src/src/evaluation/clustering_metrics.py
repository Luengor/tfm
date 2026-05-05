import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.evaluation.models import ClusteringQualityMetrics


def calculate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> ClusteringQualityMetrics:
    """
    Calculates unsupervised clustering metrics.
    
    Includes:
    - Silhouette Score (Density/Separation)
    - Calinski-Harabasz Index (Variance Ratio)
    - Davies-Bouldin Index (Cluster Similarity)
    - Noise Ratio (Percentage of unclustered points)
    - Cluster Size CV (Coefficient of Variation, measure of balance)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_samples = len(embeddings)
    
    # Filter out noise for standard sklearn metrics if necessary
    # Silhouette and CH usually treat noise as a separate cluster if included.
    # We will compute them with all labels provided.
    
    s_score = None
    if 1 < len(unique_labels) < n_samples:
        try:
            s_score = float(silhouette_score(embeddings, labels))
        except Exception:
            s_score = None
            
    ch_score = None
    if len(unique_labels) > 1:
        try:
            ch_score = float(calinski_harabasz_score(embeddings, labels))
        except Exception:
            ch_score = None

    db_score = None
    if n_clusters > 1:
        try:
            # Davies-Bouldin works best on non-noise labels
            mask = labels != -1
            if mask.sum() > n_clusters:
                db_score = float(davies_bouldin_score(embeddings[mask], labels[mask]))
        except Exception:
            db_score = None

    # Noise ratio
    noise_ratio = float(np.sum(labels == -1) / n_samples) if n_samples > 0 else None
    
    # Cluster balance (CV of sizes)
    cluster_size_cv = None
    if n_clusters > 0:
        counts = [np.sum(labels == label) for label in unique_labels if label != -1]
        if len(counts) > 1:
            mean_size = np.mean(counts)
            std_size = np.std(counts)
            cluster_size_cv = float(std_size / mean_size) if mean_size > 0 else 0.0
        else:
            cluster_size_cv = 0.0

    return ClusteringQualityMetrics(
        silhouette_score=s_score,
        calinski_harabasz_score=ch_score,
        davies_bouldin_score=db_score,
        noise_ratio=noise_ratio,
        cluster_size_cv=cluster_size_cv
    )

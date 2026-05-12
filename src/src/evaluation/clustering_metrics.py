import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix

from src.evaluation.models import ClusteringQualityMetrics, ExtrinsicMetrics


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
    if n_clusters > 1:
        try:
            mask = labels != -1
            if mask.sum() > n_clusters:
                ch_score = float(calinski_harabasz_score(embeddings[mask], labels[mask]))
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


def calculate_extrinsic_metrics(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
) -> ExtrinsicMetrics:
    """
    Calculates supervised (extrinsic) clustering metrics against ground-truth labels.

    Includes ARI, NMI, and pairwise F1 (precision/recall over same-cluster vs
    same-class pairs). Noise points (label == -1) are treated as their own cluster.
    """
    n = len(labels_pred)
    if n == 0 or n != len(labels_true):
        return ExtrinsicMetrics(ari=None, nmi=None, pairwise_f1=None, n_matched=n)

    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(normalized_mutual_info_score(labels_true, labels_pred))

    pcm = pair_confusion_matrix(labels_true, labels_pred)
    fp = float(pcm[0, 1])
    fn = float(pcm[1, 0])
    tp = float(pcm[1, 1])
    if tp + fp == 0 or tp + fn == 0:
        pairwise_f1: float | None = None
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        pairwise_f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ExtrinsicMetrics(
        ari=ari,
        nmi=nmi,
        pairwise_f1=pairwise_f1,
        n_matched=n,
        n_classes=int(len(np.unique(labels_true))),
    )

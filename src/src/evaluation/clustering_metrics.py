import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix

from src.evaluation.models import (
    ClusteringQualityMetrics,
    ExtrinsicMetrics,
    SimilaritySearchExtrinsicMetrics,
)


def calculate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> ClusteringQualityMetrics:
    """
    Calculates unsupervised clustering metrics.
    
    Includes:
    - Silhouette Score (Density/Separation)
    - Calinski-Harabasz Index (Variance Ratio)
    - Davies-Bouldin Index (Cluster Similarity)
    - Noise Ratio (Percentage of unclustered points)
    - Cluster Size CV (Coefficient of Variation, measure of balance)

    Embeddings are L2-normalized before the distance-based metrics
    (silhouette / Calinski-Harabasz / Davies-Bouldin) are computed. Some
    embedding models L2-normalize their output (CLIP, the graffiti heads,
    mobilenet_v3_normalized) and some do not (ResNet50, VGG16, InceptionV3,
    plain MobileNetV3/DINOv2, YOLO); without normalization the Euclidean
    distances these metrics use live on per-model magnitude scales, so the
    scores are not comparable across embedding families. On unit vectors
    Euclidean distance is a monotone function of cosine distance, which is the
    geometry used everywhere else in the project (similarity search defaults to
    cos_distance=True). Calinski-Harabasz and Davies-Bouldin have no cosine
    option, so normalizing the inputs is the only way to make all three
    consistent. Normalization rescales but does not reduce dimensionality — the
    metrics are still computed on the original (unreduced) representation.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_samples = len(embeddings)

    # Noise points (label == -1) are excluded from silhouette, Calinski-Harabasz
    # and Davies-Bouldin so a clustering that rejects many points is not
    # rewarded with an inflated score over the few survivors. Noise is reported
    # separately via noise_ratio.

    s_score = None
    noise_mask = labels != -1
    s_labels = labels[noise_mask]
    n_s_clusters = len(np.unique(s_labels))
    if n_s_clusters > 1 and noise_mask.sum() > n_s_clusters:
        try:
            s_score = float(silhouette_score(embeddings[noise_mask], s_labels))
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


def calculate_similarity_search_extrinsic(
    embeddings: np.ndarray,
    labels_true: np.ndarray,
    top_k: int,
    cos_distance: bool,
) -> SimilaritySearchExtrinsicMetrics:
    """
    Supervised similarity-search metrics computed via brute-force kNN over the
    provided (labeled-only) embedding subset. A neighbor is considered relevant
    when it shares the query's class.

    Returns Precision@k, Recall@k (denominator clipped to min(k, class_size-1)),
    mAP@k and MRR. k is clipped to (n-1) to allow self-exclusion.
    """
    n = len(embeddings)
    n_classes = int(len(np.unique(labels_true)))
    if n < 2 or top_k < 1:
        return SimilaritySearchExtrinsicMetrics(
            precision_at_k=None,
            recall_at_k=None,
            map_at_k=None,
            mrr=None,
            k_effective=0,
            n_matched=n,
            n_classes=n_classes,
        )

    k = min(top_k, n - 1)

    if cos_distance:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.clip(norms, 1e-12, None)
        sim = normed @ normed.T
        dist = 1.0 - sim
    else:
        # squared euclidean preserves ordering, cheaper
        sq = np.sum(embeddings ** 2, axis=1)
        dist = sq[:, None] + sq[None, :] - 2.0 * (embeddings @ embeddings.T)

    np.fill_diagonal(dist, np.inf)

    # Partial sort: top-k smallest distances per row
    nn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    nn_sorted_order = np.argsort(dist[row_idx, nn_idx], axis=1)
    nn_idx = nn_idx[row_idx, nn_sorted_order]

    neighbor_labels = labels_true[nn_idx]
    relevance = (neighbor_labels == labels_true[:, None]).astype(np.int32)  # (n, k)

    # Precision@k per-query then averaged
    precision_at_k = float(relevance.mean())

    # mAP@k
    positions = np.arange(1, k + 1)
    cum_hits = np.cumsum(relevance, axis=1)
    precision_at_i = cum_hits / positions
    relevant_counts = relevance.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap_per_query = np.where(
            relevant_counts > 0,
            (precision_at_i * relevance).sum(axis=1) / np.clip(relevant_counts, 1, None),
            0.0,
        )
    map_at_k = float(ap_per_query.mean())

    # MRR
    first_hit_idx = np.argmax(relevance, axis=1)
    has_hit = relevance.max(axis=1) > 0
    rr = np.where(has_hit, 1.0 / (first_hit_idx + 1), 0.0)
    mrr = float(rr.mean())

    # Recall@k with denominator min(k, class_size - 1)
    _, class_counts = np.unique(labels_true, return_counts=True)
    label_to_count = dict(zip(np.unique(labels_true).tolist(), class_counts.tolist()))
    relevant_totals = np.array(
        [min(k, label_to_count[label] - 1) for label in labels_true.tolist()]
    )
    valid = relevant_totals > 0
    if valid.any():
        recall_at_k = float(
            (relevance.sum(axis=1)[valid] / relevant_totals[valid]).mean()
        )
    else:
        recall_at_k = None

    return SimilaritySearchExtrinsicMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        map_at_k=map_at_k,
        mrr=mrr,
        k_effective=k,
        n_matched=n,
        n_classes=n_classes,
    )

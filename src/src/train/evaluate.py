import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from src.embedding.embeddings import EmbeddingModelNames, get_model


def evaluate_model(model_name: EmbeddingModelNames, crops_dir: str, min_samples: int = 2) -> dict:
    model = get_model(model_name)

    all_classes = sorted([d for d in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, d))])
    # Only evaluate classes with enough samples for LOO-KNN and silhouette
    classes = [
        cls for cls in all_classes
        if len(os.listdir(os.path.join(crops_dir, cls))) >= min_samples
    ]
    skipped = len(all_classes) - len(classes)
    print(f"  Classes: {len(classes)} with ≥{min_samples} samples ({skipped} singletons skipped)")

    embeddings = []
    labels = []

    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(crops_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                emb = model.gen_embedding(img)
                embeddings.append(emb)
                labels.append(label_idx)
            except Exception as e:
                print(f"  Error {img_path}: {e}")

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    embeddings_norm = F.normalize(torch.from_numpy(embeddings), p=2, dim=1).numpy()

    s_score = silhouette_score(embeddings_norm, labels, metric='cosine')

    # Leave-One-Out 1-NN accuracy
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    correct = 0
    total = len(embeddings_norm)
    for i in range(total):
        mask = np.ones(total, dtype=bool)
        mask[i] = False
        knn.fit(embeddings_norm[mask], labels[mask])
        if knn.predict(embeddings_norm[[i]])[0] == labels[i]:
            correct += 1
    accuracy = correct / total

    kmeans = KMeans(n_clusters=len(classes), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_norm)
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)

    return {"silhouette": s_score, "accuracy_1nn": accuracy, "ari": ari, "nmi": nmi}


def main():
    parser = argparse.ArgumentParser(description="Evaluate graffiti author embedding models")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to crops directory (one subfolder per author)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum crops per class for evaluation (default: 2)",
    )
    args = parser.parse_args()

    models_to_compare = [
        EmbeddingModelNames.DINOV2_VITS14,
        EmbeddingModelNames.DINOV2_GRAFFITI_AUTHOR_HEAD,
    ]

    results = {}
    for m_name in models_to_compare:
        print(f"\nEvaluating {m_name.value}...")
        results[m_name] = evaluate_model(m_name, args.dataset, min_samples=args.min_samples)

    print("\nResults:")
    header = f"{'Metric':<20} | {'DINO Base':<12} | {'DINO Author Head':<16}"
    print(header)
    print("-" * len(header))
    for metric in ["silhouette", "accuracy_1nn", "ari", "nmi"]:
        d_base = results[EmbeddingModelNames.DINOV2_VITS14][metric]
        d_head = results[EmbeddingModelNames.DINOV2_GRAFFITI_AUTHOR_HEAD][metric]
        print(f"{metric:<20} | {d_base:<12.4f} | {d_head:<16.4f}")


if __name__ == "__main__":
    main()

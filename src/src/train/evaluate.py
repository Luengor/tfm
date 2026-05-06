import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import torch.nn.functional as F

from src.embedding.embeddings import get_model, EmbeddingModelNames

def evaluate_model(model_name, crops_dir):
    model = get_model(model_name)
    
    classes = sorted([d for d in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, d))])
    
    embeddings = []
    labels = []
    
    print(f"Generating embeddings for {model_name}...")
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
                print(f"Error processing {img_path}: {e}")
                
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Normalize embeddings for cosine distance
    embeddings_norm = F.normalize(torch.from_numpy(embeddings), p=2, dim=1).numpy()
    
    # Unsupervised: Silhouette
    s_score = silhouette_score(embeddings_norm, labels, metric='cosine')
    
    # Supervised: 1-NN Accuracy using Leave-One-Out (since dataset is small)
    loo = LeaveOneOut()
    correct = 0
    total = 0
    
    # We use cosine distance for KNN
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    
    for train_index, test_index in loo.split(embeddings_norm):
        X_train, X_test = embeddings_norm[train_index], embeddings_norm[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        if pred[0] == y_test[0]:
            correct += 1
        total += 1
        
    accuracy = correct / total
    
    # Clustering metrics (using KMeans to see how well they cluster)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=len(classes), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_norm)
    
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    
    return {
        "silhouette": s_score,
        "accuracy_1nn": accuracy,
        "ari": ari,
        "nmi": nmi
    }

if __name__ == "__main__":
    CROPS_DIR = os.path.join("dataset", "crops")
    
    models_to_compare = [
        EmbeddingModelNames.DINOV2_VITS14,
        EmbeddingModelNames.DINOV2_GRAFFITI_HEAD,
        EmbeddingModelNames.MOBILENET_V3,
        EmbeddingModelNames.MOBILENET_V3_GRAFFITI_HEAD
    ]
    
    results = {}
    for m_name in models_to_compare:
        results[m_name] = evaluate_model(m_name, CROPS_DIR)
        
    print("\nComparison Results:")
    header = f"{'Metric':<20} | {'DINO Base':<10} | {'DINO Head':<10} | {'MBNet Base':<10} | {'MBNet Head':<10}"
    print(header)
    print("-" * len(header))
    for metric in ["silhouette", "accuracy_1nn", "ari", "nmi"]:
        d_base = results[EmbeddingModelNames.DINOV2_VITS14][metric]
        d_head = results[EmbeddingModelNames.DINOV2_GRAFFITI_HEAD][metric]
        m_base = results[EmbeddingModelNames.MOBILENET_V3][metric]
        m_head = results[EmbeddingModelNames.MOBILENET_V3_GRAFFITI_HEAD][metric]
        print(f"{metric:<20} | {d_base:<10.4f} | {d_head:<10.4f} | {m_base:<10.4f} | {m_head:<10.4f}")

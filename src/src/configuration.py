from src.abstractions import StorageBase, EmbeddingBase, ClusteringBase, ImageData, ReductionBase
from PIL import Image as PILImage
import numpy as np
from enum import Enum

class DistanceMethod(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class Configuration:
    def __init__(self, storage: StorageBase, embedding: EmbeddingBase, clustering: ClusteringBase, reduction: ReductionBase):
        self.storage = storage
        self.embedding = embedding
        self.clustering = clustering
        self.reduction = reduction

    def save_image(self, filename: str) -> ImageData:
        # Check if the image is already in storage
        if self.storage.has(filename):
            return self.storage.load(filename)

        # Get the image embedding
        image = PILImage.open(filename).convert("RGB")
        emb = self.embedding.gen_embedding(image)

        # Save the image data
        data = ImageData(filename=filename, embedding=emb)
        self.storage.save(data)

        return data

    def get_by_distance(self, filename: str, max_images: int = 10, cos_distance: bool = True) -> list[ImageData]:
        # Load the target image
        target_data = self.storage.load(filename)

        # Get similar images by distance
        similar_images = self.storage.get_by_distance(target_data.embedding, max_images=max_images, cos_distance=cos_distance)

        return similar_images

    def cluster_images(self, **kwargs) -> list[list[ImageData]]:
        # Get all images from storage
        images = self.storage.get_all_images()
        
        if not images:
            return []

        embeddings = [img.embedding for img in images]
        reduced_embeddings = self.reduction.reduce(embeddings)
        # Create temporary ImageData with reduced embeddings for clustering
        clustering_images = [
            ImageData(filename=img.filename, embedding=emb) 
            for img, emb in zip(images, reduced_embeddings)
        ]

        # Cluster the images
        cluster_index = self.clustering.cluster(clustering_images, **kwargs)
        n_clusters = max(cluster_index) + 1 if cluster_index else 0

        clusters = [[] for _ in range(n_clusters)]
        for i, cluster in enumerate(cluster_index):
            if cluster == -1:
                clusters.append([images[i]])
            else:
                clusters[cluster].append(images[i])

        return clusters

    def get_distance(self, filename1: str, filename2: str, distance_method: DistanceMethod = DistanceMethod.COSINE) -> float:
        data1 = self.storage.load(filename1)
        data2 = self.storage.load(filename2)

        if distance_method == DistanceMethod.COSINE:
            return 1 - np.dot(data1.embedding, data2.embedding) / (np.linalg.norm(data1.embedding) * np.linalg.norm(data2.embedding))
        elif distance_method == DistanceMethod.EUCLIDEAN:
            # Euclidean distance
            return float(np.linalg.norm(np.array(data1.embedding) - np.array(data2.embedding)))

def save_folder(folder: str, config: Configuration):
    for filename in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
        full_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(full_path):
                config.save_image(full_path)
            else:
                save_folder(full_path, config)
        except Exception as e:
            print(f"Error processing {full_path}: {e}")

if __name__ == "__main__":
    from src.storage.postgresql import PostgreSQLStorage
    from src.embedding.embeddings import EmbeddingModelNames, get_model
    from src.embedding.custom import CustomEmbeddingModel
    from src.cluster.cluster import OPTICSClusterer 
    from sys import argv
    import os
    from tqdm import tqdm

    from src.reduction import IdentityReduction
    storage = PostgreSQLStorage("postgresql://postgres:changethis@localhost:54321/postgres")

    print("Creating configuration...")
    config = Configuration(
            storage,
            get_model(EmbeddingModelNames.YOLOs),
            OPTICSClusterer(),
            IdentityReduction()
    )

    if len(argv) < 2:
        print("Clustering images...")
        clusters = config.cluster_images(metric="cosine", min_samples=2)
        for cluster in clusters:
            print(f"Cluster with {len(cluster)} images:")
            for image in cluster[:5]:  # Print first 5 images in the cluster
                print(f"  - {image.filename}")

    # Check if the argv[1] is a folder or file 
    if os.path.isdir(argv[1]):
        print("Adding images...")
        save_folder(argv[1], config)

    else:
        print("\nFinding similar images...")
        print('\n'.join(image.filename for image in config.get_by_distance(argv[1])[:10]))

    for filename in tqdm(argv[1:]):
        try:
            config.save_image(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")



from src.abstractions import StorageBase, EmbeddingBase, ClusteringBase, ImageData
from PIL import Image as PILImage
import numpy as np
from enum import Enum

class DistanceMethod(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class Configuration:
    def __init__(self, storage: StorageBase, embedding: EmbeddingBase, clustering: ClusteringBase):
        self.storage = storage
        self.embedding = embedding
        self.clustering = clustering

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

    def get_by_distance(self, filename: str, cos_distance: bool = True) -> list[ImageData]:
        # Load the target image
        target_data = self.storage.load(filename)

        # Get similar images by distance
        similar_images = self.storage.get_by_distance(target_data.embedding, cos_distance=cos_distance)

        return similar_images

    def cluster_images(self, **kwargs) -> list[list[ImageData]]:
        # Get all images from storage
        images = self.storage.get_all_images()

        # Cluster the images
        cluster_index = self.clustering.cluster(images, **kwargs)
        n_clusters = max(cluster_index) + 1

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

if __name__ == "__main__":
    from src.storage.postgresql import PostgreSQLStorage
    from src.embedding.embeddings import EmbeddingModelNames, get_model
    from src.embedding.custom import CustomEmbeddingModel
    from src.cluster.cluster import OPTICSClusterer 
    from sys import argv
    from tqdm import tqdm

    storage = PostgreSQLStorage("postgresql://postgres:changethis@localhost:54321/postgres")

    print("Creating configuration...")
    config = Configuration(
            storage,
            get_model(EmbeddingModelNames.YOLOm),
            OPTICSClusterer()
    )

    print("Adding images...")
    for filename in tqdm(argv[1:]):
        try:
            config.save_image(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Clustering images...")
    clusters = config.cluster_images(metric="cosine", min_samples=2)
    for cluster in clusters:
        print(f"Cluster with {len(cluster)} images:")
        for image in cluster[:5]:  # Print first 5 images in the cluster
            print(f"  - {image.filename}")

    if len(argv) <= 1:
        exit(0)

    print("\nFinding similar images...")
    print('\n'.join(image.filename for image in config.get_by_distance(argv[1])[:10]))


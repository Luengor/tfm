from src.abstractions import StorageBase, EmbeddingBase, ClusteringBase, ImageData, ReductionBase, SegmenterBase, BoundingBox
from PIL import Image as PILImage
from enum import Enum
import os
from tqdm import tqdm

class DistanceMethod(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class Configuration:
    def __init__(self, storage: StorageBase, embedding: EmbeddingBase, clustering: ClusteringBase, reduction: ReductionBase, segmenter: SegmenterBase):
        self.storage = storage
        self.embedding = embedding
        self.clustering = clustering
        self.reduction = reduction
        self.segmenter = segmenter

    def save_image(self, filename: str) -> list[ImageData]:
        # Check if the image is already in storage
        if self.storage.has(filename):
            return self.storage.load(filename)

        # Get the image embedding
        image = PILImage.open(filename).convert("RGB")
        
        # Segment the image to find the graffiti
        boxes = self.segmenter.segment(image)
        
        # If no boxes are detected, use the full image
        if not boxes:
            boxes = [BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=1.0)]

        # Generate embeddings for each box and save them
        images = []
        for box in boxes:
            # Scale normalized coordinates back to image dimensions for cropping
            left = box.x1 * image.width
            top = box.y1 * image.height
            right = box.x2 * image.width
            bottom = box.y2 * image.height
            
            crop = image.crop((left, top, right, bottom))
            emb = self.embedding.gen_embedding(crop)

            data = ImageData(filename=filename, bbox=box, embedding=emb)
            self.storage.save(data)
            images.append(data)

        return images 

    def get_by_distance(self, image: ImageData|str, max_images: int = 10, cos_distance: bool = True) -> list[ImageData]:
        # If the image is given as a filename, load it and take the first embedding 
        if isinstance(image, str):
            images = self.storage.load(image)
            image_data = images[0]
        else:
            image_data = image

        # Get similar images by distance
        similar_images = self.storage.get_by_distance(image_data.embedding, max_images=max_images, cos_distance=cos_distance)

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
    from src.cluster.cluster import OPTICSClusterer 
    from src.embedding.segmenters import IdentitySegmenter
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
            IdentityReduction(),
            IdentitySegmenter()
    )

    if len(argv) < 2:
        print("Clustering images...")
        clusters = config.cluster_images(metric="cosine", min_samples=2)
        for cluster in clusters:
            print(f"Cluster with {len(cluster)} images:")
            for image in cluster[:5]:  # Print first 5 images in the cluster
                print(f"  - {image.filename}")
        sys.exit(0)

    # Check if the argv[1] is a folder or file
    if os.path.isdir(argv[1]):
        print("Adding images...")
        save_folder(argv[1], config)

    for filename in tqdm(argv[1:]):
        try:
            config.save_image(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")



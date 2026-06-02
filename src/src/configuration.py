from src.abstractions import StorageBase, EmbeddingBase, ClusteringBase, ImageData, ReductionBase, SegmenterBase
from PIL import Image as PILImage, ImageFile
from enum import Enum

ImageFile.LOAD_TRUNCATED_IMAGES = True
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

        # Segment the image to find the graffiti. Fallback policy on empty
        # detection lives in the segmenter (see YoloSegmenter.allow_full_image_fallback);
        # an empty return here means the image is intentionally skipped.
        boxes = self.segmenter.segment(image)

        return self._embed_and_save(filename, image, boxes)

    def _embed_and_save(self, filename: str, image, boxes) -> list[ImageData]:
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

    def save_images_batch(self, filenames: list[str]) -> list[list[ImageData]]:
        """Single-batch ingest. The segmenter runs once on the whole batch when
        it supports it. Filenames already in storage are returned from cache
        without re-opening the image. Order preserved. Caller controls batch
        size (typically via ``segmenter.batch_size``)."""
        results: list[list[ImageData] | None] = [None] * len(filenames)

        pending_idx: list[int] = []
        pending_images = []
        for i, fn in enumerate(filenames):
            if self.storage.has(fn):
                results[i] = self.storage.load(fn)
                continue
            try:
                pending_images.append(PILImage.open(fn).convert("RGB"))
                pending_idx.append(i)
            except Exception as e:
                print(f"Error opening {fn}: {e}")
                results[i] = []

        if pending_images:
            batched_boxes = self.segmenter.segment_batch(pending_images)
            for idx, image, boxes in zip(pending_idx, pending_images, batched_boxes):
                results[idx] = self._embed_and_save(filenames[idx], image, boxes)

        return [r if r is not None else [] for r in results]

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
        non_noise = [c for c in cluster_index if c != -1]
        n_clusters = max(non_noise) + 1 if non_noise else 0

        clusters = [[] for _ in range(n_clusters)]
        # Noise points (-1) are collected into a single trailing cluster rather
        # than one singleton each, matching how the benchmark runner and the
        # extrinsic metrics treat noise (see methods.tex).
        noise: list[ImageData] = []
        for i, cluster in enumerate(cluster_index):
            if cluster == -1:
                noise.append(images[i])
            else:
                clusters[cluster].append(images[i])

        if noise:
            clusters.append(noise)

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
            get_model(EmbeddingModelNames.YOLOn),
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


